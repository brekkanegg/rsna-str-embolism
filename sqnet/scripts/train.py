import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import json

# from tensorboardX import SummaryWriter

from tqdm import tqdm

from scripts.validate import Validator  # , validator_cls
from util import tools, visualizer, logger, writer
import inputs
from inputs import augmentations


from configs import *
import models
import opts
from opts import losses


from scripts.common import Commoner


class Trainer(Commoner):
    def __init__(self, args):
        super(Trainer, self).__init__(args)

        #### 0. Setup
        self.save_dir = tools.set_save_dir(args)
        with open(os.path.join(self.save_dir, "args.json"), "w") as j:
            json.dump(vars(args), j)

        #### 1. Models
        model = getattr(models, args.model)(args)
        print(
            "Model param nums: ",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

        self.model = model.cuda()

        #### 2. Opt
        self.optimizer = opts.get_optimizer(args, self.model)
        self.scheduler = None
        if self.args.lr_scheduler is not None:
            self.scheduler = opts.get_scheduler(args, self.optimizer)

        #### 3. Data

        if args.augment is not None:
            augmentation = getattr(augmentations, args.augment)
        else:
            augmentation = None
        self.train_loader, self.val_loader = inputs.get_dataloader(
            args, transform=augmentation
        )

        #### 4. Logger
        self.writer = writer.Writer(log_dir=self.save_dir)
        self.logger = logger.Logger()
        self.logger.open(os.path.join(self.save_dir, "log.train.txt"), mode="a")
        self.logger.write("\n>> Pytorch version: {}".format(torch.__version__))
        self.logger.write("\n>> Args: {}".format(args))

        # Validator
        self.validator = Validator(
            args, is_trainval=True, writer=self.writer, val_loader=self.val_loader,
        )

    def do_train(self):

        self.setup_train()

        self.start_time = time.time()
        endurance = 0
        for epoch in range(self.resume_epoch, self.args.max_epoch):
            self.epoch = epoch

            if endurance > self.args.endurance:
                print("Stop training! No more performance gain expected!")
                print(
                    "Best saved at: ",
                    self.iteration,
                    self.epoch,
                    self.start_time,
                    self.save_dir,
                    self.tot_val_record["best"]["epoch"],
                )
                break

            self.train_one_epoch()

            if (epoch + 1) >= self.args.val_epoch:
                if (epoch + 1) % self.args.val_interval == 0:
                    val_record = self.validator.do_validate(
                        model=self.model, iteration=self.iteration
                    )

                    self.save_model(val_record, endurance)
                    self.val_log_and_write(val_record)

    def train_one_epoch(self):

        # Shuffle
        if self.epoch > 0:
            if not self.args.is_debugging:  # overfit test
                self.train_loader.dataset.loc_df = (
                    self.train_loader.dataset.get_loc_df()
                )

        self.optimizer.zero_grad()
        self.model.train()

        self.init_results()
        for i, data in enumerate(self.train_loader):

            fps = data["fp"]
            imgs = data["img"].cuda()  # .permute(0, 4, 1, 2, 3)
            anns = data["anns"].cuda()

            # self.scheduler_step(i)
            # if not (self.iteration % self.display_step == 0):

            outputs = self.model(imgs)
            if self.args.print_io:
                print("train inputs: ", imgs[0])
                print("train outputs: ", outputs[0])

            loss = self.calc_loss(fps, anns, outputs)

            if loss > 0:
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

            self.update_results(fps, anns, outputs, loss)
            self.train_log_and_write(i)

            if self.scheduler is not None:
                opts.step_scheduler(self.scheduler, global_step=self.iteration)

            self.iteration += 1

    def setup_resume(self):
        with open(os.path.join(self.save_dir, "tot_val_record.pkl"), "rb") as f:
            self.tot_val_record = pickle.load(f)

        self.iteration, self.resume_epoch = (
            self.tot_val_record["best"]["iteration"],
            self.tot_val_record["best"]["epoch"],
        )

        rep = str(self.resume_epoch)
        print("\nResume training from here: ", self.tot_val_record[rep])

        resume_model_dir = os.path.join(
            self.save_dir, "epoch_{}.pt".format(self.resume_epoch)
        )
        checkpoint = torch.load(resume_model_dir)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def setup_train(self):
        self.epoch = 0
        self.iteration = 0

        self.resume_epoch = 0

        self.tot_val_record = {
            "best": {
                "loss": np.inf,
                "comp_metric": np.inf,
                "precision": -1,
                "recall": -1,
                "f1": -1,
                "acc": -1,
                "epoch": -1,
            }
        }

        # FIXME:
        if self.args.resume_train:
            self.setup_resume()
            self.logger.write("\n\n** Resume Training Here! **")
            self.logger.write("\n>> Save Directory: {}\n".format(self.save_dir))

        else:
            self.logger.write("\n\n** Start Training Here! **")
            self.logger.write("\n>> Save Directory: {}\n\n".format(self.save_dir))

        print("\nStart Training\n")

        self.logger.set_header_columns(
            [
                "epoch",
                "iter",
                "time",
                "train_loss",
                "train_metric",
                "val_loss",
                "val_metric",
                "acc",
                "precision",
                "recall",
                "f1",
                "best_epoch",
            ]
        )
        self.logger.log_header()

        self.one_epoch_steps = len(self.train_loader)
        self.display_step = self.one_epoch_steps // self.args.display_interval

    def save_model(self, val_record, endurance):
        current = np.mean(val_record[self.args.best])
        prev_best = np.mean(self.tot_val_record["best"][self.args.best])

        model_improved = False
        if (self.args.best == "loss") or (self.args.best == "comp_metric"):
            if current < prev_best:
                model_improved = True
        else:
            if current > prev_best:
                model_improved = True

        if model_improved:

            checkpoint = {
                "epoch": self.epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            model_name = os.path.join(
                self.save_dir, "epoch_" + repr(self.epoch + 1) + ".pt"
            )
            torch.save(checkpoint, model_name)

            self.tot_val_record["best"] = val_record
            self.tot_val_record["best"]["epoch"] = self.epoch + 1
            self.tot_val_record["best"]["iteration"] = self.iteration

            endurance = 0
        else:
            endurance += 1

        self.tot_val_record[str(self.epoch + 1)] = val_record
        with open(os.path.join(self.save_dir, "tot_val_record.pkl"), "wb") as f:
            pickle.dump(self.tot_val_record, f)

        return endurance

    def train_log_and_write(self, i):

        result = self.get_results()

        take_time = tools.convert_time(time.time() - self.start_time)

        self.logger.log_result(
            [
                self.epoch,
                "{}/{}".format(i, self.one_epoch_steps),
                take_time,
                result["loss"],
                result["comp_metric"],
                "-",
                "-",
                result["acc"],
                result["precision"],
                result["recall"],
                result["f1"],
                "-",
            ]
        )

        self.writer.write_scalar(
            {"lr": self.optimizer.param_groups[0]["lr"]}, self.iteration
        )
        self.writer.write_scalars(
            {
                "statistics": {
                    "mean_{}".format(key): np.mean(value)
                    for key, value in zip(
                        ["acc", "precision", "recall", "f1"],
                        [
                            result["acc"],
                            result["precision"],
                            result["recall"],
                            result["f1"],
                        ],
                    )
                },
            },
            self.iteration,
        )

        self.writer.write_scalars(
            {"loss": {"train loss": result["loss"]}}, self.iteration,
        )
        self.writer.write_scalars(
            {"comp_metric": {"train comp metric": result["comp_metric"]}},
            self.iteration,
        )

    def val_log_and_write(self, val_record):
        take_time = tools.convert_time(time.time() - self.start_time)

        train_comp_metric = (self.comp_metric_loss.sum) / (
            self.comp_metric_weight.sum + 1e-15
        )

        self.logger.log_result(
            [
                self.epoch + 1,
                self.iteration,
                take_time,
                self.loss.avg,
                train_comp_metric,
                val_record["loss"],
                val_record["comp_metric"],
                val_record["acc"],
                val_record["precision"],
                val_record["recall"],
                val_record["f1"],
                self.tot_val_record["best"]["epoch"],
            ]
        )
        print("\r")
        self.writer.write_scalars(
            {
                "statistics": {
                    "mean_{}".format(key): np.mean(val_record[key])
                    for key in ["acc", "precision", "recall", "f1"]
                },
            },
            self.iteration,
        )
        self.writer.write_scalars(
            {"loss": {"val loss": val_record["loss"]}}, self.iteration,
        )
        self.writer.write_scalars(
            {"comp_metric": {"val comp metric": val_record["comp_metric"]}},
            self.iteration,
        )

