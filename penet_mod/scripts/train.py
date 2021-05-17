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
from inputs import cts as my_input  # , augmentations
from inputs import augmentations
from configs import *

import models
import opts
from opts import losses

# from opts.losses import FocalLoss


class Trainer:
    def __init__(self, args):
        self.args = args

        #### 0. Setup
        self.save_dir = tools.set_save_dir(args)
        with open(os.path.join(self.save_dir, "args.json"), "w") as j:
            json.dump(vars(args), j)

        #### 1. Data
        # TODO: augmentation
        augmentation = getattr(augmentations, args.augment)
        self.train_loader, self.val_loader = my_input.get_dataloader(
            args, transform=augmentation
        )

        #### 2. Model
        model = models.PENetClassifier(**vars(args))
        model.load_pretrained(PRETRAINED_WEIGHTS, "0")
        self.model = model.cuda()

        #### 3. Opt
        self.optimizer = opts.get_optimizer(args, self.model)
        self.scheduler = None
        if self.args.lr_scheduler is not None:
            self.scheduler = opts.get_scheduler(args, self.optimizer)

        #### 4. Logger
        self.writer = writer.Writer(log_dir=self.save_dir)
        self.logger = logger.Logger()
        self.logger.open(os.path.join(self.save_dir, "log.train.txt"), mode="a")
        self.logger.write("\n>> Pytorch version: {}".format(torch.__version__))
        self.logger.write("\n>> Args: {}".format(args))

        # self.visualizer = visualizer.Visualizer(
        #     args, "train", self.save_dir, self.writer
        # )

        # Validator
        self.validator = Validator(
            args, is_trainval=True, writer=self.writer, val_loader=self.val_loader,
        )

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
                "loss": -1,
                "precision": -1,
                "recall": -1,
                "f1": -1,
                "acc": 0,
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
                "val_loss",
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

    def do_train(self):

        self.setup_train()

        # print("\nStart Training!\n",)

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

            # print("precision / recall: ", pc, rc)

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

            fp = data["fp"]
            img = data["img"].cuda()  # .permute(0, 4, 1, 2, 3)
            anns = data["anns"].cuda()

            # self.scheduler_step(i)

            # if not (self.iteration % self.display_step == 0):

            outputs = self.model(img)

            if self.args.loss_type == "bce":
                criterion = torch.nn.BCEWithLogitsLoss()
            elif self.args.loss_type == "focal":
                criterion = losses.BinaryFocalLoss()

            loss = criterion(outputs, anns)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.update_results(anns, outputs, loss)
            self.train_log_and_write(i)

            # FIXME: visualizer
            # else:
            #     pass

            if self.scheduler is not None:
                opts.step_scheduler(self.scheduler, global_step=self.iteration)

            self.iteration += 1

        # self.last_loss = loss

    def save_model(self, val_record, endurance):
        if np.mean(val_record[self.args.best]) > np.mean(
            self.tot_val_record["best"][self.args.best]
        ):
            model_state_dict = self.model.state_dict()

            checkpoint = {
                "epoch": self.epoch,
                "model": model_state_dict,
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

    def init_results(self):
        self.tot_nums = tools.AverageMeter()
        self.loss = tools.AverageMeter()

        self.gt_nums = tools.AverageMeter()
        self.tp_nums = tools.AverageMeter()
        self.pred_nums = tools.AverageMeter()
        self.correct_nums = tools.AverageMeter()

    def update_results(self, anns, outputs, loss):
        gts = anns.detach().cpu().numpy()
        preds = (outputs.detach().cpu().numpy() > 0.5).astype(np.float32)

        self.tot_nums.update(len(gts))
        self.loss.update(loss.item())

        self.correct_nums.update(np.sum(gts == preds))
        self.gt_nums.update(np.sum(gts == 1))
        self.pred_nums.update(np.sum(preds))
        self.tp_nums.update(np.sum(gts * preds))

    def train_log_and_write(self, i):

        acc = self.correct_nums.sum / self.tot_nums.sum
        pc = self.tp_nums.sum / (self.pred_nums.sum + 1e-6)
        rc = self.tp_nums.sum / (self.gt_nums.sum + 1e-6)
        f1 = (2 * rc * pc) / (rc + pc + 1e-6)

        take_time = tools.convert_time(time.time() - self.start_time)

        self.logger.log_result(
            [
                self.epoch,
                "{}/{}".format(i, self.one_epoch_steps),
                take_time,
                self.loss.avg,
                "-",
                acc,
                pc,
                rc,
                f1,
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
                        ["acc", "precision", "recall", "f1"], [acc, pc, rc, f1]
                    )
                },
            },
            self.iteration,
        )

        self.writer.write_scalars(
            {"loss": {"train loss": self.loss.avg}}, self.iteration,
        )

    def val_log_and_write(self, val_record):
        take_time = tools.convert_time(time.time() - self.start_time)
        self.logger.log_result(
            [
                self.epoch + 1,
                self.iteration,
                take_time,
                self.loss.avg,
                val_record["loss"],
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

    # def scheduler_step(self, i):
    #     if self.epoch < self.args.warmup_epoch:
    #         for g in self.optimizer.param_groups:
    #             g["lr"] = (
    #                 self.args.learning_rate
    #                 / self.args.warmup_epoch
    #                 * (self.epoch + i / self.one_epoch_steps)
    #             )
    #     else:
    #         if not self.scheduler is None:
    #             self.scheduler.step(
    #                 self.epoch - self.args.warmup_epoch + i / self.one_epoch_steps
    #             )

