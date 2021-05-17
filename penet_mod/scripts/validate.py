import os
import sys
import torch
import numpy as np
import pickle
import cv2
from collections import OrderedDict

from tqdm import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix

from util import tools, visualizer
from metrics import metrics
from inputs import cts as my_input
from configs import *
import models
from opts import losses

# from opts.losses import FocalLoss


class Validator:
    def __init__(self, args, is_trainval=True, writer=None, val_loader=None):

        self.args = args
        self.is_trainval = is_trainval
        self.save_dir = tools.set_save_dir(args)
        self.writer = writer
        # self.prob_ths = prob_ths

        if is_trainval:
            self.val_loader = val_loader
        else:
            print(args)
            assert (writer is None) and (val_loader is None)
            self.val_loader = my_input.get_dataloader(self.args)

        self.visualizer = visualizer.Visualizer(args, "val", self.save_dir, self.writer)

    # def setup_load_pretrained_ckpt(self):
    #     print("\n\n>> Using pre-trained ckpt: " + self.args.load_pretrained_ckpt)
    #     checkpoint = torch.load(self.args.load_pretrained_ckpt)
    #     self.model.load_state_dict(checkpoint["model"], strict=True)

    def load_best_model(self):
        with open(os.path.join(self.save_dir, "tot_val_record.pkl"), "rb") as f:
            tot_val_record = pickle.load(f)

            best_record = tot_val_record["best"]
            best_epoch = best_record["epoch"]

            best_model_dir = os.path.join(
                self.save_dir, "epoch_{}.pt".format(best_epoch)
            )
            checkpoint = torch.load(best_model_dir)

            self.model.load_state_dict(checkpoint["model"], strict=True)
            print("\n>> Best model dir: {}".format(best_model_dir))
            print(best_record, "\n")

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

    def do_validate(self, model=None, iteration=None, do_tqdm=False):

        if not self.is_trainval:
            assert model is None

            model = models.PENetClassifier(**vars(args))
            if self.args.load_pretrained_ckpt:
                model.load_pretrained(PRETRAINED_WEIGHTS, args.gpu)
                self.model = model.cuda()

            else:
                self.load_best_model()

        else:  # trainval
            assert model is not None
            assert iteration is not None
            self.model = model  # Load train Model

        self.init_results()

        self.model.eval()  # batchnorm uses moving mean/variance instead of mini-batch mean/variance
        with torch.no_grad():

            tqdm_able = do_tqdm or (self.args.mode != "train")
            # or (self.args.is_debugging))

            for data in tqdm(self.val_loader, disable=(not tqdm_able)):

                fp = data["fp"]
                img = data["img"].cuda()  # .permute(0, 4, 1, 2, 3)
                anns = data["anns"].cuda()

                outputs = self.model(img)

                if self.args.loss_type == "bce":
                    criterion = torch.nn.BCEWithLogitsLoss()
                elif self.args.loss_type == "focal":
                    criterion = losses.BinaryFocalLoss()

                loss = criterion(outputs, anns)

                self.update_results(anns, outputs, loss)

        acc = self.correct_nums.sum / self.tot_nums.sum
        pc = self.tp_nums.sum / (self.pred_nums.sum + 1e-6)
        rc = self.tp_nums.sum / (self.gt_nums.sum + 1e-6)
        val_result = {
            "loss": self.loss.avg,
            "precision": pc,
            "recall": rc,
            "f1": (2 * rc * pc) / (rc + pc + 1e-6),
            "acc": acc,
        }

        return val_result

