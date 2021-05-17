import os
import sys
import torch
import numpy as np
import pickle
import cv2
import json
from collections import OrderedDict, namedtuple
from argparse import Namespace

from tqdm import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix

from util import tools, visualizer
from metrics import metrics
import inputs
from configs import *
import models
from opts import losses

# from opts.losses import FocalLoss
from scripts.common import Commoner


class Validator(Commoner):
    def __init__(self, args, is_trainval=True, writer=None, val_loader=None):

        super(Validator, self).__init__(args)

        self.is_trainval = is_trainval
        self.save_dir = tools.set_save_dir(args)
        self.writer = writer

        if is_trainval:
            self.val_loader = val_loader
        else:
            self.val_loader = inputs.get_dataloader(self.args)
            assert (writer is None) and (val_loader is None)
            print(self.args)

        self.visualizer = visualizer.Visualizer(args, "val", self.save_dir, self.writer)

    def load_model(self):
        if self.args.load_pretrained_ckpt is not None:
            checkpoint = torch.load(self.args.load_pretrained_ckpt)
            args_path = (
                self.args.load_pretrained_ckpt.split("/epoch_")[0] + "/args.json"
            )
            with open(args_path, "rb") as j:
                load_args = json.load(j)

            overwrite_args = vars(self.args).copy()
            for k in load_args.keys():
                if k in overwrite_args.keys():
                    overwrite_args[k] = load_args[k]

            load_args = Namespace(**overwrite_args)
            model = getattr(models, load_args.model)(load_args, use_pretrained=False)
            self.model = model.cuda()
            self.model.load_state_dict(checkpoint["model"], strict=True)

        # else:
        #     with open(os.path.join(self.save_dir, "tot_val_record.pkl"), "rb") as f:
        #         tot_val_record = pickle.load(f)

        #     best_record = tot_val_record["best"]
        #     best_epoch = best_record["epoch"]

        #     best_model_dir = os.path.join(
        #         self.save_dir, "epoch_{}.pt".format(best_epoch)
        #     )
        #     checkpoint = torch.load(best_model_dir)
        #     print("\n>> Best model dir: {}".format(best_model_dir))
        #     print(best_record, "\n")

    def do_validate(self, model=None, iteration=None, do_tqdm=False):

        if not self.is_trainval:
            assert model is None
            self.load_model()

        else:  # trainval
            assert model is not None
            assert iteration is not None
            self.model = model  # Load train Model

        self.init_results()

        self.model.eval()
        with torch.no_grad():

            tqdm_able = do_tqdm or (self.args.run != "train")

            for data in tqdm(self.val_loader, disable=(not tqdm_able)):

                fps = data["fp"]
                imgs = data["img"].cuda()  # .permute(0, 4, 1, 2, 3)
                anns = data["anns"].cuda()

                outputs = self.model(imgs)
                if self.args.print_io:
                    if self.args.phase == "exam_feat":
                        print("val inputs: ", imgs[0, :10, :])
                        print("val outputs: ", torch.sigmoid(outputs[0, :10, 0]))
                    elif self.args.phase == "image":
                        print("val inputs: ", imgs[:10])
                        print("val outputs: ", torch.sigmoid(outputs[:10, 0]))

                loss = self.calc_loss(fps, anns, outputs)
                self.update_results(fps, anns, outputs, loss)

        val_result = self.get_results()

        if not self.is_trainval:
            print(val_result)

        else:
            return val_result
