import os
import sys
import torch
import pandas as pd
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
from scripts.common_all import CommonerALL


class TestorALL(CommonerALL):
    def __init__(self, args):

        super(TestorALL, self).__init__(args)
        print(self.args)

        # Model
        checkpoint = torch.load(self.args.load_pretrained_ckpt)
        args_path = self.args.load_pretrained_ckpt.split("/epoch_")[0] + "/args.json"
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

        # Data
        self.test_loader = inputs.get_dataloader(self.args)

    def do_test(self):

        outputs_dict = {}

        self.model.eval()
        with torch.no_grad():

            for data in tqdm(self.test_loader):
                fps = data["fp"]
                imgs = data["img"].cuda()  # .permute(0, 4, 1, 2, 3)
                anns = data["anns"]

                outputs = self.model(imgs)

                # NOTE: batch_size = 1
                logits = outputs[0, :, 0].detach().cpu().numpy()
                labels = anns[0, :, 0].numpy()
                outputs_dict[fps[0]] = {}
                outputs_dict[fps[0]]["logits"] = logits
                outputs_dict[fps[0]]["labels"] = labels

        fn = "./{}_fold{}_logits.pickle".format(self.args.model, self.args.fold)
        with open(fn, "wb") as f:
            pickle.dump(outputs_dict, f)

        print(len(outputs_dict.keys()))

