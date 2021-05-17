# FIXME:

import os
import sys
import time
from tqdm import tqdm
import torch
import pickle
import numpy as np
import pandas as pd

# from apex import amp

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from inputs import chestxray, setup
from utils import visualize, tools  # , metrics
import networks

# from lib import metrics


class Testor:
    def __init__(self, args):

        self.args = args
        # self.save_dir = tools.set_save_dir(args)
        self.test_loader = chestxray.get_dataloader(self.args)
        self.visualizer = visualize.Visualizer(
            args, "test", save_dir=args.demo_save_dir
        )

        assert self.args.load_pretrained_ckpt is not None
        self.model = networks.get_model(self.args).cuda()

    def setup_load_pretrained_ckpt(self):
        print("\n\n>> Using pre-trained ckpt: " + self.args.load_pretrained_ckpt)
        checkpoint = torch.load(self.args.load_pretrained_ckpt)
        self.model.load_state_dict(checkpoint["model"], strict=True)

    def do_test(self):
        self.setup_load_pretrained_ckpt()

        self.model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            for (fp, img) in tqdm(self.test_loader):

                img = img.permute(0, 3, 1, 2).cuda()

                (pred_score_dict, pred_class_dict, pred_bbox_dict) = self.model(
                    img, mode="test"
                )[:3]

                for bi in range(len(fp)):
                    # bi_fp = fp[bi].split("png_1024/")[-1][:-4]
                    bi_fp = "/".join(fp[bi].split("demo/")[1:])[:-4]
                    bi_img = img[bi, 0, :, :].detach().cpu().numpy()
                    bi_pred_score = pred_score_dict[bi].cpu().numpy()
                    bi_pred_class = pred_class_dict[bi].cpu().numpy()
                    bi_pred_bbox = pred_bbox_dict[bi].cpu().numpy()
                    bi_pred = (bi_pred_score, bi_pred_class, bi_pred_bbox)

                    self.visualizer.save_demo_png(bi_fp, bi_img, bi_pred)
