import os
import sys
import torch
import torch.nn as nn
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
import inputs
from configs import *
import models


import h5py

# from opts.losses import FocalLoss
from scripts.common import Commoner


class Extractor(Commoner):
    def __init__(self, args):

        self.args = args

        self.train_loader, self.val_loader = inputs.get_dataloader(self.args)

        model = getattr(models, args.model)(args, use_pretrained=False)
        self.model = model.cuda()
        pretrained_model_dir = (
            HOME_DIR
            + "/ckpt/fold-{}_phase-image_model-ResNext_name-1018/epoch_{}.pt".format(
                self.args.fold, self.args.extract_epoch
            )
        )
        checkpoint = torch.load(pretrained_model_dir)
        self.model.load_state_dict(checkpoint["model"], strict=True)

    def do_extract(self, is_train=True, do_tqdm=False):

        if is_train:
            loader = self.train_loader
            feat_folder = "train"
        else:
            loader = self.val_loader
            feat_folder = "val"

        loader.dataset.loc_df.sort_values(by=["sid", "center_loc"], inplace=True)

        self.model.embedder.fc = nn.Identity()

        self.model.eval()  # batchnorm uses moving mean/variance instead of mini-batch mean/variance
        with torch.no_grad():

            current_sid = ""
            current_sorder = -1
            sid_feats = []
            npy_num = 0

            for data in tqdm(loader):
                fp = data["fp"]
                img = data["img"].cuda()

                feats = self.model(img)
                batch_feats = feats.detach().cpu().numpy()

                for bi in range(len(fp)):
                    sid, sorder = fp[bi].split("_")

                    feat_npy_dir = HOME_DIR + "/feat/fold{}/{}/{}_f2048.npy".format(
                        self.args.fold, feat_folder, current_sid
                    )

                    # transition
                    if sid != current_sid:
                        # save feats
                        if len(sid_feats) > 0:  # 처음 케이스가 아닐 때
                            sid_feats_npy = np.array(sid_feats)
                            np.save(feat_npy_dir, sid_feats_npy)
                            npy_num += 1

                        # 재 초기화
                        sid_feats = []
                        current_sid = sid
                        current_sorder = 0

                        sid_feats.append(batch_feats[bi])

                    # 같은 케이스
                    else:
                        assert int(sorder) == (current_sorder + 1)
                        current_sorder += 1

                        sid_feats.append(batch_feats[bi])

            # Last Case
            sid_feats_npy = np.array(sid_feats)
            np.save(feat_npy_dir, sid_feats_npy)
            npy_num += 1

        print("Finished, Generated feature npy file number is : ", npy_num)

