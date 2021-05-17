import os
import sys
import torch
import pandas as pd
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
from opts import losses

# from opts.losses import FocalLoss
from scripts.common import Commoner


class Testor(Commoner):
    def __init__(self, args):

        self.args = args
        if self.args.test_save_dir is None:
            self.save_dir = tools.set_save_dir(args)
        else:
            self.save_dir = self.args.test_save_dir

        self.test_loader = inputs.get_dataloader(self.args)

        model = getattr(models, self.args.model)(self.args)
        self.model = model.cuda()
        self.load_best_model()

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

    def do_test(self):

        result_df = []

        self.model.eval()
        with torch.no_grad():

            for data in tqdm(self.test_loader):

                fp = data["fp"]
                img = data["img"].cuda()  # .permute(0, 4, 1, 2, 3)

                outputs = self.model(img)

                if self.args.phase == "lstm_end2end_chunk":
                    outputs = torch.max(outputs, dim=1)[0]

                for i in range(len(fp)):
                    sid, iid = fp[i].split("_")[:2]
                    pe_present_on_image = float(
                        torch.sigmoid(outputs).detach().cpu().numpy() > 0.5
                    )
                    result_df.append([sid, iid, pe_present_on_image])

                # TODO: make result dataframe

        result_df = pd.DataFrame(
            result_df, columns=["sid", "iid", "pe_present_on_image"]
        )

        result_df.to_csv(
            "submission/phase-{}_fold-{}_submit.csv".format(
                self.args.phase, self.args.fold
            )
        )

