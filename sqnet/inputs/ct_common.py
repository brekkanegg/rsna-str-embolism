# TODO: All Labels

import torch
from torch.utils.data import Dataset
import os, sys
import numpy as np
import pandas as pd
import glob
import random
import json
import time

from .windowing import windowing
from configs import *


random.seed(19920502)
np.random.seed(19920502)


class CTCommons(Dataset):
    def __init__(self, args, transform, mode="train"):
        self.args = args
        self.transform = transform
        self.mode = mode

        self.fold_df = self.get_fold_df()
        self.loc_df = self.get_loc_df(verbosity=1)
        # self.update_fold_df_positive_ratios()

    def get_fold_df(self):

        # self.args.fold_json: temp_fold.json (5) or 7fold_final.json (5)
        with open(DATA_DIR + "/{}".format(self.args.fold_json), "r") as f:
            fold_info = json.load(f)

        if ("val" in self.mode) or (self.mode == "test"):
            fold_df = pd.DataFrame(
                fold_info["{}".format(self.args.fold)],
                columns=["StudyInstanceUID", "pe"],
            )

        else:
            if self.args.fold_json == "temp_fold.json":
                fold_nums = 5
            elif self.args.fold_json == "7fold_final.json":
                fold_nums = 7

            fold_list = list(range(fold_nums))
            fold_list.remove(self.args.fold)
            fold_df = pd.concat(
                [
                    pd.DataFrame(
                        fold_info["{}".format(i)], columns=["StudyInstanceUID", "pe"],
                    )
                    for i in fold_list
                ]
            )

        return fold_df

    def __len__(self):
        if self.args.is_debugging:
            return 20

        return len(self.loc_df)

    def __getitem__(self, index):
        pass
