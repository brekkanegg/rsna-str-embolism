import os, sys
import numpy as np
import pandas as pd
import glob
import cv2
import h5py
import random
import json
import time

from .windowing import windowing
from .ct_common import CTCommons

from configs import *


random.seed(19920502)
np.random.seed(19920502)


# Only PE Case
# TODO: multi label case


class CTFeats(CTCommons):
    def __init__(self, args, transform, mode="train"):
        super(CTFeats, self).__init__(args, transform, mode)

        self.loc_df = self.get_loc_df(verbosity=1)

        if self.mode == "train":
            self.feat_folder = "feat/fold{}/train".format(self.args.fold)

        elif (self.mode == "trainval") or (self.mode == "val") or (self.mode == "test"):
            self.feat_folder = "feat/fold{}/val".format(self.args.fold)

        t0 = time.time()

        feat_dirs = glob.glob(DATA_DIR + "/{}/*_f2048.npy".format(self.feat_folder))

        # FIXME: Use PseudoLabels

        self.feats_dict = {fd: np.load(fd) for fd in feat_dirs}

        print("Time for loading all data: ", time.time() - t0)
        print("Example nums: ", len(self.fold_df))

    def get_loc_df(self, verbosity=0):

        with open(os.path.dirname(__file__) + "/inconsistency.json", "r") as f:
            wrong_ids = list(json.load(f).keys())

        pos_ids = self.fold_df[
            (self.fold_df["pe"] == "0")
            | (self.fold_df["pe"] == "1")
            | (self.fold_df["pe"] == "2")
        ]["StudyInstanceUID"].reset_index(drop=True)

        if self.args.run != "test":
            pos_ids = [i for i in pos_ids if i not in wrong_ids]
        else:
            pos_ids = list(pos_ids)

        # FIXME: Use 'indeterminate'
        neg_ids = self.fold_df[
            (self.fold_df["pe"] == "neg") | (self.fold_df["pe"] == "inter")
        ]["StudyInstanceUID"].reset_index(drop=True)

        if self.args.run != "test":
            neg_ids = [i for i in neg_ids if i not in wrong_ids]
        else:
            neg_ids = list(neg_ids)

        # random.shuffle(neg_ids)

        if self.mode == "train":
            pos_loc_df = pd.DataFrame(pos_ids, columns=["sid"])
            pos_loc_df["is_pos"] = 1.0
            if self.args.posneg_ratio == 1:
                pos_loc_df["new_index"] = pos_loc_df.apply(
                    lambda row: row.name * 2, axis=1
                )
                pos_loc_df.set_index("new_index", inplace=True)

                neg_ids = neg_ids[: len(pos_ids)]  # 1:1 비율 맞추기

            if not self.args.use_only_positive:
                neg_loc_df = pd.DataFrame(neg_ids, columns=["sid"])
                neg_loc_df["is_pos"] = 0.0
                if self.args.posneg_ratio == 1:
                    neg_loc_df["new_index"] = neg_loc_df.apply(
                        lambda row: row.name * 2 + 1, axis=1
                    )
                    neg_loc_df.set_index("new_index", inplace=True)
            else:  # use only positive cases
                neg_loc_df = pd.DataFrame()

            tot_loc_df = pd.concat((pos_loc_df, neg_loc_df), axis=0).sort_index()
            tot_loc_df = tot_loc_df.sample(frac=1).reset_index(drop=True)

        elif ("val" in self.mode) or ("test" in self.mode):
            pos_loc_df = pd.DataFrame(pos_ids, columns=["sid"])
            pos_loc_df["is_pos"] = 1.0

            neg_loc_df = pd.DataFrame(neg_ids, columns=["sid"])
            neg_loc_df["is_pos"] = 0.0

            tot_loc_df = pd.concat((pos_loc_df, neg_loc_df), axis=0).sort_index()
            tot_loc_df = tot_loc_df.sample(frac=1).reset_index(drop=True)

        return tot_loc_df

    def __len__(self):
        if self.args.is_debugging:
            return 20

        return len(self.loc_df)

    def __getitem__(self, index):
        # ORDER: Resize --> Windowing --> Augmentation --> z normalization
        row = self.loc_df.iloc[index]
        sid = row["sid"]

        feat_dir = DATA_DIR + "/{}/{}_f2048.npy".format(self.feat_folder, sid)

        # )
        # feat_dir = DATA_DIR + "/feat_2048/train/{}_f2048.npy".format(sid)
        # img = np.load(feat_dir)
        img = self.feats_dict[feat_dir]
        img = img.astype(np.float32)

        data = {"fp": sid, "img": img}

        # if self.args.run != "test":
        # anns
        with open(DATA_DIR + "/annotations/train/{}.json".format(sid), "r") as f:
            sid_annotation = json.load(f)

        if self.args.phase == "exam_feat":
            f_len = len(sid_annotation["ordered_ID"])
            anns = np.zeros((f_len, 1))
            if len(sid_annotation["location"]) > 0:
                pe_idxs = [int(i) for i in np.array(sid_annotation["location"])[:, 1]]
                anns[pe_idxs, 0] = 1.0

        elif self.args.phase == "exam_feat_all":

            # FIXME: Double-check label order

            # Exam-level : 9
            exam_anns = [sid_annotation["negative_exam_for_pe"]]
            exam_anns += [sid_annotation["indeterminate"]]

            if sid_annotation["pe_type"] == -1:
                # Negative case - no chronic or chronic & acute
                exam_anns += [0]
                exam_anns += [0]
            else:
                exam_anns += [sid_annotation["pe_type"][0]]
                exam_anns += [sid_annotation["pe_type"][1]]

            if len(sid_annotation["location"]) > 0:  # c/l/r
                exam_anns += [int(np.array(sid_annotation["location"])[0, 3])]
                exam_anns += [int(np.array(sid_annotation["location"])[0, 2])]
                exam_anns += [int(np.array(sid_annotation["location"])[0, 4])]
            else:
                exam_anns += [0]
                exam_anns += [0]
                exam_anns += [0]

            # FIXME: -1 case for indeterminate and negative
            if sid_annotation["rv_lv_ratio_gte_1"] == -1:
                exam_anns += [0]
                exam_anns += [0]
            else:
                exam_anns += [sid_annotation["rv_lv_ratio_gte_1"]]
                exam_anns += [1 - sid_annotation["rv_lv_ratio_gte_1"]]

            exam_anns = np.array(exam_anns)[:, np.newaxis]

            # Image-level : pe
            f_len = len(sid_annotation["ordered_ID"])
            image_anns = np.zeros((f_len, 1))
            if len(sid_annotation["location"]) > 0:
                pe_idxs = [int(i) for i in np.array(sid_annotation["location"])[:, 1]]
                image_anns[pe_idxs, 0] = 1.0

            anns = np.concatenate((exam_anns, image_anns), axis=0)

            """
            Negative for PE	0.0736196319
            Indeterminate	0.09202453988
            Chronic	0.1042944785
            Acute & Chronic	0.1042944785
            Central PE	0.1877300613
            Left PE	0.06257668712
            Right PE	0.06257668712
            RV/LV Ratio >= 1	0.2346625767
            RV/LV Ratio < 1	0.0782208589
            """

        data["anns"] = anns

        return data
