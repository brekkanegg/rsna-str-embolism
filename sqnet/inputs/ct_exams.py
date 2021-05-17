import os, sys
import numpy as np
import pandas as pd
import glob
import cv2
import h5py
import random
from tqdm import tqdm
import json
import time

from .windowing import windowing
from .ct_common import CTCommons
from configs import *


random.seed(19920502)
np.random.seed(19920502)


# Only PE Case
# TODO: multi label case


class CTExams(CTCommons):
    def __init__(self, args, transform, mode="train"):
        super(CTExams, self).__init__(args, transform, mode)

        print("Example nums: ", len(self.fold_df))

    def __len__(self):
        if self.args.is_debugging:
            return 20

        return len(self.loc_df)

    def __getitem__(self, index):
        # ORDER: Resize --> Windowing --> Augmentation --> z normalization
        row = self.loc_df.iloc[index]
        sid = row["sid"]

        # Img
        # FIXME:

        h5_file_dir = DATA_DIR + "/train_{}/{}.h5".format(self.args.data_format, sid)
        with h5py.File(h5_file_dir, "r") as f:
            h5_file = np.array(f["image"])

        f_len = len(h5_file)

        img = h5_file[:, 32:480, 32:480]  # 24x448x448
        if f_len > 256:
            # ci = len(img) // 2
            if self.mode == "train":
                ci = np.random.randint(128, f_len - 128)
                img = img[ci - 128 : ci + 128, :, :]

        if self.args.image_size != 448:
            resized_img = np.zeros(
                (len(img), self.args.image_size, self.args.image_size)
            )
            for i in range(len(img)):
                resized_img[i] = cv2.resize(
                    img[i], (self.args.image_size, self.args.image_size)
                )
            img = resized_img

        if self.args.windowing == "pe":
            img = windowing(img, "pe")  # [0, 1]

            if self.transform is not None:
                img = img.astype(np.float32)
                img = self.transform(img, 1)  # WxHx24

            img = np.concatenate((img[np.newaxis],) * 3, axis=0)

        elif self.args.windowing == "3":
            img0 = windowing(img, "mediastinal")[np.newaxis]
            img1 = windowing(img, "pe")[np.newaxis]
            img2 = windowing(img, "lung")[np.newaxis]
            img = np.concatenate((img0, img1, img2), axis=0)

            if self.transform is not None:
                img = img.astype(np.float32)
                img = self.transform(img, 3)  # WxHx24

        img = img.astype(np.float32)

        # if self.args.phase == "lstm_end2end_exam":
        #     img = img.transpose((1, 0, 2, 3))

        # anns
        with open(DATA_DIR + "/annotations/train/{}.json".format(sid), "r") as f:
            sid_annotation = json.load(f)

        if self.args.phase == "exam_end2end":
            anns = np.zeros((f_len, 1))
            if len(sid_annotation["location"]) > 0:
                pe_idxs = [int(i) for i in np.array(sid_annotation["location"])[:, 1]]
                anns[pe_idxs, 0] = 1.0

            if self.args.loss_type == "rsna":
                pos_r = len(sid_annotation["location"]) / f_len
                data["pos_ratio"] = pos_r

        # elif self.args.phase == "lstm_pos":
        #     # 'Chronic', 'Acute & Chronic'
        #     anns = [sid_annotation["pe_type"][0], sid_annotation["pe_type"][1]]

        #     # 'central_pe', 'left_pe', 'right_pe'
        #     anns += [np.array(sid_annotation["location"])[0, 3]]
        #     anns += [np.array(sid_annotation["location"])[0, 2]]
        #     anns += [np.array(sid_annotation["location"])[0, 4]]

        #     # 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1'
        #     anns += [sid_annotation["pe_type"][0], 1 - sid_annotation["pe_type"][0]]

        # elif self.args.phase == "lstm_type":
        #     exam_keys = ["negative_exam_for_pe", "indeterminate"]
        #     anns = [sid_annotation[key] for key in exam_keys]

        if f_len > 256:
            if self.mode == "train":
                anns = anns[ci - 128 : ci + 128, :]

        data = {
            "fp": "{}".format(sid),
            "img": img,
            "anns": anns,
        }

        return data

    def get_loc_df(self, verbosity=0):

        with open(os.path.dirname(__file__) + "/inconsistency.json", "r") as f:
            wrong_ids = list(json.load(f).keys())

        if self.args.phase == "lstm_pos":  # use only positive
            pos_ids = self.fold_df[
                (self.fold_df["pe"] == "0")
                | (self.fold_df["pe"] == "1")
                | (self.fold_df["pe"] == "2")
            ]["StudyInstanceUID"]

            pos_loc_df = pd.DataFrame(pos_ids, columns=["sid"])
            tot_loc_df = pos_loc_df

        else:
            pos_ids = self.fold_df[
                (self.fold_df["pe"] == "0")
                | (self.fold_df["pe"] == "1")
                | (self.fold_df["pe"] == "2")
            ]["StudyInstanceUID"].reset_index(drop=True)
            pos_ids = [i for i in pos_ids if i not in wrong_ids]

            neg_ids = self.fold_df[(self.fold_df["pe"] == "neg")][
                "StudyInstanceUID"
            ].reset_index(drop=True)
            neg_ids = [i for i in neg_ids if i not in wrong_ids]

            random.shuffle(neg_ids)

            if self.mode == "train":
                pos_loc_df = pd.DataFrame(pos_ids, columns=["sid"])
                pos_loc_df["is_pos"] = 1.0
                pos_loc_df["new_index"] = pos_loc_df.apply(
                    lambda row: row.name * 2, axis=1
                )
                pos_loc_df.set_index("new_index", inplace=True)

                neg_ids = neg_ids[: len(pos_ids)]  # 1:1 비율 맞추기
                neg_loc_df = pd.DataFrame(neg_ids, columns=["sid"])
                neg_loc_df["is_pos"] = 0.0
                neg_loc_df["new_index"] = neg_loc_df.apply(
                    lambda row: row.name * 2 + 1, axis=1
                )
                neg_loc_df.set_index("new_index", inplace=True)

                tot_loc_df = pd.concat((pos_loc_df, neg_loc_df), axis=0).sort_index()

            elif "val" in self.mode:
                pos_loc_df = pd.DataFrame(pos_ids, columns=["sid"])
                pos_loc_df["is_pos"] = 1.0

                neg_loc_df = pd.DataFrame(neg_ids, columns=["sid"])
                neg_loc_df["is_pos"] = 0.0

                tot_loc_df = pd.concat((pos_loc_df, neg_loc_df), axis=0).sort_index()

        return tot_loc_df

