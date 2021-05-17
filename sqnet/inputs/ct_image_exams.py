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


class CTImage_Exams(CTCommons):
    """
    exam_label + image_label 
    """

    def __init__(self, args, transform, mode="train"):

        super(CTImage_Exams, self).__init__(args, transform, mode)
        print("Exam / Image nums: ", len(self.fold_df), len(self.loc_df))

    def get_loc_df(self, verbosity=0):

        positive_locs = self.get_center_locs(is_pos=True, verbosity=verbosity)
        negative_locs = self.get_center_locs(is_pos=False, verbosity=verbosity)

        pos_locs = [
            [sid, cl]
            for sid in positive_locs.keys()
            for cl in positive_locs[sid]["center_locs"]
        ]

        if verbosity > 0:
            print(
                "len pos ids: {} /  locs: {}".format(
                    len(positive_locs.keys()), len(pos_locs)
                )
            )

        # random.shuffle(pos_locs)

        neg_locs = [
            [sid, cl]
            for sid in negative_locs.keys()
            for cl in negative_locs[sid]["center_locs"]
        ]

        if verbosity > 0:
            print(
                "len neg ids: {} /  locs: {}".format(
                    len(negative_locs.keys()), len(neg_locs)
                )
            )
            # print("len neg locs: ", len(neg_locs))

        # random.shuffle(neg_locs)
        if self.mode == "train":
            pos_loc_df = pd.DataFrame(pos_locs, columns=["sid", "center_loc"])
            pos_loc_df["is_pos"] = 1.0

            if self.args.posneg_ratio == 1:
                pos_loc_df["new_index"] = pos_loc_df.apply(
                    lambda row: row.name * 2, axis=1
                )
                pos_loc_df.set_index("new_index", inplace=True)
                assert len(neg_locs) > len(pos_locs)
                if self.args.run != "extract":
                    neg_locs = neg_locs[: len(pos_locs)]  # 1:1 비율 맞추기

            neg_loc_df = pd.DataFrame(neg_locs, columns=["sid", "center_loc"])
            neg_loc_df["is_pos"] = 0.0

            if self.args.posneg_ratio == 1:
                neg_loc_df["new_index"] = neg_loc_df.apply(
                    lambda row: row.name * 2 + 1, axis=1
                )
                neg_loc_df.set_index("new_index", inplace=True)

            tot_loc_df = pd.concat((pos_loc_df, neg_loc_df), axis=0).sort_index()
            tot_loc_df = tot_loc_df.sample(frac=1).reset_index(drop=True)

        elif "val" in self.mode:
            pos_loc_df = pd.DataFrame(pos_locs, columns=["sid", "center_loc"])
            pos_loc_df["is_pos"] = 1.0

            neg_loc_df = pd.DataFrame(neg_locs, columns=["sid", "center_loc"])
            neg_loc_df["is_pos"] = 0.0

            tot_loc_df = pd.concat((pos_loc_df, neg_loc_df), axis=0).sort_index()

        return tot_loc_df

    def get_center_locs(self, is_pos, verbosity=0):
        t0 = time.time()

        chunk_dict = {}
        if is_pos:
            ids = self.fold_df[
                (self.fold_df["pe"] == "0")
                | (self.fold_df["pe"] == "1")
                | (self.fold_df["pe"] == "2")
            ]["StudyInstanceUID"]
        else:
            # FIXME:
            # ids = self.fold_df[(self.fold_df["pe"] == "neg")]["StudyInstanceUID"]
            ids = self.fold_df["StudyInstanceUID"]

        # NOTE: Temporary
        with open(os.path.dirname(__file__) + "/inconsistency.json", "r") as f:
            wrong_ids = json.load(f).keys()

        if (self.args.run != "extract") or (self.args.run != "test"):
            ids = [i for i in ids if i not in wrong_ids]

        for sid in ids:
            chunk_dict[sid] = {}
            with open(DATA_DIR + "/annotations/train/{}.json".format(sid), "r") as f:
                sid_annotation = json.load(f)

            h5_file_dir = DATA_DIR + "/train_{}/{}.h5".format(
                self.args.data_format, sid
            )
            with h5py.File(h5_file_dir, "r") as f:
                f_len = len(f["image"])
                if is_pos:
                    center_locs = np.array(
                        [int(i) for i in np.array(sid_annotation["location"])[:, 1]]
                    )
                else:
                    # FIXME:
                    try:
                        pos_center_locs = list(
                            [int(i) for i in np.array(sid_annotation["location"])[:, 1]]
                        )
                        center_locs = np.array(
                            [cl for cl in range(f_len) if cl not in pos_center_locs]
                        )

                    except:
                        center_locs = np.array(range(f_len))

                    if self.args.posneg_ratio == 1:
                        if self.args.run != "extract":  # use all in test case
                            sample_num = 15
                            center_locs = np.random.choice(
                                center_locs, size=sample_num, replace=True
                            )

            chunk_dict[sid]["center_locs"] = center_locs

        if verbosity > 0:
            print("Time for getting center locs: {:.1f}s".format(time.time() - t0))

        return chunk_dict

    def __len__(self):
        if self.args.is_debugging:
            return 20

        if self.args.samples_per_epoch is not None:
            tot_len = min(self.args.samples_per_epoch, len(self.loc_df))
        else:
            tot_len = len(self.loc_df)

        return tot_len

    def __getitem__(self, index):
        t0 = time.time()
        # ORDER: Resize --> Windowing --> Augmentation --> z normalization
        row = self.loc_df.iloc[index]
        sid, center_loc, image_anns = row["sid"], row["center_loc"], [row["is_pos"]]

        image_anns = np.array(image_anns)[:, np.newaxis]
        is_pe_image = row["is_pos"]

        # Exam-level : 9
        with open(DATA_DIR + "/annotations/train/{}.json".format(sid), "r") as f:
            sid_annotation = json.load(f)

        exam_anns = [sid_annotation["negative_exam_for_pe"]]  # 0
        exam_anns += [sid_annotation["indeterminate"]]  # 1

        if (sid_annotation["pe_type"] == -1) or (not is_pe_image):
            # Negative case - no chronic or chronic & acute
            exam_anns += [0]
            exam_anns += [0]
        else:  # 2,3 - chronic, chronic_and_acute
            exam_anns += [sid_annotation["pe_type"][0]]
            exam_anns += [sid_annotation["pe_type"][1]]

        if (len(sid_annotation["location"]) > 0) and (
            is_pe_image
        ):  # 4,5,6 - central/left/right
            exam_anns += [int(np.array(sid_annotation["location"])[0, 3])]
            exam_anns += [int(np.array(sid_annotation["location"])[0, 2])]
            exam_anns += [int(np.array(sid_annotation["location"])[0, 4])]
        else:
            exam_anns += [0]
            exam_anns += [0]
            exam_anns += [0]

        if sid_annotation["rv_lv_ratio_gte_1"] == -1:
            exam_anns += [0]
            exam_anns += [0]
        else:  # 7,8 - rv_lv_rati_gte_1, lt_1
            exam_anns += [sid_annotation["rv_lv_ratio_gte_1"]]
            exam_anns += [1 - sid_annotation["rv_lv_ratio_gte_1"]]

        exam_anns = np.array(exam_anns)[:, np.newaxis]

        anns = np.concatenate((exam_anns, image_anns), axis=0)

        h5_file_dir = DATA_DIR + "/train_{}/{}.h5".format(self.args.data_format, sid)
        with h5py.File(h5_file_dir, "r") as f:
            h5_file = np.array(f["image"][center_loc])

        img = h5_file[32:480, 32:480]  # 24x448x448

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
                img = self.transform(img, 3)  # 3xWxH

        img = img.astype(np.float32)

        data = {
            "fp": "{}_{}".format(sid, center_loc),
            "img": img,
            "anns": anns,
        }

        # print("get one date time2: ", time.time() - t0)

        return data
