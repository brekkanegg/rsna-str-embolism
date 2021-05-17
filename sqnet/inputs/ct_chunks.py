# TODO: All Labels

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


class CTChunks(CTCommons):
    def __init__(self, args, transform, mode="train"):
        self.slice_num = args.slice_num
        self.slice_h = args.slice_num // 2

        super(CTChunks, self).__init__(args, transform, mode)
        print("Exam / Image nums: ", len(self.fold_df), len(self.loc_df))

    def __len__(self):
        if self.args.is_debugging:
            return 20

        return len(self.loc_df)

    def __getitem__(self, index):
        # ORDER: Resize --> Windowing --> Augmentation --> z normalization
        row = self.loc_df.iloc[index]
        sid, center_loc = row["sid"], row["center_loc"]
        if self.mode != "test":
            anns = [row["is_pos"]]
            h5_file_dir = DATA_DIR + "/train_{}/{}.h5".format(
                self.args.data_format, sid
            )
            with h5py.File(h5_file_dir, "r") as f:
                h5_file = np.array(
                    f["image"][
                        center_loc - self.slice_h : center_loc + self.slice_h + 1
                    ]
                )

        else:
            iid = row["iid"]
            h5_file_dir = DATA_DIR + "/test_{}/{}.h5".format(self.args.data_format, sid)

            with h5py.File(h5_file_dir, "r") as f:
                f_len = len(np.array(f["image"]))

                if center_loc - self.slice_h < 0:
                    pad_0_h5 = np.array(f["image"][0][np.newaxis])
                    pad_0_h5 = np.concatenate(
                        (pad_0_h5,) * (self.slice_h - center_loc), axis=0
                    )
                    h5_file = np.array(f["image"][0 : center_loc + self.slice_h + 1])
                    h5_file = np.concatenate((pad_0_h5, h5_file), axis=0)

                elif center_loc + self.slice_h > f_len - 1:
                    pad_1_h5 = np.array(f["image"][f_len - 1][np.newaxis])
                    pad_1_h5 = np.concatenate(
                        (pad_1_h5,) * (center_loc + self.slice_h - (f_len - 1)), axis=0
                    )
                    h5_file = np.array(f["image"][center_loc - self.slice_h : f_len])
                    h5_file = np.concatenate((h5_file, pad_1_h5), axis=0)

                else:
                    h5_file = np.array(
                        f["image"][
                            center_loc - self.slice_h : center_loc + self.slice_h + 1
                        ]
                    )

            assert self.slice_num == len(h5_file)

        img = h5_file[:, 32:480, 32:480]  # 24x448x448

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

        data = {"img": img}

        if self.mode != "test":
            data["fp"] = "{}_{}".format(sid, center_loc)
            data["anns"] = anns
        elif self.mode == "test":
            data["fp"] = "{}_{}_{}".format(sid, iid, center_loc)

        return data

    def get_loc_df(self, verbosity=0):

        if self.mode != "test":
            positive_locs = self.get_center_locs(is_pos=True, verbosity=verbosity)
            negative_locs = self.get_center_locs(is_pos=False, verbosity=verbosity)

            pos_locs = [
                [sid, cl]
                for sid in positive_locs.keys()
                for cl in positive_locs[sid]["center_locs"]
            ]
            if verbosity > 0:
                print("len pos locs: ", len(pos_locs))

            random.shuffle(pos_locs)

            neg_locs = [
                [sid, cl]
                for sid in negative_locs.keys()
                for cl in negative_locs[sid]["center_locs"]
            ]

            if verbosity > 0:
                print("len neg locs: ", len(neg_locs))

            random.shuffle(neg_locs)

            if self.mode == "train":
                pos_loc_df = pd.DataFrame(pos_locs, columns=["sid", "center_loc"])
                # NOTE: consider only center label
                pos_loc_df["is_pos"] = 1.0
                pos_loc_df["new_index"] = pos_loc_df.apply(
                    lambda row: row.name * 2, axis=1
                )
                pos_loc_df.set_index("new_index", inplace=True)

                assert len(neg_locs) > len(pos_locs)
                neg_locs = neg_locs[: len(pos_locs)]  # 1:1 비율 맞추기
                neg_loc_df = pd.DataFrame(neg_locs, columns=["sid", "center_loc"])
                neg_loc_df["is_pos"] = 0.0
                neg_loc_df["new_index"] = neg_loc_df.apply(
                    lambda row: row.name * 2 + 1, axis=1
                )
                neg_loc_df.set_index("new_index", inplace=True)

                tot_loc_df = pd.concat((pos_loc_df, neg_loc_df), axis=0).sort_index()

            elif "val" in self.mode:
                pos_loc_df = pd.DataFrame(pos_locs, columns=["sid", "center_loc"])
                pos_loc_df["is_pos"] = 1.0

                neg_loc_df = pd.DataFrame(neg_locs, columns=["sid", "center_loc"])
                neg_loc_df["is_pos"] = 0.0

                tot_loc_df = pd.concat((pos_loc_df, neg_loc_df), axis=0).sort_index()

        elif self.mode == "test":
            sids = self.fold_df

            id_locs = []
            for sid in sids:
                with open(DATA_DIR + "/annotations/test/{}.json".format(sid), "r") as f:
                    sid_annotation = json.load(f)

                center_ids = sid_annotation["ordered_ID"]

                id_locs.extend([[sid, iid, cl] for cl, iid in enumerate(center_ids)])
                tot_loc_df = pd.DataFrame(id_locs, columns=["sid", "iid", "center_loc"])

        return tot_loc_df

        # tot_locs = self.pos_locs + self.neg_locs
        # random.shuffle(tot_locs)

        # return tot_locs

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
            ids = self.fold_df[(self.fold_df["pe"] == "neg")]["StudyInstanceUID"]

        # NOTE: Temporary
        with open(os.path.dirname(__file__) + "/inconsistency.json", "r") as f:
            wrong_ids = json.load(f).keys()

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
                    # sample_num = int(np.ceil(len(center_locs) / self.slice_num))
                    sample_num = 1

                    center_locs = np.clip(
                        center_locs, self.slice_h, f_len - self.slice_h - 1
                    )
                    center_locs = np.random.choice(center_locs, sample_num)
                else:
                    # NOTE: average_PE_img_num = 43
                    # sample_num = int(np.ceil(30 / self.slice_num))
                    sample_num = 1
                    center_locs = np.random.randint(
                        self.slice_h, f_len - self.slice_h, sample_num
                    )

            chunk_dict[sid]["center_locs"] = center_locs

        if verbosity > 0:
            print("Time for getting center locs: {:.1f}s".format(time.time() - t0))

        return chunk_dict
