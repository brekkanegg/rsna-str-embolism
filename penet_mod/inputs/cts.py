# FIXME: Postivive / Negative


import torch
from torch.utils.data import Dataset, DataLoader, sampler
import os, sys
import numpy as np
import pandas as pd
import glob
import SimpleITK as sitk
import cv2
import h5py
import random
from tqdm import tqdm
import copy
import json
import time

from .windowing import windowing
from configs import *


random.seed(19920502)
np.random.seed(19920502)


def collater(data):

    fps = [s["fp"] for s in data]
    imgs = torch.FloatTensor([s["img"] for s in data])
    anns = torch.tensor([s["anns"] for s in data])

    collated_data = {"fp": fps, "img": imgs, "anns": anns}

    return collated_data


class PulmonaryEmbolismCTs(Dataset):
    def __init__(self, args, transform, mode="train"):
        self.args = args
        self.slice_num = args.slice_num
        self.slice_h = args.slice_num // 2
        self.transform = transform
        self.mode = mode

        self.fold_df = self.get_fold_df()

        # self.pos_locs, self.neg_locs = self.get_tot_locs(verbosity=1)
        self.loc_df = self.get_loc_df(verbosity=1)

        print("Example / Image nums: ", len(self.fold_df), len(self.loc_df))

    def get_fold_df(self):
        with open(DATA_DIR + "/temp_fold.json", "r") as f:
            fold_info = json.load(f)

        if "val" in self.mode:
            fold_df = pd.DataFrame(
                fold_info["{}".format(self.args.fold)],
                columns=["StudyInstanceUID", "pe"],
            )

        else:
            fold_list = list(range(5))
            fold_list.remove(self.args.fold)
            fold_df = pd.concat(
                [
                    pd.DataFrame(
                        fold_info["{}".format(i)], columns=["StudyInstanceUID", "pe"]
                    )
                    for i in fold_list
                ]
            )

        return fold_df

    def __len__(self):
        if self.args.is_debugging:
            return 20

        # if self.mode == "train":
        #     tot_len = len(self.pos_locs) + len(self.neg_locs)

        # elif "val" in self.mode:
        #     tot_len = len(self.pos_locs) + len(self.neg_locs)

        return len(self.loc_df)

    def __getitem__(self, index):
        t0 = time.time()
        # ORDER: Resize --> Windowing --> Augmentation --> z normalization
        row = self.loc_df.iloc[index]
        sid, center_loc, anns = row["sid"], row["center_loc"], [row["is_pos"]]

        h5_file_dir = DATA_DIR + "/train_{}/{}.h5".format(self.args.data_format, sid)
        with h5py.File(h5_file_dir, "r") as f:
            h5_file = np.array(
                f["image"][center_loc - self.slice_h : center_loc + self.slice_h]
            )

        img = h5_file[:, 32:480, 32:480]  # 24x448x448

        # FIXME:
        # img = img.transpose((1, 2, 0)) # 448x448x24

        if self.args.windowing == "pe":
            img = windowing(img, "pe")  # [0, 1]

            if self.transform is not None:
                img = self.transform(img, 1)  # WxHx24

            img = np.concatenate((img[np.newaxis],) * 3, axis=0)

        elif self.args.windowing == "3":
            img0 = windowing(img, "mediastinal")[np.newaxis]
            img1 = windowing(img, "pe")[np.newaxis]
            img2 = windowing(img, "lung")[np.newaxis]
            img = np.concatenate((img0, img1, img2), axis=0)

            if self.transform is not None:
                img = self.transform(img, 3)  # 24xWxH

        # Augmentation

        # z Normalization
        # img = img.astype(np.float32)
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))

        data = {
            "fp": "{}_{}".format(sid, center_loc),
            "img": img,
            "anns": anns,
        }

        # print("get one time: ", time.time() - t0)

        return data

    def get_loc_df(self, verbosity=0):

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
            pos_loc_df["is_pos"] = 1.0
            pos_loc_df["new_index"] = pos_loc_df.apply(lambda row: row.name * 2, axis=1)
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

                    center_locs = np.random.choice(center_locs, sample_num)
                    center_locs = np.clip(
                        center_locs, self.slice_h, f_len - self.slice_h
                    )
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


def get_dataloader(args, transform=None):

    # my_sampler = sampler.WeightedRandomSampler(dataset, [1, 1])
    # my_batch_sampler = sampler.BatchSampler(my_sampler, batch_size=args.batch_size)

    if args.mode == "train":
        train_loader = DataLoader(
            dataset=PulmonaryEmbolismCTs(args, transform=transform, mode="train"),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collater,
            drop_last=True,
        )

        val_loader = DataLoader(
            dataset=PulmonaryEmbolismCTs(args, transform=None, mode="trainval"),
            batch_size=args.batch_size * 2,
            num_workers=args.num_workers,
            collate_fn=collater,
            drop_last=True,
        )

        return (train_loader, val_loader)

    elif args.mode == "val":
        val_loader = DataLoader(
            dataset=PulmonaryEmbolismCTs(args, transform=None, mode="val"),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collater,
            drop_last=False,
        )

        return val_loader

