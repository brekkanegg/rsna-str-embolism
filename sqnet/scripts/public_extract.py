import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import cv2
from collections import OrderedDict
import h5py
import glob
import time
import pydicom as dicom
from tqdm import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix

from util import tools, visualizer
from metrics import metrics
import inputs
from configs import *
import models


# from opts.losses import FocalLoss
from scripts.common import Commoner


class CTData(torch.utils.data.Dataset):
    def __init__(self, path_folder, uids):
        self.path_folder = path_folder
        self.uids = uids

    def get_pixels_hu(self, slices):
        image = np.stack([s.pixel_array for s in slices])
        image = image.astype(np.int16)

        # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)

        image = np.array(image, dtype=np.int16)
        image[image < -1024] = -1024

        return image

    # def load_scan(self, path_to_dir, order_by_instance_number=False):

    #     dicom_paths = glob.glob(path_to_dir + "/*/*.dcm")
    #     dicom_slices = [dicom.read_file(x, force=True) for x in dicom_paths]

    #     zipped = list(zip(dicom_slices, dicom_paths))

    #     if order_by_instance_number:
    #         zipped.sort(key=lambda x: int(x[0].InstanceNumber))
    #     else:
    #         zipped.sort(key=lambda x: float(x[0].ImagePositionPatient[2]))

    #     dicom_slices = list(list(zip(*zipped))[0])
    #     dicom_paths = list(list(zip(*zipped))[1])

    #     return dicom_slices, dicom_paths

    def windowing(self, img, mode="pe"):
        if mode == "mediastinal":
            WL = 40
            WW = 450
        elif mode == "pe":
            WL = 100
            WW = 700
        elif mode == "lung":
            WL = -600
            WW = 1500

        upper, lower = WL + WW // 2, WL - WW // 2
        X = np.clip(img.copy(), lower, upper)

        X = X - lower
        X = X / (upper - lower)

        return X

    def preprocess(self, image):
        """
        image: dicom_pixels: Numpy Array
        """
        img = image[:, 32:480, 32:480]  # 24x448x448
        img0 = self.windowing(img, "mediastinal")[np.newaxis]
        img1 = self.windowing(img, "pe")[np.newaxis]
        img2 = self.windowing(img, "lung")[np.newaxis]
        img = np.concatenate((img0, img1, img2), axis=0)
        img = img.astype(np.float32)

        return img

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        # Load Input
        uid = self.uids[index]
        path_to_dcm = os.path.join(self.path_folder, self.uids[index])

        dps = glob.glob(path_to_dcm + "/*/*.dcm")
        dss = [dicom.read_file(x, force=True) for x in dps]

        zipped = list(zip(dss, dps))

        zipped.sort(key=lambda x: float(x[0].ImagePositionPatient[2]))

        dicom_slices = list(list(zip(*zipped))[0])
        dicom_paths = list(list(zip(*zipped))[1])

        # dicom_slices, dicom_paths = self.load_scan(path_to_dcm)  # , dicom_info \

        dicom_pixels = self.get_pixels_hu(dicom_slices)

        """
        uid : Patient ID
        dicom_pixels : Numpy Array
        dicom_paths : DICOM Paths (same oreder with dicom_pixels)
        """

        # FIXME: Is this Effective?
        pixel_arrays = self.preprocess(dicom_pixels)

        return uid, dicom_paths, pixel_arrays


class PublicExtractor(Commoner):
    def __init__(self, args):

        self.args = args

        # Dataloader
        df_test = pd.read_csv(DATA_DIR + "/test.csv")
        test_uids = list(df_test.StudyInstanceUID.drop_duplicates())
        ctdata = CTData(path_folder=DATA_DIR + "/test", uids=test_uids)
        self.loader = torch.utils.data.DataLoader(
            ctdata, shuffle=False, batch_size=1, num_workers=0, pin_memory=False
        )

        # Model
        model = getattr(models, args.model)(args, use_pretrained=False)
        pretrained_model_dir = (
            HOME_DIR
            + "/ckpt/fold-{}_phase-image_model-ResNext_name-1018/epoch_{}.pt".format(
                self.args.fold, self.args.extract_epoch
            )
        )
        checkpoint = torch.load(pretrained_model_dir)
        model.load_state_dict(checkpoint["model"], strict=True)

        if self.args.multi_gpu:
            model = torch.nn.DataParallel(model)

        self.model = model.cuda()

    def do_extract(self):

        self.model.embedder.fc = nn.Identity()

        self.model.eval()  # batchnorm uses moving mean/variance instead of mini-batch mean/variance
        with torch.no_grad():
            for data in tqdm(self.loader):

                sid, _, pixel_arrays = data
                sid = sid[0]
                pixel_arrays = np.array(pixel_arrays)[0]

                image_feats = []
                f_len = pixel_arrays.shape[1]
                batch_images = []
                for center_loc in range(f_len):
                    image = np.array(pixel_arrays[:, center_loc, :, :])

                    batch_images.append(image)
                    if (len(batch_images) == self.args.batch_size) or (
                        center_loc == (f_len - 1)
                    ):  # batch_case or last case
                        batch = torch.from_numpy(np.array(batch_images))
                        batch = batch.cuda()

                        result = self.model(batch)  # 2048 features

                        image_feats.extend(result.detach().cpu().numpy())

                        batch_images = []

                sid_feats_npy = np.array(image_feats)

                feat_npy_dir = HOME_DIR + "/publicfeat/fold{}/{}_f2048.npy".format(
                    self.args.fold, sid
                )
                np.save(feat_npy_dir, sid_feats_npy)

