import cv2
import time
import numpy as np

from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    CLAHE,
    HueSaturationValue,
    RandomBrightnessContrast,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    OneOf,
    ToFloat,
    ShiftScaleRotate,
    GridDistortion,
    ElasticTransform,
    JpegCompression,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    Blur,
    MotionBlur,
    MedianBlur,
    GaussNoise,
    CenterCrop,
    IAAAdditiveGaussianNoise,
    GaussianBlur,
    OpticalDistortion,
    RandomSizedCrop,
    BboxParams,
)


# def augment_3d(img):

#     h, w = img.shape[0], img.shape[1]

#     aug = Compose(
#         [
#             # HorizontalFlip(p=0.1),
#             # VerticalFlip(p=0.1),
#             ShiftScaleRotate(
#                 shift_limit=0.05,
#                 scale_limit=0.05,
#                 rotate_limit=0.1,
#                 border_mode=cv2.BORDER_REPLICATE,
#                 p=1,
#             ),
#             # RandomSizedCrop(
#             #     min_max_height=(int(h * 0.9), h), height=h, width=w, p=0.25
#             # ),
#             # RandomBrightness(limit=0.1, p=0.5),
#             # GaussNoise(var_limit=0.001, mean=0., p=0.5)
#         ],
#         p=1,
#     )

#     augmented = aug(image=img)["image"]

#     return augmented


def augment_3d(img, input_channel=3):
    """
    input_channel==1 : img: slice_num x W x H
    input_channel==3 : img: 3 x slice_num x W x H
    
    """

    transform = Compose(
        [
            HorizontalFlip(p=0.1),
            ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.5,
            ),
            # RandomSizedCrop(min_max_height=(h-int(h*0.1), h), height=h, width=w, p=p),
            RandomBrightness(limit=0.2, p=0.5),
            GaussNoise(var_limit=0.001, mean=0.0, p=0.5),
        ],
        p=1,
        additional_targets={"image1": "image", "image2": "image"},
    )

    if input_channel == 1:
        img = img.transpose((1, 2, 0))
        augment = transform(image=img)
        img = augment["image"]
        img = img.transpose((2, 0, 1))

    elif input_channel == 3:

        # t0 = time.time()
        img0 = img[0].transpose((1, 2, 0))
        img1 = img[1].transpose((1, 2, 0))
        img2 = img[2].transpose((1, 2, 0))
        # print("transpose: ", time.time() - t0)

        # t0 = time.time()
        augment = transform(image=img0, image1=img1, image2=img2)
        # print("augment: ", time.time() - t0)

        # t0 = time.time()
        img0, img1, img2 = augment["image"], augment["image1"], augment["image2"]
        img = np.concatenate(
            (
                img0.transpose((2, 0, 1))[np.newaxis],
                img1.transpose((2, 0, 1))[np.newaxis],
                img2.transpose((2, 0, 1))[np.newaxis],
            ),
            axis=0,
        )
        # print("transpose2: ", time.time() - t0)

    return img
