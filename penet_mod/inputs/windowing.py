import numpy as np


def windowing(img, mode="pe"):
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
    X = X - np.min(X)
    X = X / np.max(X)
    X = X.astype("float32")
    # X = (X * 255.0).astype("uint8")

    return X

