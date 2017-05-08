import numpy as np
import cv2


def equalize_histograms(images):
    ret = []
    for img in images:
        normalized = cv2.equalizeHist(np.array(img, dtype = np.uint8))
        ret.append(normalized)

    return ret


def clahe_equalized(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ret = []

    for img in images:
        normalized = clahe.apply(np.array(img, dtype = np.uint8))
        ret.append(normalized)

    return ret


def adjust_gamma(images, gamma=1.0):
    inv_gamma = 1. / gamma
    mapping   = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    ret = []

    for img in images:
        normalized = cv2.LUT(np.array(img, dtype=np.uint8), mapping)
        ret.append(normalized)

    return ret