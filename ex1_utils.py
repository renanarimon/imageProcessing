"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import colorsys
import sys
from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 207616830


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE (1) or RGB (2)
    :return: The image object
    """
    img = cv2.imread(filename)

    # BGR --> GRAY
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # BGR --> RGB
    elif representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # scale [0,255] --> [0,1]
    # use info.max to ensure that the scale is accurate to the image
    info = np.iinfo(img.dtype)
    img = img.astype(np.float64) / info.max
    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    plt.imshow(img, cmap='gray')  # if img is gray, plot it gray
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    arrConvert = np.transpose(np.array([[0.299, 0.587, 0.114],
                                        [0.596, -0.275, -0.321],
                                        [0.212, -0.523, 0.311]]))

    return np.dot(imgRGB, arrConvert)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    arrConvert = np.transpose(np.array([[1, 0.956, 0.621],
                                        [1, -0.272, -0.647],
                                        [1, -1.106, 1.703]]))

    return np.dot(imgYIQ, arrConvert)


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # if imgOrig is RGB --> convert to YIQ and use only Y
    y, gray = isGray(imgOrig)

    # un-norm: [0,1] --> [0,255]
    imgNorm255 = (y * (255.0 / np.amax(y))).astype(np.uint8)

    # calc imgOrg histogram
    histOrg, bins = np.histogram(imgNorm255, bins=256, range=[0, 255])

    # calc cumsum & normalize
    cumSum = np.cumsum(histOrg)
    cumSum = cumSum / np.max(cumSum)

    # calc LUT
    LUT = makeLUT(cumSum)

    # map img by LUT
    imgNew = mapByLUT(LUT, imgNorm255)

    # calc imEq histogram
    histEq, bins = np.histogram(imgNew, bins=256, range=[0, 255])

    # normalize imEq
    imgNew = imgNew / np.max(imgNew)

    # transform img back to RGB
    if not gray:
        imgYIQ = transformRGB2YIQ(imgOrig)
        imgYIQ[:, :, 0] = imgNew
        imEq = transformYIQ2RGB(imgYIQ)
        return imEq, histOrg, histEq

    if gray:
        return imgNew, histOrg, histEq


def makeLUT(cumSum: np.ndarray) -> np.ndarray:
    """
    make LookUp Table (LUT):
        * index: pixel id
        * value: new intensity of this pixel
    ~should be linear~
    :param cumSum:
    :return:
    """
    lut = np.zeros(256)
    for pixel in range(256):
        lut[pixel] = np.ceil((cumSum[pixel] / np.max(cumSum)) * 255)
    return lut


def mapByLUT(LUT: np.ndarray, imgOrig: np.ndarray) -> np.ndarray:
    """
    map the new image by imgOrg and LUT.
    each pixel that have the i color in imgOrig,
    get the new color by LUT in imEq.
    :param LUT:
    :param imgOrig:
    :return: imEq
    """
    imEq = np.zeros_like(imgOrig, dtype=np.uint8)
    for i in range(256):
        imEq[imgOrig == i] = LUT[i]
    return imEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    qImgList = []
    errorList = []

    y, gray = isGray(imOrig)
    # un-norm: [0,1] --> [0,255]
    imgNorm255 = (y * (255.0 / np.amax(y))).astype(np.uint8)

    # set borders - lower=0, upper=255, and the other k-1 spread evenly (in terms of number of pixels)
    histOrg, bins = np.histogram(imgNorm255, bins=256, range=[0, 255])
    # plt.plot(range(256), histOrg)
    # plt.show()
    cumSum = np.cumsum(histOrg)
    z = np.zeros(nQuant + 1).astype(np.uint8)
    z[nQuant] = 255
    k = 1
    for i in range(255):
        if k == nQuant:
            break
        pixelNum = (k / nQuant) * cumSum[255]
        if cumSum[i] <= pixelNum <= cumSum[i + 1]:
            z[k] = i
            k += 1

    for n in range(nIter):
        # weighted mean
        q = np.zeros(nQuant).astype(np.uint8)
        i = 0
        for j in range(nQuant):
            q[i] = int(np.average(range(z[j], z[j + 1]), weights=histOrg[z[j]:z[j + 1]]))
            i += 1

        # set borders
        for i in range(1, nQuant):
            x = int(q[i - 1]) + int(q[i])
            z[i] = x / 2

        # fill tmpImg by curr quantization colors
        tmpImg = np.zeros_like(y)
        for i in range(nQuant):
            tmpImg[(z[i] <= imgNorm255) & (imgNorm255 <= z[i + 1])] = q[i]
            i += 1
        if not gray:
            imgYIQ = transformRGB2YIQ(imOrig)
            imgYIQ[:, :, 0] = tmpImg
            currImg = transformYIQ2RGB(imgYIQ)
            currImg = currImg / np.max(currImg)
            qImgList.append(currImg)
        else:
            # currImg = tmpImg / np.max(tmpImg)
            qImgList.append(tmpImg)

        # calc MSE
        mse = mean_squared_error(imgNorm255, tmpImg)

        errorList.append(mse)

    return qImgList, errorList


def isGray(imgOrig: np.ndarray) -> (np.ndarray, bool):
    gray = True
    if len(imgOrig.shape) == 3:
        imgYIQ = transformRGB2YIQ(imgOrig)
        gray = False
        y = imgYIQ[:, :, 0]
    else:
        y = imgOrig
    return y, gray
