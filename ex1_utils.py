"""
author: renana rimon
"""
import sys
from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt

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
    try:
        img = cv2.imread(filename)
    except FileNotFoundError:
        print("Wrong file or file path")

    # BGR --> GRAY
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # BGR --> RGB
    elif representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise Exception("rep must be 0 or 1")

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
    if np.amax(y) <= 0:
        raise ZeroDivisionError()
    imgNorm255 = (y * (255.0 / np.amax(y))).astype(np.uint8)

    # calc imgOrg histogram
    histOrg, bins = np.histogram(imgNorm255, bins=256, range=[0, 255])

    # calc cumsum & normalize
    cumSum = np.cumsum(histOrg)
    try:
        cumSum = cumSum / np.max(cumSum)
        # calc LUT
        lut = np.array([np.ceil((cumSum[i] / np.max(cumSum)) * 255) for i in np.arange(0, 256)]).astype(np.uint8)
    except ZeroDivisionError:
        raise "ZeroDivisionError"

    # map img by LUT
    imgNew = cv2.LUT(imgNorm255, lut)

    # calc imEq histogram
    histEq, bins = np.histogram(imgNew, bins=256, range=[0, 255])

    # normalize: [0,255] --> [0,1]
    try:
        imgNew = imgNew / np.max(imgNew)
    except ZeroDivisionError:
        raise "ZeroDivisionError"

    # back to origin color
    if not gray:
        imgYIQ = transformRGB2YIQ(imgOrig)
        imgYIQ[:, :, 0] = imgNew
        imEq = transformYIQ2RGB(imgYIQ)
        return imEq, histOrg, histEq

    if gray:
        return imgNew, histOrg, histEq


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """

    counter = 0
    qImgList = []
    errorList = []
    # handle negative args
    if nQuant < 1 or nIter < 1:
        raise Exception

    # y: graySale img, gray: True ig imgOrig was gray
    y, gray = isGray(imOrig)

    # un-norm: [0,1] --> [0,255]
    if np.amax(y) <= 0:
        raise ZeroDivisionError()
    imgNorm255 = (y * (255.0 / np.amax(y))).astype(np.uint8)

    histOrg, bins = np.histogram(imgNorm255, bins=256, range=[0, 255])
    cumSum = np.cumsum(histOrg)

    # Z: set borders - uniform pixel num each interval
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

    # main loop
    for n in range(nIter):
        # q: weighted mean of each interval
        q = np.zeros(nQuant).astype(np.uint8)
        i = 0
        for j in range(nQuant):
            q[i] = int(np.average(range(z[j], z[j + 1]), weights=histOrg[z[j]:z[j + 1]]))
            i += 1

        # Z: set borders according to q
        for i in range(1, nQuant):
            x = int(q[i - 1]) + int(q[i])
            z[i] = x / 2

        # fill tmpImg by curr quantization colors
        tmpImg = np.zeros_like(y)
        for i in range(nQuant):
            tmpImg[(z[i] <= imgNorm255) & (imgNorm255 <= z[i + 1])] = q[i]
            i += 1

        # calc MSE - break if there is convergence
        mse = (np.sqrt((imgNorm255 - tmpImg) ** 2)).mean()
        if len(errorList) > 0 and (errorList[-1] - mse) < sys.float_info.epsilon:
            counter += 1
            if counter > 10:
                break
        elif counter > 0:
            counter -= 1
        errorList.append(mse)

        # normalize: [0,255] --> [0,1]
        tmpImg /= 255
        if not gray:  # back to origin color
            imgYIQ = transformRGB2YIQ(imOrig)
            currImg = transformYIQ2RGB(np.dstack((tmpImg, imgYIQ[:, :, 1], imgYIQ[:, :, 2])))
            qImgList.append(currImg)
        else:  # gray
            qImgList.append(tmpImg)

    return qImgList, errorList


def isGray(imgOrig: np.ndarray) -> (np.ndarray, bool):
    """
    if RGB -> convert to YIQ -> take Y
    elif GRAY -> Y = imgOrig
    :param imgOrig:
    :return: y: graySale img, gray: True ig imgOrig was gray
    """
    gray = True
    if len(imgOrig.shape) == 3:
        imgYIQ = transformRGB2YIQ(imgOrig)
        gray = False
        y = imgYIQ[:, :, 0]
    else:
        y = imgOrig
    return y, gray
