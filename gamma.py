"""
author: renana rimon
"""
from __future__ import division
from __future__ import print_function

import cv2

import ex1_utils
import numpy as np

gamma_slider_max = 200
title_window = "gammaDisplay"


def getWidthHeight(image: np.ndarray) -> (int, int):
    """
    returns resized img width and height
    :param image:
    :return:
    """
    scale_percent = 60  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    return width, height


def gammaCorrect(val: int):
    """
    this func gets the desired gamma from trackbar, and by LUT makes the correct img.
    0 = dark, 200 = bright
    Vout = pow(Vin,(1/gamma))
    :param val: gamma val
    :return: correct img
    """
    g = float(val) / 100
    gamma = 100.0 if g <= 0 else 1.0 / g
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    imgCorrect = cv2.LUT(img, lut)
    width, height = getWidthHeight(imgCorrect)
    dim = (width, height)

    imS = cv2.resize(imgCorrect, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow(title_window, imS)


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img
    if rep == 1:  # grayscale
        img = cv2.imread(img_path, 2)
    else:  # RGB
        img = cv2.imread(img_path, 1)

    # create trackbar
    width, height = getWidthHeight(img)
    trackbar_name = 'GAMMA'
    cv2.namedWindow(title_window)
    cv2.resizeWindow(title_window, width, height)
    cv2.createTrackbar(trackbar_name, title_window, 100, gamma_slider_max, gammaCorrect)

    gammaCorrect(100)  # start val = 100
    cv2.waitKey()


def main():
    gammaDisplay('sinai.jpg', ex1_utils.LOAD_RGB)


if __name__ == '__main__':
    main()
