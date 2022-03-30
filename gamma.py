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
from __future__ import division
from __future__ import print_function

import cv2

import ex1_utils
from ex1_utils import LOAD_GRAY_SCALE
import cv2 as cv
import ex1_utils as ex
import numpy as np
import argparse

gamma_slider_max = 200
title_window = "gammaDisplay"


def on_trackbar(val: int):
    g = float(val) / 100
    gamma = 100.0 if g == 0 else 1.0 / g
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)

    imgCorrect = cv2.LUT(img, lut)
    cv2.imshow(title_window, imgCorrect)


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img
    if rep == 1:
        img = cv2.imread(img_path, 2)
    else:
        img = cv2.imread(img_path, 1)

    cv2.namedWindow(title_window)
    trackbar_name = 'GAMMA'
    cv2.createTrackbar(trackbar_name, title_window, 100, gamma_slider_max, on_trackbar)
    on_trackbar(100)
    cv2.waitKey()



def main():
    gammaDisplay('beach.jpg', ex1_utils.LOAD_RGB)


if __name__ == '__main__':
    main()
