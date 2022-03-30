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

from ex1_utils import LOAD_GRAY_SCALE
import cv2 as cv
import ex1_utils as ex
import argparse

alpha_slider_max = 100
title_window = "gammaDisplay"


def on_trackbar(val, img1, img2):
    alpha = val / alpha_slider_max
    beta = (1.0 - alpha)
    dst = cv.addWeighted(img1, alpha, img2, beta, 0.0)
    cv.imshow(title_window, dst)


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    img = ex.imReadAndConvert(img_path, rep)
    cv.namedWindow(title_window)
    trackbar_name = 'Alpha x %d' % alpha_slider_max
    cv.createTrackbar(trackbar_name, title_window, 0, alpha_slider_max, on_trackbar)
    # Show some stuff
    on_trackbar(5, img, img)
    # Wait until user press some key
    cv.waitKey()
    pass


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
