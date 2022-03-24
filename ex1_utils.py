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
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
