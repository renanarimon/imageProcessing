import math
from random import random

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import sklearn.preprocessing as sk

import scipy as sp
import scipy.ndimage as nd
import scipy.ndimage.filters as filters


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize Image in range (-1,1)
    :param img:
    :return:
    """
    if np.max(img) > 1:
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img


def conv1D(in_signal: np.ndarray, k_size: np.ndarray) -> np.ndarray:
    """
    Convolve a 1-D array with a given kernel
    In each iteration we will only multiply the relevant parts in each array
    :param in_signal: 1-D array
    :param k_size: 1-D array as a kernel
    :return: The convolved array
    """
    k_size = normalize(k_size)
    np.flip(k_size)  # flip kernel
    return np.asarray([
        np.dot(
            in_signal[max(0, i):min(i + len(k_size), len(in_signal))],
            k_size[max(-i, 0):len(in_signal) - i * (len(in_signal) - len(k_size) < i)],
        )
        for i in range(1 - len(k_size), len(in_signal))
    ])


def conv2D(in_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param in_image: 2D image
    :param kernel: A kernel
    :return: The convolved image
    """
    # np.flip(kernel)
    normalize(in_image)
    border_size = int(np.floor(kernel.shape[0] / 2))
    ImgWithBorders = cv2.copyMakeBorder(
        in_image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_REPLICATE,
    )
    ans = np.ndarray(in_image.shape)
    for i in range(in_image.shape[0]):
        for j in range(in_image.shape[1]):
            ans[i, j] = (np.sum(ImgWithBorders[i: i + kernel.shape[0], j: j + kernel.shape[0]] * kernel[:]))
    return ans


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    kernel = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])
    Ix = conv2D(in_image, kernel)
    Iy = conv2D(in_image, kernel.T)

    mag = np.sqrt(pow(Ix, 2) + pow(Iy, 2)).astype(np.float64)
    direction = np.arctan2(Iy, Ix)

    return direction, mag


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    gaussian = np.zeros(k_size * k_size).reshape((k_size, k_size))
    sigma = 0.3 * ((k_size - 1) * 0.5 - 1) + 0.8
    for x in range(0, k_size):
        for y in range(0, k_size):
            gaussian[x, y] = math.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) / (math.pi * (sigma ** 2) * 2)

    border_size = int(np.floor(k_size / 2))
    ImgWithBorders = cv2.copyMakeBorder(
        in_image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_REPLICATE,
    )
    return conv2D(ImgWithBorders, gaussian)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    k = cv2.getGaussianKernel(k_size, 0)
    kernel = k * k.T
    normalize(in_image)
    border_size = int(np.floor(k_size / 2))
    ImgWithBorders = cv2.copyMakeBorder(
        in_image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_REPLICATE,
    )
    c = cv2.filter2D(ImgWithBorders, -1, kernel)

    return c


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    pass


def edgeDetectionZeroCrossingLOG_over(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: my implementation
    """
    img = normalize(img)
    LoG = nd.gaussian_laplace(img, 2)
    threshold = np.absolute(LoG).mean() * 0.85
    output = np.zeros_like(LoG)
    w = output.shape[1]
    h = output.shape[0]

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if LoG[i, j] < 0:
                for x, y in neighbors:
                    if LoG[i + x, j + y] > 0:
                        output[i, j] = 1
                        break
            elif LoG[i, j] > 0:
                for x, y in neighbors:
                    if LoG[i + x, j + y] < 0:
                        output[i, j] = 1
                        break
            else:
                if (LoG[i, j - 1] > 0 and LoG[i, j + 1] < 0) \
                        or (LoG[i, j - 1] < 0 and LoG[i, j + 1] > 0) \
                        or (LoG[i - 1, j] > 0 and LoG[i + 1, j] < 0) \
                        or (LoG[i - 1, j] < 0 and LoG[i + 1, j] > 0):
                    output[i, j] = 1
    return output


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: my implementation
    """
    img = normalize(img)
    LoG_img = nd.gaussian_laplace(img, 2)
    threshold = np.absolute(LoG_img).mean() * 0.75
    output = np.zeros_like(LoG_img)
    (h, w) = output.shape

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            kernel = LoG_img[i - 1:i + 2, j - 1:j + 2]
            p = LoG_img[i, j]
            maxK = kernel.max()
            minK = kernel.min()
            if p > 0:
                zeroCross = True if minK < 0 else False
            elif p < 0:
                zeroCross = True if maxK > 0 else False
            else:
                zeroCross = True if (maxK > 0 and minK < 0) else False
            if ((maxK - minK) > threshold) and zeroCross:
                output[i, j] = 1
    return output


def edgeDetectionZeroCrossingLOG_my_lap(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: my implementation
    """
    gaussian = np.array([[1 / 16., 1 / 8., 1 / 16.], [1 / 8., 1 / 4., 1 / 8.], [1 / 16., 1 / 8., 1 / 16.]])
    laplacian = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    log = laplacian * gaussian
    img_log = cv2.filter2D(img, -1, log)

    (h, w) = img_log.shape
    border_size = 1
    imgWithBorders = cv2.copyMakeBorder(
        img_log,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_REPLICATE,
    )
    output = np.zeros_like(img_log)
    for i in range(1, h):
        for j in range(1, w):
            if imgWithBorders[i, j] < 0:
                for x, y in (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1):
                    if imgWithBorders[i + x, j + y] > 0:
                        output[i, j] = 1
                        break
            elif imgWithBorders[i, j] > 0:
                for x, y in (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1):
                    if imgWithBorders[i + x, j + y] < 0:
                        output[i, j] = 1
                        break
            else:
                if (imgWithBorders[i, j - 1] > 0 and imgWithBorders[i, j + 1] < 0) \
                        or (imgWithBorders[i, j - 1] < 0 and imgWithBorders[i, j + 1] > 0) \
                        or (imgWithBorders[i - 1, j] > 0 and imgWithBorders[i + 1, j] < 0) \
                        or (imgWithBorders[i - 1, j] < 0 and imgWithBorders[i + 1, j] > 0):
                    output[i, j] = 1
    return output


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = cv2.Canny((img * 255).astype(np.uint8), 255 / 3, 255)
    (h, w) = img.shape
    circles = []
    for r in range(min_radius, max_radius + 1):
        accumulator = np.zeros((h, w))
        for x in range(h):
            for y in range(w):
                if img[x, y] == 255:
                    for t in range(1, 361):
                        b = y - int(r * np.sin(t * np.pi / 180))
                        a = x - int(r * np.cos(t * np.pi / 180))
                        if 0 <= a < h and 0 <= b < w:
                            accumulator[a, b] += 1
        localMax(accumulator, circles, r)
    return circles


def localMax(accumulator: np.ndarray, circles: list, radius: int):
    neighbors_size = (5, 5)  # 5,5 / 4,4
    threshold = np.max(accumulator) * 0.8
    tmp_max = filters.maximum_filter(accumulator, neighbors_size)
    zeros = np.zeros_like(tmp_max)
    maxima = np.where(tmp_max == accumulator, tmp_max, zeros)
    tmp_min = filters.minimum_filter(accumulator, neighbors_size)
    diff = ((tmp_max - tmp_min) > threshold)
    maxima[diff == 0] = 0
    print("count: ", np.count_nonzero(maxima))

    (h, w) = accumulator.shape
    for i in range(h):
        for j in range(w):
            if maxima[i, j] > threshold:
                circles.append((j, i, radius))
    print("list: ", len(circles))


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    ans_img = np.zeros_like(in_image)
    border_size = int(np.floor(k_size / 2))
    imgWithBorders = cv2.copyMakeBorder(
        in_image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_REPLICATE,
    )
    if k_size % 2 != 0:
        gaus = cv2.getGaussianKernel(k_size, 1)
    else:
        gaus = cv2.getGaussianKernel(k_size + 1, 1)
    gaus = gaus@gaus.T

    (h, w) = in_image.shape
    for i in range(h):
        for j in range(w):
            pivot_v = imgWithBorders[i, j]
            neighborhood = imgWithBorders[
                         i: i + k_size,
                         j: j + k_size
                         ]
            diff = pivot_v - neighborhood
            diff_gaus = np.exp(-np.power(diff, 2) / (2 * sigma_color))
            combo = gaus * diff_gaus
            ans_img[i, j] = (combo * neighborhood / combo.sum()).sum()
    cvResult = cv2.bilateralFilter(in_image, k_size, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    return cvResult, ans_img
