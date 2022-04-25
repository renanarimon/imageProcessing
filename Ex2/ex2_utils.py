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
    k_size = normalize(k_size)  # normalize kernel
    np.flip(k_size)  # flip kernel
    return np.asarray([  # inner product of img and kernel
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
    border_size = int(np.floor(kernel.shape[0] / 2))
    ImgWithBorders = cv2.copyMakeBorder(  # pad img
        in_image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_REPLICATE,
    )
    ans = np.ndarray(in_image.shape)

    for i in range(in_image.shape[0]):  # inner product for each pixel with kernel
        for j in range(in_image.shape[1]):
            ans[i, j] = (np.sum(ImgWithBorders[i: i + kernel.shape[0], j: j + kernel.shape[1]] * kernel[:]))
    return ans


def convDerivative(in_image: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param in_image: Grayscale iamge
    :return: (directions, magnitude)
    """
    kernel = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])  # Derivative kernel
    Ix = conv2D(in_image, kernel)  # Derivative by x
    Iy = conv2D(in_image, kernel.T)  # Derivative by y

    mag = np.sqrt(pow(Ix, 2) + pow(Iy, 2)).astype(np.float64)
    direction = np.arctan2(Iy, Ix)

    return direction, mag


def create_gaussian_kernel(k_size: int):
    """
    convolve [1,1]*[1,1] till get k_size
    :param k_size:
    :return:
    """
    ones = np.ones(2, dtype=np.uint8)  # [1,1]
    kernel = np.ones(2, dtype=np.uint8)  # ans kernel
    for i in range(1, k_size - 1):
        kernel = conv1D(kernel, ones)
    kernel = kernel.reshape([k_size, -1])
    kernel = (kernel * kernel.T) / np.sum(kernel)  # make 2D & normalize
    return kernel


def blurImage1(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    gaussian = create_gaussian_kernel(k_size)
    return conv2D(in_image, gaussian)


def blurImage2(in_image: np.ndarray, k_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param k_size: Kernel size
    :return: The Blurred image
    """
    k = cv2.getGaussianKernel(k_size, 1)
    kernel = k * k.T

    return cv2.filter2D(in_image, -1, kernel)


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossing" method
    :param img: Input image
    :return: opencv solution, my implementation
    """

    pass


def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: my implementation
    """
    LoG = nd.gaussian_laplace(img, 2)  # log kernel
    ans_img = np.zeros_like(LoG)

    (h, w) = ans_img.shape
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if LoG[i, j] < 0:
                for x, y in neighbors:
                    if LoG[i + x, j + y] > 0:
                        ans_img[i, j] = 1
                        break
            elif LoG[i, j] > 0:
                for x, y in neighbors:
                    if LoG[i + x, j + y] < 0:
                        ans_img[i, j] = 1
                        break
            else:
                if (LoG[i, j - 1] > 0 and LoG[i, j + 1] < 0) \
                        or (LoG[i, j - 1] < 0 and LoG[i, j + 1] > 0) \
                        or (LoG[i - 1, j] > 0 and LoG[i + 1, j] < 0) \
                        or (LoG[i - 1, j] < 0 and LoG[i + 1, j] > 0):
                    ans_img[i, j] = 1
    return ans_img


def edgeDetectionZeroCrossingLOG1(img: np.ndarray) -> np.ndarray:
    """
    Detecting edges using "ZeroCrossingLOG" method
    :param img: Input image
    :return: my implementation
    """
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


def houghCircle1(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """

    img = cv2.GaussianBlur(img, (9, 9), 0)  # blur
    img = cv2.Canny((img * 255).astype(np.uint8), 255 / 3, 255)  # edges
    (h, w) = img.shape
    circles = []  # list of circles from all iterations
    for r in range(min_radius, max_radius + 1):
        accumulator = np.zeros((h, w))  # 2D arr to vote for the circles centers
        for x in range(h):
            for y in range(w):
                if img[x, y] == 255:  # edge pixel
                    for t in range(1, 361, 5):  # thetas
                        b = y - int(r * np.sin(t * np.pi / 180))
                        a = x - int(r * np.cos(t * np.pi / 180))
                        if 0 <= a < h and 0 <= b < w:  # if in borders
                            accumulator[a, b] += 1
        localMax(accumulator, circles, r)
    return circles



def localMax(accumulator: np.ndarray, circles: list, radius: int):
    """
    find the local maximums in the accumulator
    :param accumulator:
    :param circles:
    :param radius: curr radius
    :return: none
    """
    neighbors_size = (5, 5)
    threshold = np.max(accumulator) * 0.8
    tmp_max = filters.maximum_filter(accumulator, neighbors_size)  # makes all neighbors the maximum
    maxima = np.where(tmp_max == accumulator, tmp_max, 0)
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


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    try:
        edge_img = cv2.Canny((img * 255).astype(np.uint8), 50, 250)
        (h, w) = edge_img.shape
        edges = []
        circlesPoints = []
        circlesResult = []
        accumulator = {}
        threshold = 0.4
        thetas = 100

        for i in range(h):
            for j in range(w):
                if edge_img[i, j] == 255:
                    edges.append((i, j))

        for r in range(min_radius, max_radius + 1):
            for t in range(1, thetas):
                angle = (2 * np.pi * t) / thetas
                x = int(r * np.cos(angle))
                y = int(r * np.sin(angle))
                circlesPoints.append((x, y, r))

        for i, j in edges:
            for x, y, r in circlesPoints:
                a = j - y
                b = i - x
                vote = accumulator.get((a, b, r))
                if vote is None:
                    vote = 0
                accumulator[(a, b, r)] = vote + 1

        sortedAccumulator = sorted(accumulator.items(), key=lambda k: -k[1])
        for (x, y, r), s in sortedAccumulator:
            if s / 100 >= threshold and all((x - xc) ** 2 + (y - yc) * 2 > rc ** 2 for xc, yc, rc in circlesResult):
                circlesResult.append((x, y, r))
        return circlesResult
    except Exception as e:
        print(e)


def bilateral_filter_implement(in_image: np.ndarray, k_size: int, sigma_color: float, sigma_space: float) -> (
        np.ndarray, np.ndarray):
    """
    :param in_image: input image
    :param k_size: Kernel size
    :param sigma_color: represents the filter sigma in the color space.
    :param sigma_space: represents the filter sigma in the coordinate.
    :return: OpenCV implementation, my implementation
    """
    in_image = normalize(in_image)
    ans_img = np.zeros_like(in_image)  # changing filter -> cannot change the origin
    border_size = int(np.floor(k_size // 2))
    imgWithBorders = cv2.copyMakeBorder(  # pad image
        in_image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_REPLICATE,
    )
    if k_size % 2 != 0:  # k_size must be odd
        gaus = cv2.getGaussianKernel(k_size, 1)
    else:
        gaus = cv2.getGaussianKernel(k_size + 1, 1)
    gaus = gaus @ gaus.T

    (h, w) = in_image.shape
    for i in range(h):  # for each pixel apply filter
        for j in range(w):
            pivot_v = imgWithBorders[i, j]
            neighborhood = imgWithBorders[
                           i: i + k_size,
                           j: j + k_size
                           ]
            diff = pivot_v - neighborhood
            diff_gaus = np.exp(-np.power(diff, 2) / (2 * sigma_color))
            combo = gaus * diff_gaus
            ans_img[i, j] = (combo * neighborhood / combo.sum()).sum()  # insert to ans_img
    cvResult = cv2.bilateralFilter(in_image, k_size, sigmaColor=sigma_color,
                                   sigmaSpace=sigma_space)  # cv implementation

    return cvResult, ans_img
