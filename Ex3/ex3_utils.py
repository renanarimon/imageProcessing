import sys
from typing import List

import numpy as np
import cv2
from scipy import signal
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 100


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10,
                win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    points = []
    uv = []

    # RGB -> GRAY & normalize
    imGray1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY) if len(im1.shape) > 2 else im1
    imGray2 = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY) if len(im2.shape) > 2 else im2

    # kernels to derivative by x, y
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])

    # kernel to get t: I2 - I1
    kernel_t = np.array([[1., 1.], [1., 1.]]) * .25
    w = int(win_size / 2)

    # convolve img with kernel to derivative -> derivative img
    fx = signal.convolve2d(imGray1, kernel_x, boundary='symm', mode='same')
    fy = signal.convolve2d(imGray1, kernel_y, boundary='symm', mode='same')
    ft = signal.convolve2d(imGray1, kernel_t, boundary='symm', mode='same') + signal.convolve2d(imGray2, -kernel_t,
                                                                                                boundary='symm',
                                                                                                mode='same')

    # for each point, calculate Ix, Iy, It
    (rows, cols) = imGray1.shape
    for i in range(w, rows - w, step_size):
        for j in range(w, cols - w, step_size):
            # get the derivative in kernel location
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()

            Atb = [[-(Ix * It).sum()], [-(Iy * It).sum()]]
            AtA = [[(Ix * Ix).sum(), (Ix * Iy).sum()],
                   [(Ix * Iy).sum(), (Iy * Iy).sum()]]
            lambdas = np.linalg.eigvals(AtA)
            l1 = np.max(lambdas)
            l2 = np.min(lambdas)
            if l1 >= l2 > 1 and (l1 / l2) < 100:
                nu = np.matmul(np.linalg.pinv(AtA), Atb)  # (AtA)^-1 * Atb
                points.append([j, i])  # origin location
                uv.append([nu[0, 0], nu[1, 0]])  # new location
    return np.asarray(points), np.asarray(uv)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int,
                     stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    pass


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    pass


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    pass


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    pass


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    pass


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    # gaus = cv2.getGaussianKernel(5, sigma= 0.11)
    pyramid = []
    pyramid.append(img)
    for i in range(1, levels):
        tmp = cv2.GaussianBlur(pyramid[i-1], (5,5), 0.11)
        tmp = tmp[::2, ::2]
        pyramid.append(tmp)
    return pyramid


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gauss_pyr = gaussianPyr(img, levels)
    lap_pyr = []
    lap_pyr.append(gauss_pyr[levels-1])
    kernel = np.array([[1.],[ 2.], [1.]]) * 0.5
    kernel = kernel*kernel.T
    for i in range(levels-1, 0, -1):
        gaus = gauss_pyr[i]
        expand = np.zeros_like(gauss_pyr[i-1])  # blur!!
        expand = signal.convolve2d(expand, kernel, mode='same')
        for j in range(0, gaus.shape[0]):
            for k in range(0, gaus.shape[1]):
                expand[j*2, k*2] = gaus[j, k]
        lap = gauss_pyr[i-1] - expand
        lap_pyr.append(lap)
    lap_pyr.reverse()
    return lap_pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    up = cv2.pyrUp(lap_pyr)
    return up



def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    pass
