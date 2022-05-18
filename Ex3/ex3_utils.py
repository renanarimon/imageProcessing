import sys
from typing import List

import numpy as np
import cv2
from scipy import signal
from numpy.linalg import LinAlgError
from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 207616830


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

    # kernel to get t: I2 - I1
    kernel_t = np.array([[1., 1.], [1., 1.]]) * .25
    s = int(win_size / 2)

    # convolve img with kernel to derivative
    fx = cv2.Sobel(im2, cv2.CV_64F, 1, 0, ksize=3,
                         borderType=cv2.BORDER_DEFAULT)
    fy = cv2.Sobel(im2, cv2.CV_64F, 0, 1, ksize=3,
                         borderType=cv2.BORDER_DEFAULT)
    ft = signal.convolve2d(imGray2, kernel_t, boundary='symm', mode='same') + signal.convolve2d(imGray1, -kernel_t,
                                                                                                boundary='symm',
                                                                                                mode='same')
    # for each point, calculate Ix, Iy, It
    # by moving the kernel over the image
    (rows, cols) = imGray1.shape
    for i in range(s, rows - s, step_size):
        for j in range(s, cols - s, step_size):
            # get the derivative in the kernel location
            Ix = fx[i - s:i + s + 1, j - s:j + s + 1].flatten()
            Iy = fy[i - s:i + s + 1, j - s:j + s + 1].flatten()
            It = ft[i - s:i + s + 1, j - s:j + s + 1].flatten()

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
    Ui = Ui + 2 ∗ Ui−1, Vi = Vi + 2 ∗ Vi−1
    """
    pyr1 = gaussianPyr(img1, k)  # gauss pyramid for img1
    pyr2 = gaussianPyr(img2, k)  # gauss pyramid for img2
    ans = np.zeros((img1.shape[0], img1.shape[1], 2))  # (m,n,2) zero array to put in u,v for each pixel
    for i in range(k, 0, -1):  # for each level of pyramids (small -> big)
        points, uv = opticalFlow(pyr1[i - 1], pyr2[i - 1], stepSize, winSize)  # uv for i'th img
        for j in range(len(points)):  # change pixels uv by formula
            y, x = points[j]
            u, v = uv[j]
            ans[x, y, 0] = 2 * ans[x, y, 0] + u  # Ui = Ui + 2 ∗ Ui−1
            ans[x, y, 1] = 2 * ans[x, y, 1] + v  # Vi = Vi + 2 ∗ Vi−1

    return ans


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    MSE_min = sys.maxsize
    points, uv = opticalFlow(im1, im2)
    (u1, v1) = uv[0]
    for (u, v) in uv:
        T = np.array([
            [1, 0, u],
            [0, 1, v],
            [0, 0, 1]
        ])

        trans_img = cv2.warpPerspective(im1, T, im1.shape[::-1])
        MSE = np.square(im2 - trans_img).mean()
        if MSE < MSE_min:
            MSE_min = MSE
            u1, v1 = u, v

    T = np.array([
        [1, 0, u1],
        [0, 1, v1],
        [0, 0, 1]
    ])

    return T


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
    maxCorr = 0
    (x, y) = (0, 0)
    for i in range(im2.shape[0]):
        for j in range(im2.shape[1]):
            T = im1[i:, j:]
            ft = fft(T)
            fa = fft(im2)
            Xcorr = ifft(ft * fa)
            cumSumA = np.cumsum(im2)
            cumSumA2 = np.cumsum(im2 ** 2)
            sigmaA = np.sqrt(cumSumA2 - (cumSumA ** 2) / len(T))
            sigmaT = np.sqrt(np.std(T) * (len(T) - 1))
            nXcorr = (Xcorr - cumSumA * np.mean(T)) / (sigmaT * sigmaA)
            if nXcorr > maxCorr:
                maxCorr = nXcorr
                (x, y) = (i, j)
    norm = np.sqrt(x ** 2 + y ** 2)
    u = x / norm
    v = y / norm
    T = np.array([
        [1, 0, u],
        [0, 1, v],
        [0, 0, 1]
    ])
    return T


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    pass


def warpImages1(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    x = np.arange(0, im2.shape[1])
    y = np.arange(0, im2.shape[0])

    x, y = np.meshgrid(y, x)
    x = x.flatten()
    y = y.flatten()
    z = np.ones_like(x)

    A = np.vstack((x, y, z))
    M = np.matmul(np.linalg.pinv(T), A)

    Ix = (M[0] / M[2]).astype(np.uint8).reshape(im1.shape)
    Iy = (M[1] / M[2]).astype(np.uint8).reshape(im1.shape)

    # imgNew = np.zeros_like(im1)
    imgNew = im1[Ix, Iy]

    i = 0
    for y in range(im1.shape[1]):
        for x in range(im1.shape[0]):
            s = Ix[i]
            r = Iy[i]
            imgNew[x, y] = im1[Ix[i], Iy[i]]
            i += 1
    return imgNew


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    tx = int(T[0, 2])
    ty = int(T[1, 2])
    new_img = np.zeros_like(im1)
    h, w = new_img.shape[0], new_img.shape[1]
    for i in range(w):
        for j in range(h):
            r = j - ty
            s = i - tx
            if 0 <= r < im1.shape[0] and 0 <= s < im1.shape[1]:
                new_img[j, i] = im1[r, s]
    return new_img


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------
def blurImage(in_image: np.ndarray, k_size: int, n: int) -> np.ndarray:
    k = cv2.getGaussianKernel(k_size, -1)
    kernel = k * k.T
    return cv2.filter2D(in_image, -1, kernel * n)


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pyramid = [img]
    for i in range(1, levels):
        tmp = blurImage(pyramid[i - 1], 5, 1)
        tmp = tmp[::2, ::2]
        pyramid.append(tmp)
    return pyramid


def expandImg(img: np.ndarray, newShape) -> np.ndarray:
    expand = np.zeros(newShape)
    expand[::2, ::2] = img
    return blurImage(expand, 5, 4)


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    gauss_pyr = gaussianPyr(img, levels)
    for i in range(levels - 1):
        gauss_pyr[i] = gauss_pyr[i] - expandImg(gauss_pyr[i + 1], gauss_pyr[i].shape)
    return gauss_pyr


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    img = lap_pyr[-1]
    for i in range(len(lap_pyr) - 1, 0, -1):
        expand = expandImg(img, lap_pyr[i - 1].shape)
        img = expand + lap_pyr[i - 1]
    return img


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
    naive = mask * img_1 + (1 - mask) * img_2

    lapPyr_img1 = laplaceianReduce(img_1, levels)
    lapPyr_img2 = laplaceianReduce(img_2, levels)
    gaussPyr_mask = gaussianPyr(mask, levels)

    mergeN = lapPyr_img1[-1] * gaussPyr_mask[-1] + (1 - gaussPyr_mask[-1]) * lapPyr_img2[-1]
    for i in range(levels - 1, 0, -1):
        expand = expandImg(mergeN, lapPyr_img1[i - 1].shape)
        mergeN = expand + lapPyr_img1[i - 1] * gaussPyr_mask[i - 1] + (1 - gaussPyr_mask[i - 1]) * lapPyr_img2[i - 1]

    return naive, mergeN
