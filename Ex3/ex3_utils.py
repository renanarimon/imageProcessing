import math
import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError

from scipy import signal
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
    currImg = np.zeros(
        (pyr1[k - 2].shape[0], pyr1[k - 2].shape[1], 2))  # (m,n,2) zero array to put in u,v for each pixel
    lastImg = np.zeros((pyr1[k - 1].shape[0], pyr1[k - 1].shape[1], 2))

    points, uv = opticalFlow(pyr1[k - 1], pyr2[k - 1], stepSize, winSize)
    for j in range(len(points)):  # change pixels uv by formula
        y, x = points[j]
        u, v = uv[j]
        lastImg[x, y, 0] = u
        lastImg[x, y, 1] = v

    for i in range(k - 2, -1, -1):  # for each level of pyramids (small -> big)
        points, uv = opticalFlow(pyr1[i], pyr2[i], stepSize, winSize)  # uv for i'th img
        for j in range(len(points)):  # change pixels uv by formula
            y, x = points[j]
            u, v = uv[j]
            currImg[x, y, 0] = u
            currImg[x, y, 1] = v
        for z in range(lastImg.shape[0]):
            for r in range(lastImg.shape[1]):
                currImg[z * 2, r * 2, 0] += lastImg[z, r, 0] * 2
                currImg[z * 2, r * 2, 1] += lastImg[z, r, 1] * 2

        lastImg = currImg.copy()
        if i - 1 >= 0:
            currImg.fill(0)
            currImg.resize((pyr1[i - 1].shape[0], pyr1[i - 1].shape[1], 2))

    return currImg


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


# ------------------ Help functions ----------------------

def getWarpMatrix(method, theta, tx, ty) -> np.ndarray:
    """

    :param method: rigid / trans / rigid_opp
    :param theta: angle (for trans put 0)
    :param tx: move x
    :param ty: move y
    :return: correct warping matrix
    """
    if method == "rigid":
        return np.array([[math.cos(theta), -math.sin(theta), tx],
                         [math.sin(theta), math.cos(theta), ty],
                         [0, 0, 1]], dtype=np.float64)
    elif method == "trans":
        return np.array([[1, 0, tx],
                         [0, 1, ty],
                         [0, 0, 1]], dtype=np.float64)
    elif method == "rigid_opp":
        return np.array([[math.cos(theta), math.sin(theta), 0],
                         [-math.sin(theta), math.cos(theta), 0],
                         [0, 0, 1]], dtype=np.float64)


def findTheta(im1: np.ndarray, im2: np.ndarray) -> float:
    """
    find angle of rotation between im1 to im2
    :param im1:
    :param im2:
    :return: theta
    """
    min_mse = 1000
    theta = 0

    # find best angle
    for t in range(360):
        matrix_rigid = getWarpMatrix("rigid", t, 0, 0)
        curr_rigid_img = cv2.warpPerspective(im1, matrix_rigid, im1.shape[::-1])  # warp img
        mse = np.square(np.subtract(im2, curr_rigid_img)).mean()  # mse with curr angle
        if mse < min_mse:  # if this angle gave better result -> change
            min_mse = mse
            theta = t
    return theta


# ------------------ Translation & Rigid by LK ----------------------

def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    MSE_min = sys.maxsize
    points, uv = opticalFlow(im1, im2)  # get uv by LK
    (u1, v1) = uv[0]

    # find the best u,v --> minimize the MSE
    for (u, v) in uv:
        T = getWarpMatrix("trans", 0, u, v)
        trans_img = cv2.warpPerspective(im1, T, im1.shape[::-1])
        MSE = np.square(im2 - trans_img).mean()
        if MSE < MSE_min:
            MSE_min = MSE
            u1, v1 = u, v

    return getWarpMatrix("trans", 0, u1, v1)


def findRigid(im1: np.ndarray, im2: np.ndarray, method) -> np.ndarray:
    """
    Help function for  rigidLK & rigidCorr
    :param im1: origin img
    :param im2: rigid img
    :param method: translation Lk / corr
    :return: rigid matrix from im1 to im2
    """
    theta = findTheta(im1, im2)  # find theta
    matrix_rigid_opp = getWarpMatrix("rigid_opp", theta, 0, 0)  # matrix to rotate img back to origin
    img_back = cv2.warpPerspective(im2, matrix_rigid_opp, im2.shape[::-1])  # rotate img back to origin
    T = method(im1, img_back)  # find translation matrix

    return getWarpMatrix("rigid", theta, T[0, 2], T[1, 2])  # rigid matrix: translation+rotation


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    return findRigid(im1, im2, method=findTranslationLK)


# ------------------ Translation & Rigid by correlation ----------------------

def opticalFlowNCC(im1: np.ndarray, im2: np.ndarray, step_size, win_size):
    h = win_size // 2  # half of win
    uv = np.zeros((im1.shape[0], im1.shape[1], 2))  # for each pixel insert uv

    def Max_corr_idx(win: np.ndarray):
        """
        win1: img1 curr window (template)
        norm1: norm of win1
        (same for img2)

        NCC = (win1-mean(win1) * win2-mean(win2)) / (||win1|| * ||win2||)

        :param win:
        :return:
        """
        max_corr = -1000
        corr_idx = (0, 0)
        win1 = win.copy().flatten() - win.mean()
        norm1 = np.linalg.norm(win1, 2)  # normalize win1

        # correlate win1 with img2, and sum the corr in current window
        for i in range(h, im2.shape[0] - h - 1):
            for j in range(h, im2.shape[1] - h - 1):
                win2 = im2[i - h: i + h + 1, j - h: j + h + 1]  # get curr window from img2
                win2 = win2.copy().flatten() - win2.mean()
                norm2 = np.linalg.norm(win2, 2)  # normalize win2
                norms = norm1 * norm2  # ||win1|| * ||win2||
                corr = 0 if norms == 0 else np.sum(win1 * win2) / norms  # correlation sum

                # take the window that maximize the corr
                if corr > max_corr:
                    max_corr = corr
                    corr_idx = (i, j)  # top left pixels of curr window
        return corr_idx

    # each iteration take window from img2, and send to 'Max_corr_idx()' to find template matching
    for y in range(h, im1.shape[0] - h - 1, step_size):
        for x in range(h, im1.shape[1] - h - 1, step_size):
            template = im1[y - h: y + h + 1, x - h: x + h + 1]
            index = Max_corr_idx(template)  # index of best 'template matching' in img2
            uv[y - h, x - h] = np.flip(index - np.array([y, x]))

    return uv


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    take median u,v from 'opticalFlowNCC()'
    :param im1: origin img
    :param im2: translated img
    :return: translation matrix
    """
    uvs = opticalFlowNCC(im1, im2, 32, 13)  # get uv of all pixels
    u, v = np.median(np.masked_where(uvs == np.zeros(2), uvs), axis=(0, 1)).filled(0)  # take the median u,v
    return getWarpMatrix("trans", 0, u, v)


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    return findRigid(im1, im2, findTranslationCorr)


# ------------------ Warping ----------------------

def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    if im2.ndim == 3:  # RGB img
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    img_warp = np.zeros(im2.shape)
    TP = np.linalg.pinv(T)
    for i in range(im2.shape[0]):
        for j in range(im2.shape[1]):
            curr_idx = np.array([i, j, 1])  # curr index in new_img
            idx_orig = TP @ curr_idx  # this pixel index after rotation in im2
            x = (idx_orig[0] // idx_orig[2]).astype(int)  # back to 2D
            y = (idx_orig[1] // idx_orig[2]).astype(int)  # back to 2D

            if 0 <= x < im2.shape[0] and 0 <= y < im2.shape[1]:  # if index is in img range
                img_warp[i, j] = im2[x, y]  # insert pixel to new_img

    return img_warp


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
