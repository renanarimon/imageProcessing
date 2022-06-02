import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimum and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    disparity_map = np.zeros((img_l.shape[0], img_l.shape[1]))

    # for each pixel in img_l, take a window to compare with img_r
    for x in range(k_size, img_l.shape[0] - k_size):
        for y in range(k_size, img_l.shape[1] - k_size):
            win_l = img_l[x - k_size:x + k_size + 1, y - k_size:y + k_size + 1]
            SSD = sys.maxsize  # curr pixel best SSD
            disparity = 0  # curr pixel best disparity

            # win_r is in the same pixels as win_l
            # shift win_r to left in given range
            # take the d that minimize the SSD
            for d in range(disp_range[0], disp_range[1]):
                ssd_tmp = 0
                if (y - k_size - d) >= 0 and (y + k_size + 1 - d) < img_r.shape[1]:
                    win_r = img_r[x - k_size:x + k_size + 1, y - k_size - d:y + k_size + 1 - d]
                    for u in range(k_size):
                        for v in range(k_size):
                            ssd_tmp += (win_l[u, v] - win_r[u, v]) ** 2
                if ssd_tmp < SSD:
                    SSD = ssd_tmp
                    disparity = d
            disparity_map[x, y] = disparity

    return disparity_map


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: The Maximum disparity range. Ex. 80
    k_size: Kernel size for computing the NormCorolation, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    disparity_map = np.zeros((img_l.shape[0], img_l.shape[1]))  # best disparity for each pixel

    # for each pixel in img_l, take a window to compare with img_r
    for x in range(k_size, img_l.shape[0] - k_size):
        for y in range(k_size, img_l.shape[1] - k_size):
            win_l_tmp = img_l[x - k_size:x + k_size + 1, y - k_size:y + k_size + 1]
            win_l = win_l_tmp.copy().flatten() - win_l_tmp.mean()
            norm1 = np.linalg.norm(win_l, 2)  # normalize win_l

            NCC = -1  # curr pixel best NCC
            disparity = 0  # curr pixel best disparity

            # win_r is in the same pixels as win_l
            # shift win_r to left in given range
            # take the d that maximize the NCC
            for d in range(disp_range[0], disp_range[1]):
                NCC_tmp = 0
                if (y - k_size - d) >= 0 and (y + k_size - d) < img_r.shape[1]:
                    win_r_tmp = img_r[x - k_size:x + k_size + 1, y - k_size - d:y + k_size + 1 - d]
                    win_r = win_r_tmp.copy().flatten() - win_r_tmp.mean()
                    norm2 = np.linalg.norm(win_r, 2)  # normalize win2
                    norms = norm1 * norm2  # ||win1|| * ||win2||
                    NCC_tmp = 0 if norms == 0 else np.sum(win_l * win_r) / norms  # correlation sum

                if NCC_tmp > NCC:
                    NCC = NCC_tmp
                    disparity = d
            disparity_map[x, y] = disparity

    return disparity_map


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
    Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
    returns the homography and the error between the transformed points to their
    destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

    src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
    dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

    return: (Homography matrix shape:[3,3], Homography error)
    """
    # create A - calibration matrix
    A = []
    for i in range(src_pnt.shape[0]):
        Xs, Ys = src_pnt[i]
        Xd, Yd = dst_pnt[i]
        A.append([Xs, Ys, 1, 0, 0, 0, -Xd * Xs, -Xd * Ys, -Xd])
        A.append([0, 0, 0, Xs, Ys, 1, -Yd * Xs, -Yd * Ys, -Yd])
    A = np.array(A)

    _, _, vh = np.linalg.svd(A)  # SVD
    H = (vh[-1].reshape((3, 3)) / vh[-1, -1])  # Homography matrix

    # make all matrix in right shape to calc error
    homo_src = np.vstack((src_pnt.T, np.ones(src_pnt.shape[0])))  # src_pnt.shape[0] x 2 --> 3 x src_pnt.shape[0]
    homo_dst = np.vstack((dst_pnt.T, np.ones(dst_pnt.shape[0])))
    pnt_new = H.dot(homo_src)  # trans src points with the Homography matrix

    error = np.sqrt(np.sum((pnt_new / pnt_new[-1] - homo_dst) ** 2))
    return H, error


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """
    dst_pts = []

    # function to display the coordinates of
    # of the points clicked on the image
    def click_event(event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            dst_pts.append((x, y))

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', dst_img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    dst_pts = np.array(dst_pts)

    ##### Your Code Here ######

    # out = dst_img * mask + src_out * (1 - mask)
    # plt.imshow(out)
    # plt.show()
