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
    half = k_size // 2
    for x in range(half, img_l.shape[0] - half):
        for y in range(half, img_l.shape[1] - half):
            win_l = img_l[x - half:x + half + 1, y - half:y + half + 1]
            ssd = sys.maxsize
            disparity = 0
            for d in range(disp_range[0], disp_range[1]):
                ssd_tmp = 0
                if (y - half - d) >= 0 and (y + half + 1 - d) < img_r.shape[1]:
                    win_r = img_r[x - half:x + half + 1, y - half - d:y + half + 1 - d]
                    for u in range(k_size):
                        for v in range(k_size):
                            ssd_tmp += (win_l[u, v] - win_r[u, v]) ** 2
                if ssd_tmp < ssd:
                    ssd = ssd_tmp
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
    disparity_map = np.zeros((img_l.shape[0], img_l.shape[1]))
    half = k_size // 2
    for x in range(half, img_l.shape[0] - half):
        for y in range(half, img_l.shape[1] - half):
            win_l = img_l[x - half:x + half + 1, y - half:y + half + 1]
            NCC = -1
            disparity = 0
            for d in range(disp_range[0], disp_range[1]):
                NCC_tmp = 0
                R_lr = 0
                R_rr = 0
                R_ll = 0
                if (y - half - d) >= 0 and (y + half + 1 - d) < img_r.shape[1]:
                    win_r = img_r[x - half:x + half + 1, y - half - d:y + half + 1 - d]
                    for u in range(k_size):
                        for v in range(k_size):
                            R_lr += win_l[u, v] * win_r[u, v]
                            R_rr += win_r[u, v] * win_r[u, v]
                            R_ll += win_l[u, v] * win_l[u, v]
                    NCC_tmp = 0 if np.sqrt(R_rr, R_ll) == 0 else R_lr / np.sqrt(R_rr, R_ll)

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
    pass


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
    Displays both images, and lets the user mark 4 or more points on each image.
    Then calculates the homography and transforms the source image on to the destination image.
    Then transforms the source image onto the destination image and displays the result.

    src_img: The image that will be ’pasted’ onto the destination image.
    dst_img: The image that the source image will be ’pasted’ on.

    output: None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    ##### Your Code Here ######

    # out = dst_img * mask + src_out * (1 - mask)
    # plt.imshow(out)
    # plt.show()
