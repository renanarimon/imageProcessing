import numpy as np
import cv2
import scipy.ndimage as nd


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
    kernel = kernel * kernel.T  # make 2D
    return kernel / np.sum(kernel)  # normalize


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


def houghCircle(img: np.ndarray, min_radius: int, max_radius: int) -> list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    To find Edges you can Use OpenCV function: cv2.Canny
    :param img: Input image
    :param min_radius: Minimum circle radius
    :param max_radius: Maximum circle radius
    :return: A list containing the detected circles,
                [(x,y,radius),(x,y,radius),...]
    """
    try:
        edge_img = cv2.Canny((img * 255).astype(np.uint8), 50, 250)  # get edge img by canny
        (h, w) = edge_img.shape
        edges = []  # edge pixels
        circlesPoints = []  # (x, y, r): points on circles according to given radius
        circlesResult = []  # detected circles
        accumulator = {}  # vote for each pixel
        threshold = 0.4
        thetas = 100

        # save edge pixels
        for i in range(h):
            for j in range(w):
                if edge_img[i, j] == 255:
                    edges.append((i, j))

        # (x, y, r): points on circles according to given radius
        for r in range(min_radius, max_radius + 1):
            for t in range(1, thetas):
                angle = (2 * np.pi * t) / thetas
                x = int(r * np.cos(angle))
                y = int(r * np.sin(angle))
                circlesPoints.append((x, y, r))

        # vote for each pixel
        for i, j in edges:
            for x, y, r in circlesPoints:
                a = j - y
                b = i - x
                vote = accumulator.get((a, b, r))
                if vote is None:  # point has no votes yet
                    vote = 0
                accumulator[(a, b, r)] = vote + 1  # vote++

        sortedAccumulator = sorted(accumulator.items(), key=lambda k: -k[1])
        for (x, y, r), s in sortedAccumulator:
            if s / 100 >= threshold and all((x - x1) ** 2 + (y - y1) * 2 > r1 ** 2 for x1, y1, r1 in circlesResult):
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
    top_down = k_size
    left_right = k_size
    in_image = cv2.normalize(in_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im_pad = np.pad(in_image.astype(np.float32), ((top_down, top_down), (left_right, left_right)), 'edge')
    ans = np.zeros_like(in_image)
    for i in range(in_image.shape[0]):
        for j in range(in_image.shape[1]):
            pivot_v = in_image[i, j]
            neighbor_hood = im_pad[i: i + 2*k_size + 1, j: j + 2*k_size + 1]
            diff = pivot_v - neighbor_hood
            diff_gau = np.exp(-np.power(diff, 2) / (2 * sigma_color))
            gaus = cv2.getGaussianKernel(2 * k_size + 1, k_size)
            gaus = gaus*gaus.T
            combo = gaus * diff_gau
            result = (combo * neighbor_hood).sum() / combo.sum()
            ans[i, j] = result.sum()
    return cv2.bilateralFilter(in_image, k_size, sigma_color, sigma_space), ans