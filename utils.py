import numpy as np
import cv2
import copy

def psnr(img1, img2):
    """
    Computes psnr.
    :param img1: first img
    :param img2: second img
    :return: psnr
    """
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def entropy(array):
    """
    Computes entropy.
    """
    len_arr = array.shape[0]

    counts = np.bincount(array)
    probs = counts / len_arr

    probs[probs == 0] = 10e-10
    entropy = 0.
    for i in probs:
        entropy -= i * np.log2(i)

    return entropy


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return ycbcr

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return rgb

class jpeg():
    def __init__(self, dir, N=8, n=7, percentage=0):
        super(jpeg, self).__init__()
        self.img = self.get_img_matrix(dir)
        self.N = N
        self.DCT_matrix = self.get2D_DCT_matrix(N)

        self.img_DCT = self.do_full_DCT_transform(self.img, N)
        self.img_rev_DCT = self.do_full_rev_DCT_transform(self.img_DCT, N)

        self.img_DCT_after_simple_compress = self.simple_compression(self.img_DCT, n, N)
        self.img_after_simple_compress = self.do_full_rev_DCT_transform(self.img_DCT_after_simple_compress, N)

        self.quantization_matrix = self.get_quantization_matrix(percentage)
        self.DCT_after_full_quantization = self.full_dequantization(self.full_quantization(self.img_DCT, self.quantization_matrix, N), self.quantization_matrix, N)
        self.img_after_ok_compression = self.do_full_rev_DCT_transform(self.DCT_after_full_quantization, N)


    def get_quantization_matrix(self, percentage):
        """
        https://habr.com/en/post/206264/
        get jpeg quantization matrix
        :param percentage (int [0, 100]): level of compression
        :return: jpeg quantization matrix
        """
        # quantization_matrix_50 = [
        #                             [16, 11, 10, 16, 24, 40, 51, 61],
        #                             [12, 12, 14, 19, 26, 58, 60, 55],
        #                             [14, 13, 16, 24, 40, 57, 69, 56],
        #                             [14, 17, 22, 29, 51, 87, 80, 62],
        #                             [18, 22, 37, 56, 68, 109, 103, 77],
        #                             [24, 35, 55, 64, 81, 104, 113, 92],
        #                             [49, 64, 78, 87, 103, 121, 120, 101],
        #                             [72, 92, 95, 98, 112, 100, 103, 99],
        #                          ]

        quantization_matrix_50 = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                quantization_matrix_50[i, j] = (i + j)*percentage + 1

        # quantization_matrix_50 = np.array(quantization_matrix_50)

        #print(quantization_matrix_50.shape)
        def clip(quantization_matrix):
            quantization_matrix[quantization_matrix > 255] = 255
            quantization_matrix[quantization_matrix < 1] = -1
            return quantization_matrix

        return clip(quantization_matrix_50)

    def get2D_DCT_matrix(self, N):
        """
        #https://www.youtube.com/watch?time_continue=1&v=tW3Hc0Wrgl0&feature=emb_logo
        get DCT transform matrix
        :param N (int): size of jpeg bbox
        :return: NxN 2d DCT matrix
        """

        matrix = np.zeros((N,N))
        matrix[0,:] = np.sqrt(1/N)
        v_idxs = np.array(range(N))
        for u, u_axis in enumerate(matrix):
            if u == 0:
                continue

            matrix[u, :] = np.sqrt(2/N) * np.cos( ((2*v_idxs+1)*np.pi*u)/(2*N) )
        return matrix


    def get_img_matrix(self, dir, YCbCr_key=True):
        """
        get numpy YCbCr or RGB matrix
        :param dir: img dir
        :param YCbCr_key: key to get YCbCr
        :return: numpy YCbCr or RGB
        """
        if YCbCr_key:
            img = cv2.resize(cv2.imread(dir), (512, 256))
            img = rgb2ycbcr(img)
        else:
            img = cv2.resize(cv2.imread(dir), (512, 256))

        return img / 255

    def do_small_DCT_transform(self, img_box, DCT_matrix):
        """
        #https://www.youtube.com/watch?time_continue=1&v=tW3Hc0Wrgl0&feature=emb_logo
        convert spatial domain into freq domain for one bbox
        :param img_box (numpy): NxN part of img
        :param DCT_matrix (numpy): NxN DCT matrix
        :return: NxN img array after DCT
        """
        channels = img_box.shape[2]
        for i in range(channels):
            img_box[:,:, i] = np.dot(np.dot(DCT_matrix, img_box[:,:, i]), np.transpose(DCT_matrix))
        return img_box

    def do_small_rev_DCT_transform(self, img_box, DCT_matrix):
        """
        #https://www.youtube.com/watch?time_continue=1&v=tW3Hc0Wrgl0&feature=emb_logo
        convert freq domain into spatial domain for one bbox
        :param img_box (numpy): NxN part of img after DCT
        :param DCT_matrix (numpy): NxN DCT matrix
        :return: NxN img array
        """
        channels = img_box.shape[2]
        for i in range(channels):
            img_box[:,:, i] = np.dot(np.dot( np.transpose(DCT_matrix), img_box[:,:, i]), DCT_matrix)
        return img_box

    def do_full_DCT_transform(self, img, N):
        """
        convert spatial domain into freq domain for full img
        :param img (numpy): img
        :param N (int): size of jpeg bbox
        :return: img after DCT
        """
        img = copy.deepcopy(img)
        h, w = img.shape[0],img.shape[1]
        for i in range(int(h / N)):
            i = i * N
            for j in range(int(w / N)):
                j = j * N
                # cv2.imshow("before", img[i:i+N,j:j+N, :])
                # cv2.waitKey(0)
                img[i:i+N,j:j+N, :] = self.do_small_DCT_transform(img[i:i+N,j:j+N, :], self.DCT_matrix)
                # cv2.imshow("after", img[i:i + N, j:j + N, :])
                # cv2.waitKey(0)
                a = 0
        return img

    def do_full_rev_DCT_transform(self, img_DCT, N):
        """
        convert freq domain into spatial domain for full img
        :param img_DCT(numpy): img after DCT
        :return: img
        """
        img_DCT = copy.deepcopy(img_DCT)
        h, w = img_DCT.shape[0], img_DCT.shape[1]
        for i in range(int(h / N)):
            i = i * N
            for j in range(int(w / N)):
                j = j * N
                # cv2.imshow("before", img_DCT[i:i+N,j:j+N, :])
                # cv2.waitKey(0)
                img_DCT[i:i+N,j:j+N] = self.do_small_rev_DCT_transform(img_DCT[i:i + N, j:j + N], self.DCT_matrix)
                # cv2.imshow("after", img_DCT[i:i + N, j:j + N, :])
                # cv2.waitKey(0)
        return img_DCT


    def simple_compression(self, img_DCT, n, N):
        """
        https://habr.com/en/post/206264/
        set last n coef to zero
        convert freq domain into spatial domain for full img
        :param img_DCT(numpy): img after DCT
        :param n (int): number of last coef
        :param N (int): size of jpeg bbox
        :return: img
        """

        def set_to_zero(img_DCT_part, n):
            img_DCT_part[N-n:,N-n:,:] = 0
            return img_DCT_part

        img_DCT_after_simple_compress = copy.deepcopy(img_DCT)
        h, w = img_DCT.shape[0], img_DCT.shape[1]
        for i in range(int(h / N)):
            i = i * N
            for j in range(int(w / N)):
                j = j * N

                img_DCT_after_simple_compress[i:i+N,j:j+N] = set_to_zero(img_DCT_after_simple_compress[i:i+N,j:j+N], n)

        return img_DCT_after_simple_compress

    def small_quantization(self, img_DCT_part, quantization_matrix):
        """
        https://habr.com/en/post/206264/
        do quantization for part of img DCT matrix
        :param img_DCT_part(numpy): img part after DCT
        :param quantization_matrix (numpy): jpeg quantization matrix
        :param N (int): size of jpeg bbox
        :return: img part after DCT and quantization
        """
        channels = img_DCT_part.shape[2]
        for i in range(channels):
            img_DCT_part[:, :, i] = np.round(img_DCT_part[:, :, i] * 255 / quantization_matrix) / 255
        return img_DCT_part

    def full_quantization(self, img_DCT, quantization_matrix, N):
        """
        https://habr.com/en/post/206264/
        do quantization for full img DCT matrix
        :param img_DCT (numpy): img after DCT
        :param quantization_matrix (numpy): jpeg quantization matrix
        :return: img part after DCT and quantization
        """
        img_DCT_after_compress = copy.deepcopy(img_DCT)
        h, w = img_DCT.shape[0], img_DCT.shape[1]
        for i in range(int(h / N)):
            i = i * N
            for j in range(int(w / N)):
                j = j * N

                img_DCT_after_compress[i:i+N,j:j+N] = self.small_quantization(img_DCT_after_compress[i:i+N,j:j+N], quantization_matrix)

        return img_DCT_after_compress

    def small_dequantization(self, img_DCT_part, quantization_matrix):
        """
        https://habr.com/en/post/206264/
        do dequantization for part of img DCT matrix
        :param img_DCT_part(numpy): img part after DCT
        :param quantization_matrix (numpy): jpeg quantization matrix
        :param N (int): size of jpeg bbox
        :return: img part after DCT and dequantization
        """
        channels = img_DCT_part.shape[2]
        for i in range(channels):
            img_DCT_part[:, :, i] = np.round(img_DCT_part[:, :, i] * 255 * quantization_matrix) / 255
        return img_DCT_part

    def full_dequantization(self, img_DCT, quantization_matrix, N):
        """
        https://habr.com/en/post/206264/
        do dequantization for full img DCT matrix
        :param img_DCT (numpy): img after DCT
        :param quantization_matrix (numpy): jpeg quantization matrix
        :return: img part after DCT and dequantization
        """
        img_DCT_after_compress = copy.deepcopy(img_DCT)
        h, w = img_DCT.shape[0], img_DCT.shape[1]
        for i in range(int(h / N)):
            i = i * N
            for j in range(int(w / N)):
                j = j * N

                img_DCT_after_compress[i:i+N,j:j+N] = self.small_dequantization(img_DCT_after_compress[i:i+N,j:j+N], quantization_matrix)

        return img_DCT_after_compress

    def check_DCT(self):
        cv2.imshow("input", self.img)
        cv2.imshow("DCT", self.img_DCT)
        cv2.imshow("rev_DCT", self.img_rev_DCT)
        cv2.imshow("simple compress", ycbcr2rgb(self.img_after_simple_compress * 255) / 255)
        cv2.imshow("ok compress after dequantization", ycbcr2rgb(self.img_after_ok_compression * 255) / 255)
        cv2.waitKey(0)
