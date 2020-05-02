import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt


def psnr(img1, img2):
    """
    Computes psnr.
    :param img1: first img
    :param img2: second img
    :return: psnr
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def get__BC_magnitude(val):
    return np.floor(np.log2(abs(val) + 1)), val


def AC_count_bits(array):
    """
    count full bits consumed for img array (AC coefficients)
    :param array: (run, level) array
    return (int): bits
    """
    unique_rows, counts = np.unique(array, axis=0, return_counts=True)
    finale_bit_for_run = np.sum(counts) * 4
    bits_for_level = [(4 + get__BC_magnitude(val[1])[0]) for val in unique_rows]
    finale_bits_for_level = np.sum([level_bits * count for level_bits, count in zip(bits_for_level, counts)])

    return finale_bit_for_run + finale_bits_for_level


def differential_coding(array):
    """
    differential coding
    return: differential code array
    """
    res = [array[0] - np.mean(array)]
    for i in range(1, len(array)):
        res.append(array[i] - array[i - 1])
    return res


def DC_count_bits(array):
    """
    count full bits consumed for img array (DC coefficients)
    :param array: 1D-array of DC coefficients
    return (int): bits
    """
    differential_code = differential_coding(array)
    unique_vals, counts = np.unique(differential_code, axis=0, return_counts=True)
    bits_for_val = [(4 + get__BC_magnitude(val)[0]) for val in unique_vals]
    finale_bits_for_DC = np.sum([level_bits * count for level_bits, count in zip(bits_for_val, counts)])

    return finale_bits_for_DC


def code_seq(array):
    """
    Series Length Coding
    return: (run, level) array
    """
    prev = array[0]
    count = 1
    res = []
    for i in array[1:]:
        if count > 15 and prev == 0 and i == prev:
            res.append([15, 0])
            count = 1
            continue

        if i == prev and prev == 0:
            count += 1
            continue
        res.append([count, i])
        prev = i
        count = 1
    res.append([0, 0])
    return res


def entropy(array):
    """
    Computes entropy.
    """
    len_arr = array.shape[0]
    # ensure int arr
    array = np.array(array).astype(int)
    array = np.array(code_seq(array))
    unique_rows, counts = np.unique(array, axis=0, return_counts=True)

    probs = counts / np.sum(counts)
    entropy = 0.
    for i in probs:
        entropy -= i * np.log2(i)

    return entropy  # * len(unique_rows)


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:, :, [1, 2]] += 128
    return ycbcr


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:, :, [1, 2]] -= 128
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
        self.DCT_after_full_quantization = self.full_dequantization(
            self.full_quantization(self.img_DCT, self.quantization_matrix, N), self.quantization_matrix, N)
        self.img_after_ok_compression = self.do_full_rev_DCT_transform(self.DCT_after_full_quantization, N)

        self.get_PSNR_percentage_graph(1)
        self.get_bits_percentage_graph(1)
        self.get_PSNR_bits_graph(1)

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
                quantization_matrix_50[i, j] = (i + j) * percentage + 1

        # quantization_matrix_50 = np.array(quantization_matrix_50)

        # print(quantization_matrix_50.shape)
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

        matrix = np.zeros((N, N))
        matrix[0, :] = np.sqrt(1 / N)
        v_idxs = np.array(range(N))
        for u, u_axis in enumerate(matrix):
            if u == 0:
                continue

            matrix[u, :] = np.sqrt(2 / N) * np.cos(((2 * v_idxs + 1) * np.pi * u) / (2 * N))
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
            img_box[:, :, i] = np.dot(np.dot(DCT_matrix, img_box[:, :, i]), np.transpose(DCT_matrix))
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
            img_box[:, :, i] = np.dot(np.dot(np.transpose(DCT_matrix), img_box[:, :, i]), DCT_matrix)
        return img_box

    def do_full_DCT_transform(self, img, N):
        """
        convert spatial domain into freq domain for full img
        :param img (numpy): img
        :param N (int): size of jpeg bbox
        :return: img after DCT
        """
        img = copy.deepcopy(img)
        h, w = img.shape[0], img.shape[1]
        for i in range(int(h / N)):
            i = i * N
            for j in range(int(w / N)):
                j = j * N
                # cv2.imshow("before", img[i:i+N,j:j+N, :])
                # cv2.waitKey(0)
                img[i:i + N, j:j + N, :] = self.do_small_DCT_transform(img[i:i + N, j:j + N, :], self.DCT_matrix)
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
                img_DCT[i:i + N, j:j + N] = self.do_small_rev_DCT_transform(img_DCT[i:i + N, j:j + N], self.DCT_matrix)
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
            img_DCT_part[N - n:, N - n:, :] = 0
            return img_DCT_part

        img_DCT_after_simple_compress = copy.deepcopy(img_DCT)
        h, w = img_DCT.shape[0], img_DCT.shape[1]
        for i in range(int(h / N)):
            i = i * N
            for j in range(int(w / N)):
                j = j * N

                img_DCT_after_simple_compress[i:i + N, j:j + N] = set_to_zero(
                    img_DCT_after_simple_compress[i:i + N, j:j + N], n)

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

                img_DCT_after_compress[i:i + N, j:j + N] = self.small_quantization(
                    img_DCT_after_compress[i:i + N, j:j + N], quantization_matrix)

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

                img_DCT_after_compress[i:i + N, j:j + N] = self.small_dequantization(
                    img_DCT_after_compress[i:i + N, j:j + N], quantization_matrix)

        return img_DCT_after_compress

    def get_PSNR_bits_graph(self, step):
        """
        plot bits/PSNR graph
        :param step (int): plot step
        :return: None
        """

        fig = plt.figure()
        plt.title("bits/PSNR")
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_ylabel('bits')
        ax1.set_xlabel('PSNR')

        plt.plot(self.PSNR, self.bits_arr[:, 0], label="Y bits")
        plt.plot(self.PSNR, self.bits_arr[:, 1], label="Cb bits")
        plt.plot(self.PSNR, self.bits_arr[:, 2], label="Cr bits")

        plt.legend(loc='upper right', borderaxespad=0.)
        plt.savefig("bits-PSNR.jpg")
        plt.show()

    def walk(self, matrix):
        """
        https://qna.habr.com/q/146043
        Move all zeros to the end
        :param DCT (numpy): DCT array
        :return: DCT 1D-array with all zeros in the end
        """
        zigzag = []
        for index in range(1, len(matrix)):
            slice = [i[:index] for i in matrix[:index]]
            diag = [slice[i][len(slice) - i - 1] for i in range(len(slice))]
            if len(diag) % 2:
                diag.reverse()
            zigzag += diag
        return zigzag[1:]

    # def walk(self, matrix):
    #     """
    #     https://qna.habr.com/q/146043
    #     Move all zeros to the end
    #     :param DCT (numpy): DCT array
    #     :return: DCT 1D-array with all zeros in the end
    #     """
    #     return list(matrix.reshape(-1))

    def full_walk(self, img_DCT, N):
        """
        Move all zeros to the end for all img
        :param DCT (numpy): DCT array
        :return: seq (run, level) 1D-array for AC and 1D-array of DC coef
        """
        img_DCT_after_zig_zag = copy.deepcopy(img_DCT)
        res = []
        DC_arr = []
        h, w = img_DCT.shape[0], img_DCT.shape[1]
        for i in range(int(h / N)):
            i = i * N
            for j in range(int(w / N)):
                j = j * N

                res += code_seq(self.walk(img_DCT_after_zig_zag[i:i + N, j:j + N]))
                DC_arr.append(img_DCT_after_zig_zag[i, j])
        res = np.array(res)
        return res, DC_arr

    def get_bits_percentage_graph(self, step):
        """
        plot percentage/biys graph
        :param step (int): plot step
        :return: None
        """
        bits_list = []
        for percentage in range(0, 100, step):
            quantization_matrix = self.get_quantization_matrix(percentage)
            DCT_after_full_quantization = self.full_quantization(self.img_DCT, quantization_matrix, self.N) * 255
            DCT_after_zig_zag = list(
                map(lambda i: self.full_walk(DCT_after_full_quantization[:, :, i], self.N), list(range(3))))
            AC_seq_arr = []
            DC_arr = []
            for i in DCT_after_zig_zag:
                AC_seq_arr.append(i[0])
                DC_arr.append(i[1])

            AC_seq_arr = np.array(AC_seq_arr)
            # img_after_ok_compression = ycbcr2rgb(self.do_full_rev_DCT_transform(DCT_after_full_quantization, self.N) * 255)
            AC_full_bits_arr = np.array(list(map(lambda i: AC_count_bits(AC_seq_arr[i]), list(range(3)))))
            DC_full_bits_arr = np.array(list(map(lambda i: DC_count_bits(DC_arr[i]), list(range(3)))))
            full_bits_arr = AC_full_bits_arr + DC_full_bits_arr
            bits_list.append(full_bits_arr)
        bits_arr = np.array(bits_list)

        fig = plt.figure()
        plt.title("percentage/bits")
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_xlabel('R')
        ax1.set_ylabel('bits')

        plt.plot(range(0, 100, step), bits_arr[:, 0], label="Y bits")
        plt.plot(range(0, 100, step), bits_arr[:, 1], label="Cb bits")
        plt.plot(range(0, 100, step), bits_arr[:, 2], label="Cr bits")

        plt.legend(loc='upper right', borderaxespad=0.)
        plt.savefig("percentage-bits.jpg")
        plt.show()
        self.bits_arr = bits_arr

    def get_PSNR_percentage_graph(self, step):
        """
        plot percentage/PSNR graph
        :param step (int): plot step
        :return: None
        """
        PSNR = []
        img_DCT = copy.deepcopy(self.img_DCT)
        ref = self.img * 255

        for percentage in range(0, 100, step):
            quantization_matrix = self.get_quantization_matrix(percentage)
            DCT_after_full_quantization = self.full_dequantization(
                self.full_quantization(img_DCT, quantization_matrix, self.N), quantization_matrix, self.N)
            img_after_ok_compression = self.do_full_rev_DCT_transform(DCT_after_full_quantization, self.N) * 255
            PSNR_jpg = psnr(img_after_ok_compression, ref)
            PSNR.append(PSNR_jpg)

        fig = plt.figure()
        plt.title("percentage/PSNR")
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_xlabel('R')
        ax1.set_ylabel('PSNR')
        plt.plot(range(0, 100, step), PSNR, label="PSNR")
        plt.legend(loc='upper right', borderaxespad=0.)
        plt.savefig("percentage-PSNR.jpg")
        plt.show()
        self.PSNR = PSNR

    def check_DCT(self):
        cv2.imshow("input", self.img)
        cv2.imshow("DCT", self.img_DCT)
        cv2.imshow("rev_DCT", self.img_rev_DCT)
        cv2.imshow("simple compress", ycbcr2rgb(self.img_after_simple_compress * 255) / 255)
        cv2.imshow("ok compress after dequantization", ycbcr2rgb(self.img_after_ok_compression * 255) / 255)
        cv2.waitKey(0)
