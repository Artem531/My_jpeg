from utils import jpeg, entropy, rgb2ycbcr, psnr
import cv2
from skimage.measure import compare_psnr

dir = 'cats.bmp'

a = jpeg(dir)
a.check_DCT()
