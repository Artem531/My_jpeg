from utils import jpeg, entropy, rgb2ycbcr
import cv2
from skimage.measure import compare_psnr

dir = 'cats.bmp'
img = cv2.imread(dir)
cv2.imwrite("cats.png", img)
cv2.imwrite("cats.jpg", img)

ref_jpg = rgb2ycbcr(cv2.resize(cv2.imread("cats.jpg"), (512, 256))).astype(int)
ref_png = rgb2ycbcr(cv2.resize(cv2.imread("cats.png"), (512, 256))).astype(int)

ref_entropy_jpg = list(map(lambda i: entropy(ref_jpg[:, :, i].reshape(-1)), list(range(3))))
ref_entropy_png = list(map(lambda i: entropy(ref_png[:, :, i].reshape(-1)), list(range(3))))
print("ref_entropy_jpg: ", ref_entropy_jpg)
print("ref_entropy_png: ", ref_entropy_png)

a = jpeg(dir)
simple_jpg = (a.img_after_simple_compress * 255).astype(int)
simple_jpg[simple_jpg < 0] = 0
ok_jpg = (a.img_after_ok_compression * 255).astype(int)
ok_jpg[ok_jpg < 0] = 0

My_simple_entropy_jpg = list(map(lambda i: entropy(simple_jpg[:, :, i].reshape(-1)), list(range(3))))
My_ok_entropy_jpg = list(map(lambda i: entropy(ok_jpg[:, :, i].reshape(-1)), list(range(3))))
print("My_simple_entropy_jpg: ", My_simple_entropy_jpg)
print("My_ok_entropy_jpg: ", My_ok_entropy_jpg)


print("psnr", compare_psnr(ref_jpg, ok_jpg, data_range=None))
a.check_DCT()
