#!/usr/bin/env python3
# coding: utf-8
# File: find_contours.py
# Author: lxw
# Date: 2/28/18 10:39 PM

"""
References:
1. OpenCV安装、使用的汇总 [OpenCV 3 Tutorials, Resources, and Guides](https://www.pyimagesearch.com/opencv-tutorials-resources-guides/)
2. 在Ubuntu16.04上安装OpenCV [Ubuntu 16.04: How to install OpenCV](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/#comment-451565)
3. [Checking your OpenCV version using Python](https://www.pyimagesearch.com/2015/08/10/checking-your-opencv-version-using-python/#comment-451559)
4. [Getting Started with Images](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html#additional-resources) 学习手册
5. [OpenCV-Python API documentation](http://opencv-python-tutroals.readthedocs.io/en/latest/index.html)
"""

import cv2
import imutils
from matplotlib import pyplot as plt

# load the Tetris block image, convert it to grayscale, and threshold the image.
print("OpenCV Version: {}".format(cv2.__version__))    # 3.4.1

# image = cv2.imread("../data/input/images/tetris_blocks.png")
image = cv2.imread("../data/input/images/lxw_logo.png")
# Warning: Color image loaded by OpenCV is in BGR mode. But Matplotlib displays in RGB mode.
# So color images will not be displayed correctly in Matplotlib if image is read with OpenCV.
# plt.imshow(image, cmap="gray", interpolation="bicubic")    # NO
plt.imshow(image[:, :, ::-1], cmap="gray", interpolation="bicubic")    # OK
# hide tick values on X and Y axis
plt.xticks([])
plt.yticks([])
plt.show()

gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.threshold(src, thresh, maxval, type, dst=None)
# thresh_img = cv2.threshold(gray_img, 225, 255, cv2.THRESH_BINARY_INV)[1]
thresh_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)[1]
cv2.namedWindow("thresh_img", cv2.WINDOW_NORMAL)
cv2.imshow("thresh_img", thresh_img)

plt.imshow(thresh_img, cmap="gray", interpolation="bicubic")
# hide tick values on X and Y axis
plt.xticks([])
plt.yticks([])
plt.show()

if imutils.is_cv2():
    (cnts, _) = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
elif imutils.is_cv3():
    (_, cnts, _) = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw the contours on the image
cv2.drawContours(image, cnts, -1, (240, 0, 159), 3)    # 使用紫色绘制轮廓
# cv2.drawContours(image, cnts, -1, (0, 0, 0), 3)    # 使用黑色绘制轮廓

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image", image)

k = cv2.waitKey(0)
if k == ord("s"):    # "S": to save and exit
    cv2.imwrite("../data/output/thresh_img.png", thresh_img)
    cv2.imwrite("../data/output/image.png", image)

cv2.destroyAllWindows()
