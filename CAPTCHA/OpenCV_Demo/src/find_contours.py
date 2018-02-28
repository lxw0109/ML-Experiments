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
4. [OpenCV API](https://docs.opencv.org/3.4.1/)
"""

import cv2
import imutils

# load the Tetris block image, convert it to grayscale, and threshold the image.
print("OpenCV Version: {}".format(cv2.__version__))    # 3.4.1
image = cv2.imread("../data/tetris_blocks.png")
# image = cv2.imread("../data/lxw.jpg")
# image = cv2.imread("../data/lxw_logo.png")
# image = cv2.imread("../data/two_dogs.jpg")
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh_img = cv2.threshold(gray_img, 225, 255, cv2.THRESH_BINARY_INV)[1]    # cv2.threshold(src, thresh, maxval, type, dst=None)
# thresh_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)[1]    # cv2.threshold(src, thresh, maxval, type, dst=None)
cv2.imshow("Image", thresh_img)

if imutils.is_cv2():
    (cnts, _) = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
elif imutils.is_cv3():
    (_, cnts, _) = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# draw the contours on the image
cv2.drawContours(image, cnts, -1, (240, 0, 159), 3)
cv2.imshow("Image", image)
cv2.waitKey(0)
