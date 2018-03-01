#!/usr/bin/env python3
# coding: utf-8
# File: cats_detector.py
# Author: lxw
# Date: 2/28/18 10:44 PM

"""
References:
1. [Detecting cats in images with OpenCV](https://www.pyimagesearch.com/2016/06/20/detecting-cats-in-images-with-opencv/)
2. [Good posts on OpenCV](https://www.pyimagesearch.com/done/)
"""

def main():
    import argparse
    import cv2

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    ap.add_argument("-c", "--cascade", default="../data/input/haarcascade_frontalcatface.xml", help="path to cat detector haar cascade")
    args = vars(ap.parse_args())    # <class 'dict'>

    # load the input image and convert it to grayscale
    image = cv2.imread(args["image"])
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray Cat Faces", gray_img)

    # load the cat detector Haar cascade, then detect cat faces in the input image
    detector = cv2.CascadeClassifier(args["cascade"])
    rects = detector.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=10, minSize=(75, 75))    # a list of 4-tuples

    # loop over the cat faces and draw a rectangle surrounding each
    # for (i, (x, y, w, h)) in enumerate(rects):
    for i, (x, y, w, h) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # show the detected cat faces
    cv2.imshow("Cat Faces", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()