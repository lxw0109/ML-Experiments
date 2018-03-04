import cv2
import imutils

from matplotlib import pyplot as plt


def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """
    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # NOTE: 把图片按照w,h中较大的(并按照原比例)进行resize(但是有什么区别？按照短的感觉也对啊？)
    # 按照w,h中较大的进行resize，是保证w, h在resize后都不要太大，从而保证能够处理到图片中尽可能多的像素点，减少数据损失
    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))    # NOTE: cv2.resize()第二个参数要求是先width，然后再height

    # return the pre-processed image
    return image


def display_image(image):
    # plt.imshow(image[:, :, ::-1], cmap="gray", interpolation="bicubic")    # NO
    plt.imshow(image, cmap="gray", interpolation="bicubic")    # OK
    # hide tick values on X and Y axis
    plt.xticks([])
    plt.yticks([])
    plt.show()