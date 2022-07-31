from __future__ import print_function, division
import cv2
import numpy as np

def crop_to_square(image):
    # 1280x720
    #print(image.size) #ex:(w:500,h:281)size no gazou
    height, width = image.shape[:2]
    # 720, 1280
    size = min(height, width) #ex:h:281 no houga tiisai
    # 720
    #left, upper = image.width, image.height
    #right, bottom = image.width,image.height
    left, top = (width - size) // 2, (height - size) // 2
    # 280, 0
    right, bottom = (width + size) // 2, (height + size) // 2
    # 1000, 720
    top, bottom, left, right = top + 208, bottom, left + 104, right - 104 #512*512 image(realsense nomi)
    # 208, 720, 384, 896
    image = image[top : bottom, left : right]
    return image

def load_image(path):
    image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    image = crop_to_square(image)
    return image

#resize(original)
def resize_image(image, label, resize_height, resize_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')

    if (resize_width <= image.shape[1]) and (resize_height <= image.shape[0]):
        image = cv2.resize(image, dsize=(resize_width, resize_height))
        label = cv2.resize(label, dsize=(resize_width, resize_height))
        return image, label
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (resize_height, resize_width, image.shape[0], image.shape[1]))

def reverse_color(img):
    return cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR)
