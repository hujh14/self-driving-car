import cv2
import numpy as np
import matplotlib.pyplot as plt

class StopLineDetector:

    def __init__(self):
    	pass

    def detect(self, image):
        image = cv2.resize(image, tuple(reversed(map(lambda x:x/2,image.shape[:2]))))
    	pass

if __name__=="__main__":
    template = io.imread('images/template_base.png', as_grey=True)
    detector = TrafficLightDetector()

    image = io.imread('images/full_image.png', as_grey=True)
    h,w = image.shape
    # image = resize(image, (h/2, w/2))
    # image = io.imread('images/true_big.png', as_grey=True)
    # image = io.imread('images/false.png', as_grey=True)

    match = detector.detect(image)
    # plt.imshow(hog_image, cmap=plt.cm.gray)
    # plt.show()

    print match

