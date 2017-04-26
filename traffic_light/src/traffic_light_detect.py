import cv2

from detect import Detector

class TrafficLightDetector:

    def __init__(self):
    	self.detector = Detector()

    def detect(self, image):
        boxes = self.detector.detect(image)
        traffic_light_boxes = self.filter(boxes, "traffic light")
        box_image = self.detector.drawBoxes(image, traffic_light_boxes)

        return traffic_light_boxes, box_image

    def filter(self, boxes, category):
        filtered_boxes = []
        for box in boxes:
            left, right, top, bot, mess, max_indx, confidence = box
            if mess == category:
                filtered_boxes.append(box)
        return filtered_boxes
    	

if __name__=="__main__":
    detector = TrafficLightDetector()

