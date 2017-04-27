#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge

from traffic_light_detect import TrafficLightDetector
import time

class TrafficLightDetectNode:
    def __init__(self):
        self.bridge = CvBridge()

        self.detector = TrafficLightDetector()

        self.pub_image = rospy.Publisher("/traffic_lights", Image, queue_size=1)
        self.sub_image = rospy.Subscriber("/image_raw", Image, self.processImage, queue_size=1)
        self.t0 = time.clock()

    def processImage(self, image_msg):
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        boxes, box_image = self.detector.detect(image)

        image_ros_msg = self.bridge.cv2_to_imgmsg(box_image, "bgr8")
        self.pub_image.publish(image_ros_msg)
        print time.clock() - t0, "seconds wall time"


if __name__=="__main__":
    rospy.init_node('TrafficLightDetect')
    TrafficLightDetectNode()
    rospy.spin()

