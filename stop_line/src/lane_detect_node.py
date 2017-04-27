#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from lane_detect import LaneDetector

class LaneDetectNode:
    def __init__(self):
        self.bridge = CvBridge()

        self.detector = LaneDetector()

        # self.pub_stop_line = rospy.Publisher("/lane", Image, queue_size=1)
        self.pub_debug = rospy.Publisher("/lane/debug", Image, queue_size=1)
        self.sub_image = rospy.Subscriber("/image_raw", Image, self.processImage, queue_size=1)

    def processImage(self, image_msg):
        image = self.bridge.imgmsg_to_cv2(image_msg)

        stop_line, debug = self.detector.detect(image)
        print stop_line

        image_ros_msg = self.bridge.cv2_to_imgmsg(debug, "bgr8")
        self.pub_debug.publish(image_ros_msg)

if __name__=="__main__":
    rospy.init_node('LaneDetect')
    LaneDetectNode()
    rospy.spin()

