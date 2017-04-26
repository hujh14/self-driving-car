#!/usr/bin/env python
import rospy # standard ros with python package
from sensor_msgs.msg import Image  # the rostopic message we subscribe/publish
from cv_bridge import CvBridge # package to convert rosmsg<->cv2

from stop_line_detect import StopLineDetector

class StopLineDetectNode:
    def __init__(self):
        # Bridge to convert to and from cv2 and rosmsg
        self.bridge = CvBridge()

        self.detector = StopLineDetector()

        self.pub_image = rospy.Publisher("/traffic_lights", Image, queue_size=1)
        self.sub_image = rospy.Subscriber("/camera_raw", Image, self.processImage, queue_size=1)
        self.sub_camera_info = rospy.Subscriber("/camera_info", Image, self.saveCameraInfo, queue_size=1)

    def processImage(self, image_msg):
        image = self.bridge.imgmsg_to_cv2(image_msg)

        detected_light = self.detector.detect(image)

        image_ros_msg = self.bridge.cv2_to_imgmsg(covered_image, "bgr8")
        self.pub_image.publish(image_ros_msg)

    def saveCameraInfo(self, msg):
        self.camera_height = 0

if __name__=="__main__":
    rospy.init_node('StopLineDetect')
    StopLineDetectNode()
    rospy.spin()

