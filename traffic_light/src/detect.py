#! /usr/bin/env python

from darkflow.net.build import TFNet
from tensorflow import flags
import cv2

flags.DEFINE_string("test", "./test/", "path to testing directory")
flags.DEFINE_string("binary", "./bin/", "path to .weights directory")
flags.DEFINE_string("config", "./cfg/", "path to .cfg directory")
flags.DEFINE_string("dataset", "../pascal/VOCdevkit/IMG/", "path to dataset directory")
flags.DEFINE_string("backup", "./ckpt/", "path to backup folder")
flags.DEFINE_string("summary", "./summary/", "path to TensorBoard summaries directory")
flags.DEFINE_string("annotation", "../pascal/VOCdevkit/ANN/", "path to annotation directory")
flags.DEFINE_float("threshold", 0.1, "detection threshold")
flags.DEFINE_string("model", "", "configuration of choice")
flags.DEFINE_string("trainer", "rmsprop", "training algorithm")
flags.DEFINE_float("momentum", 0.0, "applicable for rmsprop and momentum optimizers")
flags.DEFINE_boolean("verbalise", True, "say out loud while building graph")
flags.DEFINE_boolean("train", False, "train the whole net")
flags.DEFINE_string("load", "", "how to initialize the net? Either from .weights or a checkpoint, or even from scratch")
flags.DEFINE_boolean("savepb", False, "save net and weight to a .pb file")
flags.DEFINE_float("gpu", 0.0, "how much gpu (from 0.0 to 1.0)")
flags.DEFINE_float("lr", 1e-5, "learning rate")
flags.DEFINE_integer("keep",20,"Number of most recent training results to save")
flags.DEFINE_integer("batch", 16, "batch size")
flags.DEFINE_integer("epoch", 1000, "number of epoch")
flags.DEFINE_integer("save", 2000, "save checkpoint every ? training examples")
flags.DEFINE_string("demo", '', "demo on webcam")
flags.DEFINE_boolean("profile", False, "profile")
flags.DEFINE_boolean("json", False, "Outputs bounding box information in json format.")
flags.DEFINE_boolean("saveVideo", False, "Records video from input video or camera")
FLAGS = flags.FLAGS

class Detector:
	def __init__(self):
		checkpoint = 8987
		model = "./cfg/tiny-yolo-udacity.cfg"

		FLAGS.load = checkpoint
		FLAGS.model = model

		self.tfnet = TFNet(FLAGS)

	def detect(self, image):
		boxes = self.tfnet.return_predict(image)
		return boxes

	def drawBoxes(self, image, boxes):
		box_image = image.copy()
		h,w,_ = box_image.shape
		for box in boxes:
			left, right, top, bot, mess, max_indx, confidence = box

			BLUE = (255,0,0)
			thick = int((h + w) // 300)
			cv2.rectangle(box_image, (left, top), (right, bot), BLUE, thick)
			cv2.putText(box_image, mess, (left, top - 12), 0, 1e-3 * h, BLUE,thick//3)
		return box_image


if __name__ == "__main__":
	detector = Detector()
	image_path = "./test/street/street3.png"
	image = cv2.imread(image_path)

	boxes = detector.detect(image)
	box_image = detector.drawBoxes(image, boxes)
	print boxes
	cv2.imshow("detections", box_image)
	cv2.waitKey(0)

