#!/usr/bin/env python

import rospy
import argparse
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import numpy as np

mask_dist = 6.4

def callback(image_depth, image_rgb):
	# Convert your ROS Image message to OpenCV2
	bridge = CvBridge()
	try: 
	    depth_img = bridge.imgmsg_to_cv2(image_depth, "passthrough")
	except CvBridgeError, e1:
	    print(e1)
	    return 

	try: 
	    rgb_img = bridge.imgmsg_to_cv2(image_rgb, "bgr8")
	except CvBridgeError, e2:
	    print(e2)
	    return

	# print depth_img.shape
	# print depth_img.dtype

	# for i in range(depth_img.shape[0]):
	# 	for j in range(depth_img.shape[1]):
	# 		if depth_img[i, j] == depth_img[i, j] and depth_img[i, j] <= 6.4:
	# 			imgZero[i, j] = 255

	imgZero = depth_img < 6.4

	b , g , r = cv2.split(rgb_img)

	b = b & imgZero
	# g = g & imgZero
	# r = r & imgZero

	merged = cv2.merge([b,g,r])

	print merged


	cv2.imshow('merged', merged)
	cv2.waitKey(5)


if __name__ == "__main__":
	
	''' import parameters '''
	parser = argparse.ArgumentParser()
	parser.add_argument('--depth', dest='depth', type=str,
                        default='/camera/depth/image_raw', help='depth image topic name')
	parser.add_argument('--rgb', dest='rgb', type=str,
                        default='/camera/rgb/image_raw', help='rgb image topic name')

	args = parser.parse_args()

	depth_topic = args.depth
	rgb_topic = args.rgb

	''' ros init '''
	rospy.init_node('lantern', anonymous=True)

	depth_sub = message_filters.Subscriber(depth_topic, Image)
	rgb_sub = message_filters.Subscriber(rgb_topic, Image)

	ts = message_filters.TimeSynchronizer([depth_sub, rgb_sub], 10)
	ts.registerCallback(callback)
 	rospy.spin()
