#!/usr/bin/env python

import sys, os, time,csv, cv2, argparse
sys.path.append(os.path.join(os.getcwd(), '/home/ubuntu/chg_workspace/git/darknet/python/'))

import numpy as np
from datetime import datetime

import darknet as dn
import pdb

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
#from object_msgs.msg import *

bridge = CvBridge()

updated = 0
cv2_img = []

def image_callback(msg):
    global updated
    global cv2_img

    if updated == 0:
        try:
            # Convert your ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            updated = 1
        except CvBridgeError, e:
            print e

if __name__ == '__main__':
    rospy.init_node('detector', anonymous=True)
    #boxes_pub = rospy.Publisher('/objects', ObjectsInBoxes, queue_size=1)

    # Parameters and weights load
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', dest='cam', type=int,
                        default='0', help='camera id')
    parser.add_argument('--topic', dest='topic', type=str,
                        default='/usb_cam/image_raw', help='image topic name')
    args = parser.parse_args()

    cam_id = args.cam
    topic = args.topic

    darknet_path = "/home/ubuntu/chg_workspace/git/darknet"

    dn.set_gpu(0)

    net = dn.load_net(darknet_path + "/cfg/yolov3.cfg", darknet_path + "/yolov3.weights", 0)
    meta = dn.load_meta(darknet_path + "/cfg/coco.data")

    # Use opencv to open a camera
    if sys.argv[1] == '--cam':
        # video preprocess
        cap = cv2.VideoCapture(cam_id)
        fps = 0.0
        while cap.isOpened():
            start = time.time()
            imArr = {}
            results = {}

            ret, img = cap.read()

            if ret == True:
                cv2.imwrite(darknet_path + 'cam.png', img)
                r = dn.detect(net, meta, darknet_path + 'cam.png')
                print r

    #r = dn.detect(net, meta, darknet_path + "/data/dog.jpg")


