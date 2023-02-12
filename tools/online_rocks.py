"""
Katwijk dataset processing.

Reads in a particular traverse/part from Katwijk dataset and records the data
to a rosbag. In addition, the stereo stream (as dictated by the user) is
processed into a dense point cloud and the rocks are detected and classified.

Parker Lusk
31 Jan 2023
"""

import argparse
import pathlib
import sys

import cv2 as cv
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import scipy

from tqdm import tqdm

import rospy
import rosbag
import cv_bridge

import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import visualization_msgs.msg as visualization_msgs


import sys; sys.path.append('..')
import katwijk

def plot_ground_truth(dataset):
    xyz = np.array(list(dataset.gpsutm))[:,2:5]

    fig, ax = plt.subplots()
    ax.scatter(xyz[:,0], xyz[:,1])
    for s in katwijk.rocks.SIZE_NUM:
        ax.scatter(dataset.get_rocks(s)[:,0], dataset.get_rocks(s)[:,1], 40,
                    color=katwijk.rocks.SIZE_CLR[s], marker='x')

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Process Katwijk dataset into rosbag with detections')
    parser.add_argument('-f', '--dataset-root',
            type=pathlib.Path,
            default='/media/plusk01/721fc902-b795-4e13-980a-ef6014eb03f0/datasets/katwijk',
            help='Katwijk dataset root')
    parser.add_argument('-o', '--out-bag',
            type=pathlib.Path,
            default=pathlib.Path(__file__).absolute().parent / 'out.bag',
            help='Output rosbag')
    parser.add_argument('-t', '--traverse', type=int, default=1, help='Traverse number')
    parser.add_argument('-p', '--part', type=int, default=1, help='Part number')
    parser.add_argument('--pancam', action='store_true', help='Use PanCam')
    parser.add_argument('--loccam', action='store_true', help='Use LocCam')
    args = parser.parse_args()

    if not args.pancam and not args.loccam:
        args.pancam = True

    if args.pancam and args.loccam:
        parser.error("Only one of --pancam and --loccam can be active at the same time")
        sys.exit(2)

    args.cam = 'PanCam' if args.pancam else 'LocCam'

    return args


if __name__ == '__main__':
    args = parse_args()

    rospy.init_node('katwijk_rocks', anonymous=False)

    pub_img0 = rospy.Publisher(f"{args.cam}/left/image_raw", sensor_msgs.Image, queue_size=1)
    pub_dets = rospy.Publisher(f"{args.cam}/left/detections", sensor_msgs.Image, queue_size=1)
    pub_odom = rospy.Publisher(f"{args.cam}/left/odom", nav_msgs.Odometry, queue_size=1)
    pub_refmap = rospy.Publisher('viz_rocks', visualization_msgs.MarkerArray, queue_size=1)
    pub_rocks = rospy.Publisher(f"rocks", sensor_msgs.PointCloud2, queue_size=1)

    dataset = katwijk.dataset.load(args.dataset_root, args.traverse, args.part)

    katwijk.ros.publish_static_tf(dataset)

    # plot_ground_truth(dataset)

    processor = katwijk.rocks.StereoProcessor(dataset, args.cam, maxdepth=20)

    bridge = cv_bridge.CvBridge()

    idataset = katwijk.dataset.Iterable(dataset)
    for t, key, data in idataset:
        print(t, key)

        if key == args.cam:
            _, img0, img1 = next(data)
            rocks, debugimg = processor.process(t, img0, img1, generate_debug_img=True, visualize_img=False, visualize_pcd=False)
            dbgimg = bridge.cv2_to_imgmsg(debugimg, "bgr8")
            dbgimg.header.frame_id = f"{args.cam}0"
            dbgimg.header.stamp = rospy.Time.from_sec(t)
            pub_dets.publish(dbgimg)

            msg0 = bridge.cv2_to_imgmsg(img0, "bgr8")
            msg0.header.frame_id = f"{args.cam}0"
            msg0.header.stamp = rospy.Time.from_sec(t)
            pub_img0.publish(msg0)

            pcd = o3d.geometry.PointCloud()
            for rock in rocks:
                pcd += rock[1]
            msg = katwijk.ros.o3d_to_ros(pcd)
            msg.header.stamp = rospy.Time.from_sec(t)
            msg.header.frame_id = f"{args.cam}"
            pub_rocks.publish(msg)


        if key == f"{args.cam}OdomGT":
            katwijk.ros.publish_tf(dataset, t, data, parent='odom', child=args.cam)
            msg = katwijk.ros.build_odom_msg(t, data, parent='odom', child=args.cam)
            pub_odom.publish(msg)

        if key == 'GPS-UTM':
            markers_msg = katwijk.ros.build_rock_marker_array(dataset)
            pub_refmap.publish(markers_msg)
