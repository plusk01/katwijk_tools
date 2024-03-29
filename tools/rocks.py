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


import sys; sys.path.append('..')
import katwijk

def plot_ground_truth(dataset):
    neh = np.array(list(dataset.gpsutm))[:,2:5]
    srocks = dataset.rocks[np.where(dataset.rocks[:,0]==0)]
    mrocks = dataset.rocks[np.where(dataset.rocks[:,0]==1)]
    lrocks = dataset.rocks[np.where(dataset.rocks[:,0]==2)]

    # import ipdb; ipdb.set_trace()

    COLORS = plt.get_cmap('tab10').colors

    fig, ax = plt.subplots()

    ax.scatter(neh[:,0], neh[:,1])
    ax.scatter(srocks[:,1], srocks[:,2], 40, color=COLORS[0], marker='x')
    ax.scatter(mrocks[:,1], mrocks[:,2], 40, color=COLORS[1], marker='x')
    ax.scatter(lrocks[:,1], lrocks[:,2], 40, color=COLORS[2], marker='x')

    plt.show()

    # import ipdb; ipdb.set_trace()


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

    dataset = katwijk.dataset.load(args.dataset_root, args.traverse, args.part)

    # plot_ground_truth(dataset)

    processor = katwijk.rocks.StereoProcessor(dataset, args.cam, maxdepth=20)


    idataset = katwijk.dataset.Iterable(dataset)
    for t, key, data in idataset:
        print(t, key)

        if key == 'LocCam':
            _, img0, img1 = next(data)

            cv.imshow('img', img0)
            cv.waitKey(1)
            # import ipdb; ipdb.set_trace()

        if key == 'LocCamOdomGT':
            print(data)

    # processor.classify_rocks(record_debug=True, visualize_img=True, visualize_pcd=False)

    # rocks, debugimg = processor.process(t, img0, img1, generate_debug_img=True, visualize_img=True, visualize_pcd=False)


    # recorder = katwijk.ros.BagRecorder(dataset, args.out_bag)

    # processor.record_to_bag()

    # recorder.record_imu()
    # recorder.record_gpsutm()
    # if args.cam == 'PanCam':
    #     recorder.record_pancam()
    #     ts = dataset.pancam_timestamps
    # elif args.cam == 'LocCam':
    #     recorder.record_loccam()
    #     ts = dataset.loccam_timestamps
    # recorder.record_groundtruth_odom(ts)

    # recorder.close()