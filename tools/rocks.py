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

class KatwijkRockProcessor:
    def __init__(self, dataset, recorder, cam, mindepth=1, maxdepth=30):
        self.dataset = dataset
        self.recorder = recorder
        self.cam = cam
        self.camdata = dataset.get_stereo_camdata(cam)

        self.mindepth = mindepth
        self.maxdepth = maxdepth

        self._init_stereo_rectify_map()
        self.stereo = self._init_stereo_block_matching()

    def _init_stereo_rectify_map(self, visualize=False):
        _, img0, img1 = next(self.camdata)

        h, w = img0.shape[:2]

        K0, D0, R0, P0 = self.dataset.get_stereo_calib(self.cam, 0)
        K1, D1, R1, P1 = self.dataset.get_stereo_calib(self.cam, 1)

        self._xmap0, self._ymap0 = cv.initUndistortRectifyMap(K0, D0, R0, P0, (w, h), cv.CV_32FC1)
        self._xmap1, self._ymap1 = cv.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv.CV_32FC1)
        self._Q = self.dataset.get_stereo_Q(self.cam)

        if visualize:
            # convert to grayscale
            grey0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
            grey1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

            # rectify images from stereo pair
            rect0 = cv.remap(grey0, self._xmap0, self._ymap0, cv.INTER_LANCZOS4)
            rect1 = cv.remap(grey1, self._xmap1, self._ymap1, cv.INTER_LANCZOS4)

            # stitch together, horizontally
            rect = cv.hconcat([rect0, rect1])
            rect = cv.cvtColor(rect, cv.COLOR_GRAY2BGR)

            # draw green lines (they should hit the same objects in both frames)
            for i in range(0, h, 32):
                cv.line(rect, (0, i), (rect.shape[1], i), (0,255,0), 1)

            cv.imshow('Rectified (lines should hit same objects)', rect)
            cv.waitKey(0)

    def _init_stereo_block_matching(self, basic=False):
        if basic:
            # stereo block matching
            stereo = cv.StereoBM_create()
            stereo.setMinDisparity(4)
            stereo.setNumDisparities(128)
            stereo.setBlockSize(21)
            stereo.setSpeckleRange(16)
            stereo.setSpeckleWindowSize(45)
        else:
            # Set disparity parameters
            # Note: disparity range is tuned according to specific parameters obtained through trial and error. 
            win_size = 8
            min_disp = 4
            max_disp = 127 #min_disp * 9
            num_disp = max_disp - min_disp # Needs to be divisible by 16
            # Create Block matching object. 
            stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                numDisparities=num_disp, blockSize=15,
                                uniquenessRatio=5, speckleWindowSize=45,
                                speckleRange=16, disp12MaxDiff=12,
                                P1=8*3*win_size**2, #8*3*win_size**2,
                                P2=32*3*win_size**2) #32*3*win_size**2)
        return stereo

    def _rotate_and_separate(self, pcd, t_s, ransac_distthr=0.1, voxel_size=0.05, start_above_ground=0.10, visualize=False):
        """
        Rotates a point cloud to be ground aligned and
        separate the ground from the rest of the scene.
        
        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
        t_s : float -- timestamp corresponding to the stereo pair that generated the point cloud
        start_above_ground : float -- meters above the ground to crop out

        Returns
        -------
        pcd : o3d.geometry.PointCloud -- ground pcd
        pcd : o3d.geometry.PointCloud -- non-ground pcd
        """
        T_01 = self.dataset.get_stereo_T_01(self.cam)
        T_gc0 = self.dataset.get_stereo_T_gc0(self.cam, t_s)

        pcd = pcd.voxel_down_sample(voxel_size)

        if visualize:
            Fc0 = o3d.geometry.TriangleMesh.create_coordinate_frame()
            Fc1 = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T_01)
            Fg = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(np.linalg.inv(T_gc0))
            o3d.visualization.draw_geometries([Fc0, Fc1, Fg, pcd])

        plane_model, plane_pts_idx = pcd.segment_plane(distance_threshold=ransac_distthr, ransac_n=3, num_iterations=1000)
        n = plane_model[:3].reshape((3,1))

        pcd.transform(T_gc0)

        y = np.asarray(pcd.select_by_index(plane_pts_idx).points)[:,2].mean()

        bbox = pcd.get_axis_aligned_bounding_box()
        minb = bbox.min_bound.copy()
        # minb[0] = -20
        minb[2] = y + start_above_ground
        bbox.min_bound = minb
        maxb = bbox.max_bound.copy()
        maxb[2] = 0
        bbox.max_bound = maxb

        pcd_ground = pcd.select_by_index(plane_pts_idx)
        pcd_nonground = pcd.select_by_index(plane_pts_idx, invert=True).crop(bbox)

        return pcd_ground, pcd_nonground, T_gc0

    def _classify_rock_pcd(self, pcd):
        """
        Determine if a pcd is of a rock; if so, classify its size
        
        Parameters
        ----------
        pcd : o3d.geometry.PointCloud -- rock subpcd
        """
        HEIGHT_LARGE = 0.8
        HEIGHT_MEDIUM = 0.55
        HEIGHT_SMALL = 0.3

        HEIGHT_LARGE = 0.65
        HEIGHT_MEDIUM = 0.375
        HEIGHT_SMALL = 0.125

        hs = np.array([HEIGHT_SMALL, HEIGHT_MEDIUM, HEIGHT_LARGE])
        h = np.asarray(pcd.points)[:,2].max() - np.asarray(pcd.points)[:,2].min()

        if h > HEIGHT_SMALL / 2.:
            c = np.argmin(np.abs(hs - h))
            return c, h
        else:
            return None, h

    @property
    def point_cloud_from_stereo(self):
        # TODO: only need disparity within an ROI (i.e., we can block out the sky)
        # TODO: consider first downsampling the images (and the stereo rectification matrices)

        for t_s, img0, img1 in self.camdata:
            # convert to grayscale
            grey0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
            grey1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

            # rectify images from stereo pair
            rect0 = cv.remap(grey0, self._xmap0, self._ymap0, cv.INTER_LANCZOS4)
            rect1 = cv.remap(grey1, self._xmap1, self._ymap1, cv.INTER_LANCZOS4)

            disparity = self.stereo.compute(rect0, rect1).astype(np.float32) / 16
            xyz = cv.reprojectImageTo3D(disparity, self._Q, handleMissingValues=False)

            # not entirely sure why it was put behind the camera, but flip it
            pts = -xyz.reshape((-1,3))

            # determine which points are in bounds
            idx, = np.where((pts[:,2] > self.mindepth) & (pts[:,2] < self.maxdepth))

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts[idx])
            # TODO: is this right? should we be using img0 or img1 or something else?
            pcd.colors = o3d.utility.Vector3dVector(img0.reshape((-1,3))[idx] / 255)

            yield t_s, pcd, img0, img1

    def classify_rocks(self, record_debug=False, visualize_img=False, visualize_pcd=False):
        for t_s, pcd, img0, img1 in tqdm(self.point_cloud_from_stereo, desc="Classifying Rocks"):

            img0orig = img0.copy()

            pcd_ground, pcd_nonground, T_gc0 = self._rotate_and_separate(pcd, t_s, voxel_size=0.01)

            labels = np.array(pcd_nonground.cluster_dbscan(eps=0.3, min_points=100, print_progress=False))
            print(f"Found {labels.max()+1} clusters")

            if record_debug or visualize_img:
                COLORS = plt.get_cmap('tab10').colors

            K0, D0, R0, P0 = self.dataset.get_stereo_calib(self.cam, 0)

            for l in range(labels.max()+1):
                subpcd = pcd_nonground.select_by_index(np.argwhere(labels == l))

                size, h = self._classify_rock_pcd(subpcd)
                if size is None:
                    continue

                length = np.asarray(subpcd.points)[:,0].max() - np.asarray(subpcd.points)[:,0].min()
                width = np.asarray(subpcd.points)[:,1].max() - np.asarray(subpcd.points)[:,1].min()
                height = np.asarray(subpcd.points)[:,2].max() - np.asarray(subpcd.points)[:,2].min()

                print(f"Rock size {size}: lxwxh={length:.2f} x {width:.2f} x {height:.2f} m, with {len(subpcd.points)} points")


                if record_debug or visualize_img or visualize_pcd:
                    clr = COLORS[size]
                    # clr = COLORS[l%len(COLORS)]
                    N = np.count_nonzero(labels == l)
                    colors = np.tile(clr, (N,1))

                    # Get points, transformed back into cam0
                    subpcd.transform(np.linalg.inv(T_gc0))
                    xyz = np.asarray(subpcd.points)

                    # Project points into image
                    pix, _ = cv.projectPoints(xyz, (0,0,0), (0,0,0), K0, D0)
                    pix = pix.squeeze()

                    C = tuple(map(lambda x: int(x*255), clr))[::-1]
                    for p in pix:
                        cv.circle(img0, (int(p[0]), int(p[1])), 5, C, -1)

                    np.asarray(pcd_nonground.colors)[np.argwhere(labels == l).flatten()] = colors

            if visualize_img:

                cv.imshow(self.cam, img0)
                cv.imshow(f"{self.cam} orig", img0orig)
                cv.waitKey(1)

            if visualize_pcd:

                pcd_ground.paint_uniform_color([1,0,0])
                Fg = o3d.geometry.TriangleMesh.create_coordinate_frame()
                Fc0 = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T_gc0)
                o3d.visualization.draw_geometries([Fg, Fc0, pcd_nonground])

            if record_debug:
                rostime = rospy.Time.from_sec(t_s)

                msg0 = self.recorder.bridge.cv2_to_imgmsg(img0, "bgr8")
                msg0.header.frame_id = f"{self.cam}0"
                msg0.header.stamp = rostime

                self.recorder.bag.write(self.recorder._ensure_valid(f"{self.cam}/left/detections"), msg0, rostime)


    def record_to_bag(self):
        pass


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


    recorder = katwijk.ros.BagRecorder(dataset, args.out_bag)

    processor = KatwijkRockProcessor(dataset, recorder, args.cam, maxdepth=20)
    processor.classify_rocks(record_debug=True, visualize_img=True, visualize_pcd=False)
    processor.record_to_bag()

    recorder.record_imu()
    recorder.record_gpsutm()
    if args.cam == 'PanCam':
        recorder.record_pancam()
    elif args.cam == 'LocCam':
        recorder.record_loccam()

    recorder.close()