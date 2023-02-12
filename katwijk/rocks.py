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

SIZE_NUM = (0, 1, 2)
SIZE_STR = ('Small', 'Medium', 'Large')
SIZE_CLR = tuple(plt.get_cmap('tab10').colors[i] for i in SIZE_NUM)

class StereoProcessor:
    def __init__(self, dataset, cam, mindepth=1, maxdepth=30):
        self.dataset = dataset
        self.cam = cam

        self.mindepth = mindepth
        self.maxdepth = maxdepth

        self.rectification_initialized = False

        self.stereo = self._init_stereo_block_matching()

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

    def _init_stereo_rectify_map(self, img0, img1, visualize=False):
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
            return c
        else:
            return None

    def _point_cloud_from_stereo(self, img0, img1):
        """
        Create a point cloud from a rectified stereo pair.
        
        Parameters
        ----------
        img0 : (h,w) np.array -- raw image left
        img1 : (h,w) np.array -- raw image right

        Returns
        -------
        pcd : o3d.geometry.PointCloud
        """
        # TODO: only need disparity within an ROI (i.e., we can block out the sky)
        # TODO: consider first downsampling the images (and the stereo rectification matrices)

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

        return pcd

    # def classify_rocks(self, record_debug=False, visualize_img=False, visualize_pcd=False):
    #     for t_s, pcd, img0, img1 in tqdm(self.point_cloud_from_stereo, desc="Classifying Rocks"):

    #         img0orig = img0.copy()

    #         pcd_ground, pcd_nonground, T_gc0 = self._rotate_and_separate(pcd, t_s, voxel_size=0.01)

    #         labels = np.array(pcd_nonground.cluster_dbscan(eps=0.3, min_points=100, print_progress=False))
    #         print(f"Found {labels.max()+1} clusters")

    #         if record_debug or visualize_img:
    #             COLORS = plt.get_cmap('tab10').colors

    #         K0, D0, R0, P0 = self.dataset.get_stereo_calib(self.cam, 0)

    #         for l in range(labels.max()+1):
    #             subpcd = pcd_nonground.select_by_index(np.argwhere(labels == l))

    #             size, h = self._classify_rock_pcd(subpcd)
    #             if size is None:
    #                 continue

    #             length = np.asarray(subpcd.points)[:,0].max() - np.asarray(subpcd.points)[:,0].min()
    #             width = np.asarray(subpcd.points)[:,1].max() - np.asarray(subpcd.points)[:,1].min()
    #             height = np.asarray(subpcd.points)[:,2].max() - np.asarray(subpcd.points)[:,2].min()

    #             print(f"Rock size {size}: lxwxh={length:.2f} x {width:.2f} x {height:.2f} m, with {len(subpcd.points)} points")


    #             if record_debug or visualize_img or visualize_pcd:
    #                 clr = COLORS[size]
    #                 # clr = COLORS[l%len(COLORS)]
    #                 N = np.count_nonzero(labels == l)
    #                 colors = np.tile(clr, (N,1))

    #                 # Get points, transformed back into cam0
    #                 subpcd.transform(np.linalg.inv(T_gc0))
    #                 xyz = np.asarray(subpcd.points)

    #                 # Project points into image
    #                 pix, _ = cv.projectPoints(xyz, (0,0,0), (0,0,0), K0, D0)
    #                 pix = pix.squeeze()

    #                 C = tuple(map(lambda x: int(x*255), clr))[::-1]
    #                 for p in pix:
    #                     cv.circle(img0, (int(p[0]), int(p[1])), 5, C, -1)

    #                 np.asarray(pcd_nonground.colors)[np.argwhere(labels == l).flatten()] = colors

    #         if visualize_img:

    #             cv.imshow(self.cam, img0)
    #             cv.imshow(f"{self.cam} orig", img0orig)
    #             cv.waitKey(1)

    #         if visualize_pcd:

    #             pcd_ground.paint_uniform_color([1,0,0])
    #             Fg = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #             Fc0 = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T_gc0)
    #             o3d.visualization.draw_geometries([Fg, Fc0, pcd_nonground])

    #         if record_debug:
    #             rostime = rospy.Time.from_sec(t_s)

    #             msg0 = self.recorder.bridge.cv2_to_imgmsg(img0, "bgr8")
    #             msg0.header.frame_id = f"{self.cam}0"
    #             msg0.header.stamp = rostime

    #             self.recorder.bag.write(self.recorder._ensure_valid(f"{self.cam}/left/detections"), msg0, rostime)

    def _segment_and_classify(self, pcd, T_gc0):
        """
        Segment and classify rocks from a point cloud.
        
        Parameters
        ----------
        pcd : o3d.geometry.PointCloud -- input point cloud, points w.r.t to {g}
        T_gc0 : (4,4) np.array -- left camera (0) w.r.t {g} (ground)

        Returns
        -------
        rocks : List[(sz, o3d.geometry.PointCloud, int)], sz: 0/S, 1/M, 2/L
        """
        T_c0g = np.linalg.inv(T_gc0)

        labels = np.array(pcd.cluster_dbscan(eps=0.3, min_points=100, print_progress=False))

        rocks = []
        for l in range(labels.max()+1):
            subidx = np.argwhere(labels == l)
            subpcd = pcd.select_by_index(subidx)

            size = self._classify_rock_pcd(subpcd)

            if size is not None:
                # length = np.asarray(subpcd.points)[:,0].max() - np.asarray(subpcd.points)[:,0].min()
                # width = np.asarray(subpcd.points)[:,1].max() - np.asarray(subpcd.points)[:,1].min()
                # height = np.asarray(subpcd.points)[:,2].max() - np.asarray(subpcd.points)[:,2].min()

                # print(f"Rock size {size}: lxwxh={length:.2f} x {width:.2f} x {height:.2f} m, with {len(subpcd.points)} points")
                
                # Transform rock pcd back into cam0
                subpcd.transform(T_c0g)
                subpcd.paint_uniform_color(SIZE_CLR[size])

                rocks.append((size, subpcd, subidx))

        return rocks

    def process(self, t, img0, img1, generate_debug_img=False, visualize_img=False, visualize_pcd=False):
        """
        Classify rocks by processing a stereo pair.
        
        Parameters
        ----------
        t : float -- timestamp of image
        img0 : (h,w) np.array -- left image of stereo pair
        img1 : (h,w) np.array -- right image of stereo pair

        Returns
        -------
        rocks : List[(sz, o3d.geometry.PointCloud, int)], sz: 0/S, 1/M, 2/L
        img : (h,w) np.array -- debug image or original image
        """        
        if not self.rectification_initialized:
            self._init_stereo_rectify_map(img0, img1)
            self.rectification_initialized = True


        # 1. Generate a point cloud from a rectified stereo camera
        pcd = self._point_cloud_from_stereo(img0, img1)

        # 2. Separate ground and non-ground points; transform point cloud into 'level ground' frame
        pcd_ground, pcd_nonground, T_gc0 = self._rotate_and_separate(pcd, t, voxel_size=0.01)

        # 3. Detect and classify rocks; transform back into (left) camera frame
        rocks = self._segment_and_classify(pcd_nonground, T_gc0)


        # visualize, if requested
        if generate_debug_img or visualize_img or visualize_pcd:
            dbg0 = img0.copy()
            K0, D0, _, _ = self.dataset.get_stereo_calib(self.cam, 0)

            for size, rockpcd, rock_pt_idx in rocks:
                clr = SIZE_CLR[size]
                # colors = np.tile(clr, (len(rock_pt_idx),1))

                # Get points (already expressed in (left) camera frame)
                xyz = np.asarray(rockpcd.points)

                # Project points into image
                pix, _ = cv.projectPoints(xyz, (0,0,0), (0,0,0), K0, D0)
                pix = pix.squeeze()

                C = tuple(map(lambda x: int(x*255), clr))[::-1]
                for p in pix:
                    cv.circle(dbg0, (int(p[0]), int(p[1])), 5, C, -1)

                np.asarray(pcd_nonground.colors)[rock_pt_idx] = clr

            if visualize_img:
                cv.imshow(self.cam, dbg0)
                cv.imshow(f"{self.cam} orig", img0)
                cv.waitKey(1)

            if visualize_pcd:
                pcd_ground.paint_uniform_color([1,0,0])
                Fg = o3d.geometry.TriangleMesh.create_coordinate_frame()
                Fc0 = o3d.geometry.TriangleMesh.create_coordinate_frame().transform(T_gc0)
                o3d.visualization.draw_geometries([Fg, Fc0, pcd_nonground])

        return rocks, dbg0 if generate_debug_img else img0