import cv2 as cv
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tqdm import tqdm

import rospy
import rosbag
import cv_bridge

import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs


class BagRecorder:
    def __init__(self, dataset, outbag):
        self.dataset = dataset
        self.outbag = outbag
        self.bag = rosbag.Bag(outbag.as_posix(), 'w')
        self.bridge = cv_bridge.CvBridge()

        print(f"Opened bag for writing: {self.outbag.as_posix()}")

    def _ensure_valid(self, topic):
        return topic if topic[0] == '/' else f"/{topic}"

    def _build_cinfo(self, cam, i):
        K, D, R, P = self.dataset.get_stereo_calib(cam, i)

        cinfo = sensor_msgs.CameraInfo()
        cinfo.distortion_model = 'plumb_bob'
        cinfo.K = K.flatten().tolist()
        cinfo.D = np.r_[D, 0].tolist()
        cinfo.R = R.flatten().tolist()
        cinfo.P = P.flatten().tolist()

        return cinfo

    def _record_stereo(self, cam, topic):
        cinfo0 = self._build_cinfo(cam, 0)
        cinfo1 = self._build_cinfo(cam, 1)

        camdata = self.dataset.get_stereo_camdata(cam)
        
        for t_s, img0, img1 in tqdm(camdata, desc=topic):
            rostime = rospy.Time.from_sec(t_s)

            msg0 = self.bridge.cv2_to_imgmsg(img0, "bgr8")
            msg0.header.frame_id = f"{cam}0"
            msg0.header.stamp = rostime
            cinfo0.header = msg0.header
            cinfo0.height, cinfo0.width = img0.shape[:2]

            msg1 = self.bridge.cv2_to_imgmsg(img1, "bgr8")
            msg1.header.frame_id = f"{cam}1"
            msg1.header.stamp = rostime
            cinfo1.header = msg1.header
            cinfo1.height, cinfo1.width = img1.shape[:2]

            self.bag.write(self._ensure_valid(f"{topic}/left/image_raw"), msg0, rostime)
            self.bag.write(self._ensure_valid(f"{topic}/left/camera_info"), cinfo0, rostime)
            self.bag.write(self._ensure_valid(f"{topic}/right/image_raw"), msg1, rostime)
            self.bag.write(self._ensure_valid(f"{topic}/right/camera_info"), cinfo1, rostime)

    def record_gpsutm(self, topic='ground_truth_pose'):
        # TODO: do coordinate frames make sense?
        initialized = False
        for utm in tqdm(self.dataset.gpsutm, desc=topic):
            t_s, _, n, e, h, _, _, _ = utm
            rostime = rospy.Time.from_sec(t_s)

            if not initialized:
                initialized = True
                x0, z0 = n, e

            msg = geometry_msgs.PoseStamped()
            msg.header.frame_id = 'world'
            msg.header.stamp = rostime
            msg.pose.position.x = n - x0
            msg.pose.position.y = 0
            msg.pose.position.z = e - z0
            msg.pose.orientation.x = 0
            msg.pose.orientation.y = 0
            msg.pose.orientation.z = 0
            msg.pose.orientation.w = 1
            self.bag.write(self._ensure_valid(topic), msg, rostime)

    def record_velodyne(self, topic='velodyne'):
        pass

    def record_imu(self, topic='imu'):
        # TODO: do coordinate frames make sense?
        for imu in tqdm(self.dataset.imu, desc=topic):
            t_s, ax, ay, az, gx, gy, gz, _, _, _ = imu
            rostime = rospy.Time.from_sec(t_s)

            msg = sensor_msgs.Imu()
            msg.header.frame_id = 'imu'
            msg.header.stamp = rostime
            msg.linear_acceleration.x = ax
            msg.linear_acceleration.y = ay
            msg.linear_acceleration.z = az
            msg.angular_velocity.x = gx
            msg.angular_velocity.y = gy
            msg.angular_velocity.z = gz
            self.bag.write(self._ensure_valid(topic), msg, rostime)


    def record_loccam(self, topic='LocCam'):
        self._record_stereo('LocCam', topic)

    def record_pancam(self, topic='PanCam'):
        self._record_stereo('PanCam', topic)

    def close(self):
        self.bag.close()
        print(f"Recorded bag to {self.outbag.as_posix()}")
