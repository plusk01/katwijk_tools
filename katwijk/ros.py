import cv2 as cv
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot

import rospy
import rosbag
import cv_bridge
import tf2_ros

import std_msgs.msg as std_msgs
import sensor_msgs.msg as sensor_msgs
import geometry_msgs.msg as geometry_msgs
import nav_msgs.msg as nav_msgs
import visualization_msgs.msg as visualization_msgs

from . import rocks
from . import transforms as tfs

# tf2 broadcasters
_tf2 = None
_tf2_static = None

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

    def record_gpsutm(self, topic='gpsutm'):
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

    def record_groundtruth_odom(self, ts, topic='odom_gt'):

        xy, θ, v = self.dataset.get_ground_truth_at(ts)

        for data in tqdm(np.c_[ts, xy, θ, v], desc=topic):
            t, x, y, th, vx, vy = data
            rostime = rospy.Time.from_sec(t)

            q = Rot.from_rotvec(th * np.array((0,0,1))).as_quat()

            msg = navigation_msgs.Odometry()
            msg.header.frame_id = 'world'
            msg.header.stamp = rostime
            msg.child_frame_id = 'camera'
            msg.pose.pose.position.x = x
            msg.pose.pose.position.y = y
            msg.pose.pose.position.z = 0
            msg.pose.pose.orientation.x = q[0]
            msg.pose.pose.orientation.y = q[1]
            msg.pose.pose.orientation.z = q[2]
            msg.pose.pose.orientation.w = q[3]
            msg.twist.twist.linear.x = np.linalg.norm([vx, vy])
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


def publish_static_tf(dataset):
    global _tf2_static
    if _tf2_static is None:
        _tf2_static = tf2_ros.StaticTransformBroadcaster()


    T_map_g0 = dataset.get_T_map_g0()
    q = Rot.from_matrix(T_map_g0[:3,:3]).as_quat()

    msg = geometry_msgs.TransformStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = 'map'
    msg.child_frame_id = 'gps0'

    msg.transform.translation.x = T_map_g0[0,3]
    msg.transform.translation.y = T_map_g0[1,3]
    msg.transform.translation.z = T_map_g0[2,3]
    msg.transform.rotation.x = q[0]
    msg.transform.rotation.y = q[1]
    msg.transform.rotation.z = q[2]
    msg.transform.rotation.w = q[3]

    _tf2_static.sendTransform(msg)




    T_g0_odom = np.linalg.inv(tfs.T_zg)
    q = Rot.from_matrix(T_g0_odom[:3,:3]).as_quat()

    msg = geometry_msgs.TransformStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = 'gps0'
    msg.child_frame_id = 'odom'

    msg.transform.translation.x = T_g0_odom[0,3]
    msg.transform.translation.y = T_g0_odom[1,3]
    msg.transform.translation.z = T_g0_odom[2,3]
    msg.transform.rotation.x = q[0]
    msg.transform.rotation.y = q[1]
    msg.transform.rotation.z = q[2]
    msg.transform.rotation.w = q[3]

    _tf2_static.sendTransform(msg)


def publish_tf(dataset, t, T, parent='odom', child='LocCam'):
    global _tf2
    if _tf2 is None:
        _tf2 = tf2_ros.TransformBroadcaster()

    q = Rot.from_matrix(T[:3,:3]).as_quat()

    msg = geometry_msgs.TransformStamped()
    msg.header.stamp = rospy.Time.from_sec(t)
    msg.header.frame_id = parent
    msg.child_frame_id = child

    msg.transform.translation.x = T[0,3]
    msg.transform.translation.y = T[1,3]
    msg.transform.translation.z = T[2,3]
    msg.transform.rotation.x = q[0]
    msg.transform.rotation.y = q[1]
    msg.transform.rotation.z = q[2]
    msg.transform.rotation.w = q[3]

    _tf2.sendTransform(msg)

def build_odom_msg(t, T, parent='odom', child='LocCam'):
    q = Rot.from_matrix(T[:3,:3]).as_quat()

    msg = nav_msgs.Odometry()
    msg.header.stamp = rospy.Time.from_sec(t)
    msg.header.frame_id = parent
    msg.child_frame_id = child

    msg.pose.pose.position.x = T[0,3]
    msg.pose.pose.position.y = T[1,3]
    msg.pose.pose.position.z = T[2,3]
    msg.pose.pose.orientation.x = q[0]
    msg.pose.pose.orientation.y = q[1]
    msg.pose.pose.orientation.z = q[2]
    msg.pose.pose.orientation.w = q[3]

    return msg


def build_rock_marker_array(dataset, frame_id='map'):
    msg = visualization_msgs.MarkerArray()

    scales = (0.5, 1, 2)

    for s in rocks.SIZE_NUM:
        clr = rocks.SIZE_CLR[s]
        for i, r in enumerate(dataset.get_rocks(s)):
            m = visualization_msgs.Marker()
            m.header.frame_id = frame_id
            m.header.stamp = rospy.Time.now()
            m.ns = 'rocks'
            m.id = len(msg.markers)
            m.type = visualization_msgs.Marker.SPHERE
            m.action = visualization_msgs.Marker.ADD
            m.pose.position.x = r[0]
            m.pose.position.y = r[1]
            m.pose.position.z = r[2]
            m.pose.orientation.w = 1
            m.scale.x = m.scale.y = m.scale.z = scales[s]
            m.color.r = clr[0]
            m.color.g = clr[1]
            m.color.b = clr[2]
            m.color.a = 1

            msg.markers.append(m)

    return msg

def o3d_to_ros(pcd):
    """
    Converts an Open3D point cloud to a ROS PointCloud2.
    
    Parameters
    ----------
    pcd : o3d.geometry.PointCloud

    Returns
    -------
    msg : sensor_msgs.PointCloud2
    """

    pts = np.asarray(pcd.points)
    clr = np.asarray(pcd.colors)
    ptsrgb = np.c_[pts, clr]

    dtype = np.float32
    data = ptsrgb.astype(dtype).tobytes()
    itemsize = np.dtype(dtype).itemsize

    fields = [sensor_msgs.PointField(
        name=n, offset=i*itemsize, datatype=sensor_msgs.PointField.FLOAT32, count=1)
        for i, n in enumerate('xyzrgb')]

    return sensor_msgs.PointCloud2(
        height=1,
        width=ptsrgb.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 6),
        row_step=(itemsize * 6 * ptsrgb.shape[0]),
        data=data
    )
    