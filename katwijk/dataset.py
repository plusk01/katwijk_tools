import csv
import datetime
import pathlib

import cv2 as cv
import numpy as np
import scipy
import utm

from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot

from . import trajectory
from . import rocks
from . import transforms as tfs

def load(root, traverse, part):
    calibdir = pathlib.Path(__file__).absolute().parent.parent / 'calib'
    return KatwijkDataset(root, traverse, part, calibdir)


class KatwijkDataset:
    def __init__(self, root, traverse, part, calibdir):
        self.root = root
        self.traverse = traverse
        self.part = part
        self.datadir = root / f"Traverse{traverse}" / f"Part{part}"
        self.calibdir = calibdir

        self._paths = {
            'LocCam': self.datadir / 'LocCam',
            'LocCamCalib': self.calibdir / 'LocCam_calibstruct.mat',
            'PanCam': self.datadir / 'PanCam',
            'PanCamCalib': self.calibdir / 'PanCam_calibstruct.mat',
            'ToF': self.datadir / 'ToF',
            'Velodyne': self.datadir / 'Velodyne',
            'GPS-LLA': self.datadir / 'gps-latlong.txt',
            'GPS-UTM': self.datadir / 'gps-utm31.txt',
            'IMU': self.datadir / 'imu.txt',
            'WheelOdom': self.datadir / 'odometry.txt',
            'PTU': self.datadir / 'ptu.txt',
            'Rocks': {
                0: self.root / 'Rock Data' / self.__rockdata('small', traverse),
                1: self.root / 'Rock Data' / self.__rockdata('medium', traverse),
                2: self.root / 'Rock Data' / self.__rockdata('large', traverse),
            }
        }

        self._data = {}

        # Load (or at least prepare to load) data into memory.
        self._load_imu()
        self._load_gpsutm()
        self._load_ptu()
        self._load_wheelodom()
        self._load_rocks()
        self._load_pancam()
        self._load_loccam()

        self._build_gpstraj()

    @staticmethod
    def __rockdata(sz, traverse):
        """
        Get rock data path
        
        Parameters
        ----------
        sz : str -- 'small', 'medium', 'large' -- rock size
        traverse : int -- 1, 2, 3

        Returns
        -------
        file : pathlib.Path -- relative file path to rocks txt
        """
        if traverse == 1 or traverse == 2:
            file = pathlib.Path('Rock Positions Traverse 1 and 2') / f"{sz}-rocks-traverse12.txt"
        elif traverse == 3:
            file = pathlib.Path('Rock Positions Traverse 3')  / f"{sz}-rocks-traverse3.txt"
        else:
            raise Exception(f"Traverse '{traverse}' not valid")
        return file

    @staticmethod
    def __date_converter(e):
        """
        Date converter

            b'2015_11_26_12_53_35_149' --> 1448560415.149
        
        Parameters
        ----------
        e : entry from data csv file

        Returns
        -------
        t : Unix timestamp
        """
        x = datetime.datetime.strptime(e.decode(), '%Y_%m_%d_%H_%M_%S_%f')
        return x.timestamp()

    @staticmethod
    def __skip(e):
        return -1

    def _load_imu(self):
        data = np.loadtxt(self._paths['IMU'], converters={0: self.__date_converter})
        self._data['IMU'] = data

    def _load_gpsutm(self):
        data = np.loadtxt(self._paths['GPS-UTM'], converters={0: self.__date_converter, 1: self.__skip})

        # For whatever reason the dataset is broken and the UTM data is actually lla
        easting, northing, _, _ = utm.from_latlon(data[:,2], data[:,3])
        data[:,2] = easting
        data[:,3] = northing

        self._data['GPS-UTM'] = data

    def _load_ptu(self):
        data = np.loadtxt(self._paths['PTU'], converters={0: self.__date_converter})
        self._data['PTU'] = data

    def _load_wheelodom(self):
        data = np.loadtxt(self._paths['WheelOdom'], converters={0: self.__date_converter})
        self._data['WheelOdom'] = data

    def _load_rocks(self):
        data = {}
        for s in rocks.SIZE_NUM:
            xy = np.loadtxt(self._paths['Rocks'][s])
            data[s] = np.c_[xy, np.zeros((len(xy,)))]
        self._data['Rocks'] = data

    def _load_stereo(self, cam):
        cam0 = sorted(self._paths[cam].glob('*_0.png'))
        cam1 = sorted(self._paths[cam].glob('*_1.png'))
        self._data[cam] = {'cam0': cam0, 'cam1': cam1}


        stereoParams = scipy.io.loadmat(self._paths[f"{cam}Calib"], simplify_cells=True)['stereoParams']

        # Load camera intrinsic calibration details
        K0 = stereoParams['CameraParameters1']['IntrinsicMatrixInternal']
        D0 = np.r_[stereoParams['CameraParameters1']['RadialDistortion'],
                    stereoParams['CameraParameters1']['TangentialDistortion']]
        K1 = stereoParams['CameraParameters2']['IntrinsicMatrixInternal']
        D1 = np.r_[stereoParams['CameraParameters2']['RadialDistortion'],
                    stereoParams['CameraParameters2']['TangentialDistortion']]

        # Load extrinsic calib details
        T_12 = np.eye(4)
        T_12[:3,3] = stereoParams['TranslationOfCamera2'] * 1e-3
        T_12[:3,:3] = stereoParams['RotationOfCamera2']
        T_21 = np.linalg.inv(T_12)

        w, h = cv.imread(cam0[0].as_posix()).shape[:2]

        R0, R1, P0, P1, Q, ROI0, ROI1 = cv.stereoRectify(K0, D0, K1, D1, (w, h), T_21[:3,:3], T_21[:3,3])

        self._data[f"{cam}Calib"] = {
                0: {'K': K0, 'D': D0, 'R': R0, 'P': P0},
                1: {'K': K1, 'D': D1, 'R': R1, 'P': P1},
                'Q': Q
            }

    def _load_pancam(self):
        self._load_stereo('PanCam')

    def _load_loccam(self):
        self._load_stereo('LocCam')

    def _get_stereo_at_idx(self, cam, idx):
        cam0, cam1 = self._data[cam]['cam0'], self._data[cam]['cam1']
        f0, f1 = cam0[idx], cam1[idx]

        t0_s = self.__date_converter(f0.stem[7:-2].encode('ascii'))
        t1_s = self.__date_converter(f1.stem[7:-2].encode('ascii'))
        assert t0_s == t1_s, "Left and right stereo cam files out of sync"

        img0 = cv.imread(f0.as_posix())
        img1 = cv.imread(f1.as_posix())

        yield t0_s, img0, img1

    def _get_stereo(self, cam):
        for i in range(len(self._data[cam]['cam0'])):
            return self._get_stereo_at_idx(cam, i)

    def _get_stereo_timestamps(self, cam):
        return [self.__date_converter(f.stem[7:-2].encode('ascii')) for f in self._data[cam]['cam0']]

    def _build_gpstraj(self):
        t = self._data['GPS-UTM'][:,0]
        e = self._data['GPS-UTM'][:,2]
        n = self._data['GPS-UTM'][:,3]
        self._gpstraj = trajectory.NavState2DTrajectory(t, e, n)

    def get_rocks(self, s):
        """
        Ground truth rock positions

        Parameters
        ----------
        s : int -- rock size (rocks.SIZE_NUM)

        Returns
        -------
        data : (n,3) np.array -- x, y, z rock positions (z=0)
        """
        return self._data['Rocks'][s]

    @property
    def imu(self):
        """
        IMU packets

            timestamp, ax, ay, az, gx, gy, gz, iax, iay, iaz

        where ia is from inclinometer. Units are in m/s^2 and rad/s.

        Returns
        -------
        data : (10,) np.array
        """
        yield from self._data['IMU']

    @property
    def gpsutm(self):
        """
        GPS packets

            timestamp, -1, easting, northing, altitute, n stddev, e stddev, a stddev

        Units are in m.

        Returns
        -------
        data : (8,) np.array
        """
        yield from self._data['GPS-UTM']

    @property
    def ptu(self):
        """
        PTU packets

            timestamp, pan (rad), tilt (rad)

        Returns
        -------
        data : (3,) np.array
        """
        yield from self._data['PTU']

    @property
    def wheelodom(self):
        """
        Wheel odometry

            timestamp,
                fl, fr, cl, cr, bl, br wheel angular displacement (rad),
                fl, fr, bl, br steering angle (rad),
                rocker, left bogie, right bogie orientation (rad)

            whaaaat? this returns a (34,) ... ?

        Returns
        -------
        data : (14,)
        """
        yield from self._data['WheelOdom']

    @property
    def pancam(self):
        return self._get_stereo('PanCam')

    @property
    def pancam_timestamps(self):
        return self._get_stereo_timestamps('PanCam')

    @property
    def loccam(self):
        return self._get_stereo_timestamps('LocCam')

    @property
    def loccam_timestamps(self):
        return [self.__date_converter(f.stem[7:-2].encode('ascii')) for f in self._data['LocCam']['cam0']]

    def get_stereo_camdata(self, cam):
        if cam == 'PanCam':
            return self.pancam
        elif cam == 'LocCam':
            return self.loccam
        else:
            raise Exception(f"'{cam}' is not a valid stereo camera")

    def get_ground_truth_at(self, t, parent='imu_odom', child='LocCam0'):
        """
        Returns the ground truth pose of GPS at the requested time.

        The position is of the GPS unit w.r.t the UTM map (ENU) origin. The
        heading is of the robot's forward direction (assuming velocity is only
        ever in body-x, i.e., a non-holonomic vehicle) w.r.t the x-axis of
        the UTM map---e.g., the east direction (ENU).
        
        Parameters
        ----------
        t : float -- time [s] at which to get ground truth

        Returns
        -------

        """
        # initial position of the GPS w.r.t the GPS-UTM map
        T_map_g0 = self.get_T_map_g0()

        xy, θ, v = self._gpstraj.sample_navstate(t)
        T_map_gps = trajectory.to_pose3d(xy, θ)

        # gps w.r.t imu
        T_zg = tfs.T_zg

        # LocCam left (0) w.r.t IMU
        T_z_lc0 = tfs.T_z_lc0

        N = len(T_map_gps)

        # parent = 'gps_odom'
        # child = 'gps'

        # parent = 'imu_odom'
        # child = 'imu'

        # parent = 'cam_odom'
        # child = 'LocCam0'

        if parent == 'map' and child == 'gps':
            return T_map_gps

        if parent == 'map':
            T_pg = T_map_g0

        elif parent == 'gps_odom':
            T_pg = np.eye(4)

        elif parent == 'imu_odom':
            T_pg = T_zg

        elif parent == 'LocCam0_odom':
            T_pg = np.linalg.inv(T_z_lc0) @ T_zg

        else:
            raise NotImplemented(f"Unknown parent frame '{parent}'")

        if child == 'gps':
            T_gc = np.eye(4)

        elif child == 'imu':
            T_gc = np.linalg.inv(T_zg)

        elif child == 'LocCam0':
            T_gc = np.linalg.inv(T_zg) @ T_z_lc0

        else:
            raise NotImplemented(f"Unknown child frame '{child}'")


        T_pc = np.array([np.eye(4)] * N)
        for i in range(N):
            T_pc[i] = T_pg @ np.linalg.inv(T_map_g0) @ T_map_gps[i] @ T_gc


        return T_pc

    def get_stereo_calib(self, cam, i):
        """
        Gets the stereo calib.
        
        Parameters
        ----------
        cam : str -- 'PamCam' or 'LocCam'
        i : int -- 0 (left) or 1 (right)

        Returns - http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
        -------
        K : (3,3) np.array -- camera intrinsics
        D : (4,) np.array -- distortion coeffs (k1, k2, t1, t2)
        R : (3,3) np.array -- rotation matrix for rectification
        P : (3,4) np.array -- projection / camera matrix
        """
        return (self._data[f"{cam}Calib"][i][key] for key in ('K', 'D', 'R', 'P'))

    def get_stereo_Q(self, cam):
        """
        Gets the Q matrix from the stereo rectification process.
        
        There is likely a way to construct Q from K, D, R, P, but...

        Parameters
        ----------
        cam : str -- 'PanCam' or 'LocCam'

        Returns
        -------
        Q : (4,4) np.array -- disparity-to-depth mapping matrix for reprojectImageTo3D
        """
        return self._data[f"{cam}Calib"]['Q']

    def get_stereo_T_01(self, cam):
        """
        Gets the transform of the right camera w.r.t the left camera.
        
        Parameters
        ----------
        cam : str -- 'PanCam' or 'LocCam'

        Returns
        -------
        T_01 : (4,4) np.array -- SE(3) of right cam (1) w.r.t left (0)
        """
        if cam == 'LocCam':
            T_01 = np.eye(4)
            T_01[:3,3] = (0.12, 0, 0)
        elif cam == 'PanCam':
            T_01 = np.eye(4)
            T_01[:3,3] = (0.5, 0, 0)
        else:
            raise Exception(f"Camera '{cam}' is invalid")
        return T_01

    def get_stereo_T_gc0(self, cam, t):
        """
        Gets the transform of the left camera w.r.t the level ground frame.

        The IMU is assumed to be roughly level w.r.t the ground, and so
        the level ground frame is actually just the IMU frame (Fz) of Katwijk.
        
        Parameters
        ----------
        cam : str -- 'PanCam' or 'LocCam'

        Returns
        -------
        T_gc0 : (4,4) np.array -- SE(3) of left cam (0) w.r.t ground
        """
        if cam == 'LocCam':
            T_gc0 = np.eye(4)
            T_gc0[:3,:3] = Rot.from_quat((0.572, 0.572, -0.416, -0.416)).as_matrix()
            T_gc0[:3,3] = (-0.382, -0.078, 0.557)
        elif cam == 'PanCam':
            # PTU fixed mount position w.r.t IMU
            T_zd = np.eye(4)
            T_zd[:3,:3] = Rot.from_quat((0,0,1,0)).as_matrix()
            T_zd[:3,3] = (-0.138, -0.005, 1.286)

            # PTU link w.r.t PTU fixed mount
            pan, tilt = self.find_closest_ptu_angles(t)
            T_dp = np.eye(4)
            T_dp[:3,:3] = Rot.from_euler('xyz', (0, tilt, pan)).as_matrix()

            # left camera w.r.t PTU link
            T_pc0 = np.eye(4)
            T_pc0[:3,:3] = Rot.from_quat((0.5, -0.5, 0.5, -0.5)).as_matrix()
            T_pc0[:3,3] = (0.01, 0.25, 0.054)

            T_gc0 = T_zd @ T_dp @ T_pc0
        else:
            raise Exception(f"Camera '{cam}' is invalid")
        return T_gc0

    def find_closest_ptu_angles(self, t):
        """
        Finds the closest PTU angles by matching timestamps.
        
        Parameters
        ----------
        t : float -- timestamp of stereo image pair
        """
        idx = np.argmin(np.abs(self._data['PTU'][:,0] - t))
        return self._data['PTU'][idx,1:]

    def get_T_map_g0(self):
        """
        Gets the transform of the initial GPS frame {g} w.r.t the UTM map.

        Note that we define the orientation of the GPS frame to be pi yaw
        of the IMU frame (e.g., it is flu).

        Returns
        -------
        T_map_g0 : (4,4) np.array -- SE(3) of GPS {g} w.r.t UTM Map
        """
        xy, θ, v = self._gpstraj.sample_navstate(0)
        T_map_g0 = trajectory.to_pose3d(xy, θ)[0]
        return T_map_g0


class Iterable:
    def __init__(self, dataset):
        self.dataset = dataset
        self.KEYS = ('IMU', 'GPS-UTM', 'PTU', 'WheelOdom',
                    'PanCam', 'LocCam', 'PanCamOdomGT', 'LocCamOdomGT')

        self.curr_idx = 0
        self.data = self._organize_data()

    def _organize_data(self):
        # assumption: all data is already chronological

        data = None
        for k, key in enumerate(self.KEYS):

            if key.endswith('Cam'):
                t = self.dataset._get_stereo_timestamps(key)
            elif key.endswith('CamOdomGT'):
                t = self.dataset._get_stereo_timestamps(key[:-6])
            else:
                t = self.dataset._data[key][:,0]
            n = len(t)
            d = np.c_[t, k * np.ones((n,)), np.arange(n)]

            # stack data
            data = d if data is None else np.r_[data, d]

        idx = data[:,0].argsort()
        return data[idx]

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_idx < len(self.data):
            t, k, i = self.data[self.curr_idx]
            key = self.KEYS[int(k)]
            self.curr_idx += 1
            if key.endswith('Cam'):
                data = self.dataset._get_stereo_at_idx(key, int(i))
            elif key.endswith('CamOdomGT'):
                data = self.dataset.get_ground_truth_at(t)[0]
            else:
                data = self.dataset._data[key][int(i), 1:]
            return t, key, data

        raise StopIteration