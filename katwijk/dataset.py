import datetime
import pathlib

import cv2 as cv
import numpy as np
import scipy
import utm

from tqdm import tqdm
from scipy.spatial.transform import Rotation as Rot


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
                's': self.root / 'Rock Data' / self.__rockdata('small', traverse),
                'm': self.root / 'Rock Data' / self.__rockdata('medium', traverse),
                'l': self.root / 'Rock Data' / self.__rockdata('large', traverse),
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
        northing, easting, _, _ = utm.from_latlon(data[:,2], data[:,3])
        data[:,2] = northing
        data[:,3] = easting

        self._data['GPS-UTM'] = data

    def _load_ptu(self):
        data = np.loadtxt(self._paths['PTU'], converters={0: self.__date_converter})
        self._data['PTU'] = data

    def _load_wheelodom(self):
        data = np.loadtxt(self._paths['WheelOdom'], converters={0: self.__date_converter})
        self._data['WheelOdom'] = data

    def _load_rocks(self):
        data = {}
        for s, size in enumerate(('s', 'm', 'l')):
            data[s] = np.loadtxt(self._paths['Rocks'][size])
            data[s] = np.c_[s*np.ones((len(data[s]),)), data[s]]
        self._data['Rocks'] = np.r_[data[0], data[1], data[2]]

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

    def _get_stereo(self, cam):
        cam0, cam1 = self._data[cam]['cam0'], self._data[cam]['cam1']
        for f0, f1 in zip(cam0, cam1):
            t0_s = self.__date_converter(f0.stem[7:-2].encode('ascii'))
            t1_s = self.__date_converter(f1.stem[7:-2].encode('ascii'))
            assert t0_s == t1_s, "Left and right stereo cam files out of sync"

            img0 = cv.imread(f0.as_posix())
            img1 = cv.imread(f1.as_posix())

            yield t0_s, img0, img1

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

            timestamp, -1, northing, easting, altitute, n stddev, e stddev, a stddev

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
    def rocks(self):
        """
        Rocks

            size (0, 1, 2), northing, easting

        Returns
        -------
        data : (N,3)
        """
        return self._data['Rocks']

    @property
    def pancam(self):
        return self._get_stereo('PanCam')

    @property
    def loccam(self):
        return self._get_stereo('LocCam')

    def get_stereo_camdata(self, cam):
        if cam == 'PanCam':
            return self.pancam
        elif cam == 'LocCam':
            return self.loccam
        else:
            raise Exception(f"'{cam}' is not a valid stereo camera")

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
