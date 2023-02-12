import numpy as np

from scipy.spatial.transform import Rotation as Rot


#
# GPS {g} w.r.t IMU {z}
#

# n.b., we assume that the GPS has (flu) axes (pi yaw w.r.t {z})

T_zg = np.eye(4)
T_zg[:3,:3] = Rot.from_euler('xyz', (0, 0, np.pi)).as_matrix()
T_zg[:3,3] = (0.2209, -0.0515, 1.5785)

#
# LocCam left {lc0} w.r.t IMU {z}
#

T_z_lc0 = np.eye(4)
T_z_lc0[:3,:3] = Rot.from_quat((0.572, 0.572, -0.416, -0.416)).as_matrix()
T_z_lc0[:3,3] = (-0.382, -0.078, 0.557)