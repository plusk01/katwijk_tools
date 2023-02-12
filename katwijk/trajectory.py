import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy.spatial.transform import Rotation as Rot


def to_pose3d(xy, θ):
    """
    Converts positions and heading angle to a list of 3D poses
    
    Parameters
    ----------
    xy : (N,2) np.array -- position samples along the trajectory [m]
    θ : (N,) np.array -- heading angles samples along the trajectory [rad]

    Returns
    -------
    poses : List[(4,4) np.array]
    """
    N = len(θ)

    T = np.array([np.eye(4)] * N)
    rθ = (np.array([[0,0,1]]).T * θ).T
    T[:,:3,:3] = Rot.from_rotvec(rθ).as_matrix()
    T[:,:2,3] = xy

    return T


def group_consecutives(vals, step=1):
    """
    Splits up a 1d array into groups of consecutive subarrays.
    
    Parameters
    ----------
    vals : list-like 1d array of values
    step : int -- the step between values such that
                  those values are considered consecutive

    Returns
    -------
    result : list[lists] -- list of consecutive subarrays
    """
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


class NavState2DTrajectory:
    def __init__(self, t, x, y, shift=False):
        x = x - x[0] if shift else x
        y = y - y[0] if shift else y

        self.t, self.x, self.y = self._average_stationary_points(t, x, y)

        self.xs = scipy.interpolate.CubicSpline(self.t, self.x, bc_type='clamped')
        self.ys = scipy.interpolate.CubicSpline(self.t, self.y, bc_type='clamped')

        self.θ0 = self._determine_initial_heading()

    def _average_stationary_points(self, t, x, y, δ=5e-2):
        """
        Average out stationary points.

        A noisy sensor (e.g., GPS) may cause the position to jump around
        the static position. Fitting a spline through these points would
        cause undesirable behavior. This function identifies clusters of
        stationary points and averages them out.
        
        Parameters
        ----------
        t : (n,) np.array -- times
        x : (n,) np.array -- x position
        y : (n,) np.array -- y position
        δ : float -- stationary threshold on consecutive distances

        Returns
        -------
        t : (m,) np.array -- times
        x : (m,) np.array -- x position
        y : (m,) np.array -- y position
                where m <= n
        """

        # calculate the Euclidean distance between consecutive points
        d = np.diff(np.c_[x, y], axis=0)
        D = np.sqrt((d**2).sum(axis=1))

        idx, = np.where(D < δ)
        groups = group_consecutives(idx)

        for g in groups[::-1]:
            if len(g) == 0:
                continue

            j = g[-1] + 1

            t[j] = t[j] #t[g+[j]].mean()
            x[j] = x[g+[j]].mean()
            y[j] = y[g+[j]].mean()

            t = np.delete(t, g)
            x = np.delete(x, g)
            y = np.delete(y, g)

        return t, x, y

    def _determine_initial_heading(self, dt=0.01):

        ts = np.arange(self.t[0], self.t[-1], dt)
        v = self.sample(ts, 1)
        θ = np.arctan2(v[:,1], v[:,0])

        θ0 = θ[0]

        speed = np.linalg.norm(v, axis=1)
        idx, = np.where(speed < 1e-3)

        # for places with zero speed, need to fix heading
        groups = group_consecutives(idx)
        if len(groups) > 0:
            g = groups[0]
            if len(g) > 0:
                θ0 = θ[g[-1]+1]

        return θ0

    def plot_samples(self, ts):
        xy, θ, v = self.sample_navstate(ts)

        fig, ax = plt.subplots()
        # ax.plot(self.xs(ts), self.ys(ts), 'o', markersize=2)
        ax.plot(xy[:,0], xy[:,1], 'o', markersize=2)
        ax.plot(self.x, self.y, 'o', color='r', markersize=5)
        ax.axis('equal')
        ax.grid(alpha=0.3)

        fig, axes = plt.subplots(nrows=3)
        axes[0].plot(ts, xy[:,0])
        axes[0].plot(ts, xy[:,1])
        axes[1].plot(ts, v[:,0])
        axes[1].plot(ts, v[:,1])
        axes[2].plot(ts, np.rad2deg(θ))

        plt.show()

    def sample(self, ts, d=0):
        """
        Sample the trajectory at user-specified times.

        If times are sampled before the support of the spline, then the result
        is clamped to be the first data point of the spline, i.e., at t0.

        Parameters
        ----------
        ts : List[times] -- times to sample the trajectory at [s]
        d : int -- order of derivative

        Returns
        -------
        xy : (N,2) np.array -- samples along the trajectory [m]
        """
        xy = np.c_[self.xs(ts, d), self.ys(ts, d)]

        # find the i where ts[i] >= self.t[0]
        t0 = self.t[0]
        idx, = np.where(ts >= t0)

        # import ipdb; ipdb.set_trace()

        # # There were no requested sample times within the trajectory!
        # if len(idx) == 0:
        #     return None

        # everything before i was extrapolated and so should be clamped
        # i = idx[0]
        i = len(xy) if len(idx) == 0 else idx[0]
        if i > 0:
            xy[:i,:] = np.c_[self.xs(t0,d), self.ys(t0,d)]

        return xy

    def sample_navstate(self, ts):
        """
        Sample NavStates (pose, vel) along the trajectory.
        
        Parameters
        ----------
        ts : List[times] -- times to sample the trajectory at [s]

        Returns
        -------
        xy : (N,2) np.array -- position samples along the trajectory [m]
        θ : (N,) np.array -- heading angles samples along the trajectory [rad]
        v : (N,2) np.array -- velocity samples along the trajectory [m/s]
        """
        xy = self. sample(ts)
        v = self.sample(ts, 1)
        θ = np.arctan2(v[:,1], v[:,0])

        speed = np.linalg.norm(v, axis=1)
        idx, = np.where(speed < 1e-3)

        # for places with zero speed, need to fix heading
        groups = group_consecutives(idx)
        for g in groups:
            if len(g) == 0:
                continue
            if g[0] == 0:
                # figure out initial heading...
                θ[g] = self.θ0

        return xy, θ, v


    def write_trajectory(self, filepath, ts):
        """
        Write a trajectory to file.

        Parameters
        ----------
        filepath : pathlib.Path -- path to new trajectory file
        ts : List[times] -- times to sample the trajectory at [s]
        """
        xy, θ, v = self.sample_navstate(ts)
        
        # convert timestamps to strings
        fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        t_str = list(map(fmt, t))

        with open(filepath, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=' ')
            for i, ts in enumerate(t_str):
                writer.writerow((ts, *xy[i], θ[i], *v[i]))