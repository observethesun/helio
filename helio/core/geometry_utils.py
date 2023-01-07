"""Utils for geometry tranformation."""
import numpy as np
from scipy.linalg import expm
from skimage.transform import rotate

def xy_to_xyz(xy, rad):
    """Map points from plane onto sphere.
    |
    y
    |
    ----x-axis-----

    Parameters
    ----------
    xy : ndarray
        Array of (x, y) coordinates. (0, 0) is solar disk center.
    rad : scalar
        The radius of sphere.

    Returs
    ------
    xyz : ndarray
        Array of (x, y, z) coordinates.
    """
    xy = xy / rad
    y = xy[:, 0]
    z = xy[:, 1]
    x = np.sqrt(1 - y*y - z*z)
    return np.array([x, y, z]).T

def xyz_to_sp(xyz, deg=True):
    """Map xyz coordinates on a sphere to longitude and latitude.

    Parameters
    ----------
    xyz : ndarray
        (x, y, z) coordinates on a sphere. (0, 0, 0) is sphere center.
    deg : bool
        Return in degrees. Default to True.

    Returns
    -------
    arr : ndarray
        Long and lat. Both in (-pi/2 to pi/2). Latitude is positive to the North."""
    x, y, z = xyz.T
    lat = np.arcsin(z)
    lng = np.pi/2 - np.arctan2(x, y)
    if deg:
        lat = np.rad2deg(lat)
        lng = np.rad2deg(lng)
    return np.array([lng, lat]).T

def sp_to_xyz(coords, deg=True):
    """Spherical coordinates in long, lat to cartesian xyz.

    Parameters
    ----------
    coords : ndarray
        Long and lat coordinates. Both in (-pi/2 to pi/2).
        Latitude is positive to the North.
    deg : bool
        If True, all angles are in degrees. Default to True.

    Returns
    -------
    xyz : ndarray
        Cartesian xyz coordinates.
    """
    if deg:
        coords = np.deg2rad(coords)
    phi, theta = coords.T
    z = np.sin(theta)
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    return np.array([x, y, z]).T

def sp_to_xy(coords, deg=True):
    """Map long, lat to cartesian xy and mask front side point.
    |
    y
    |
    ----x-axis-----

    Parameters
    ----------
    coords : ndarray
        Long and lat coordinates. Long is in (-pi, pi).
        Lat is in (-pi/2 to pi/2), positive to the North.
    deg : bool
        If True, all angles are in degrees. Default to True.

    Returns
    -------
    (xy, valid) : tuple
        Cartesian xy coordinates and mask of front side point.
    """
    xyz = sp_to_xyz(coords, deg=deg)
    x = xyz[:, 1]
    y = xyz[:, 2]
    valid = ~(xyz[:, 0] < 0)
    return np.array([x, y]).T, valid

def rotation_matrix(axis, theta, deg=True):
    """Rotation matrix along `axis` on angle `theta`."""
    if deg:
        theta = np.deg2rad(theta)
    return expm(np.cross(np.eye(3), axis/np.linalg.norm(axis)*theta))

def rotate_B0(xyz, B0, deg=True): #pylint: disable=invalid-name
    """Rotate xyz so that z=0 corresponds to actual solar equator.

    Parameters
    ----------
    xyz : ndarray
        (x, y, z) coordinates on a sphere. (0, 0, 0) is sphere center.
    B0 : scalar
        Heliographic latitude of disk center.

    Returns
    -------
    xyz : ndarray
        Rotated coordinates.
    """
    rmat = rotation_matrix(np.array([0, 1, 0]), -B0, deg=deg)
    return np.dot(rmat, xyz.T).T

def rotate_at_center(data, angle, center=None, deg=True, labels=False, background=0, **kwargs):
    """Rotate disk image to P=0 around disk center.

       Parameters
       ----------
       data : array
           Array to rotate.
       angle : scalar
           Rotation angle.
       center : tuple
           Rotation center.
       deg : bool
           Angles are in degrees. Default True.
       labels : bool
           Data contains labels. Default False.
       background : scalar
           Background label.
       kwargs : misc
           Any additional named arguments to ``skimage.transform.rotate`` method.

       Returns
       -------
       data : array
       Rotated array.
       """
    angle = angle if deg else np.rad2deg(angle)
    if labels:
        res = np.full_like(data, background)
        for lbl in np.unique(data):
            if lbl == background:
                continue
            mask = data == lbl
            mask = rotate(mask, angle, center=center, preserve_range=True, **kwargs) > 0.5
            res[mask] = lbl
        return res
    is_bool = data.dtype == np.bool
    data = rotate(data, angle, center=center, preserve_range=True, **kwargs)
    return data > 0.5 if is_bool else data

def xy_to_carr(xy, rad, B0, L0, deg=True):
    """Get carrington coordinates from xy.
    |
    y  (x0, y0)
    |
    ----x-axis-----

    Parameters
    ----------
    xy : ndarray
        Array of (x, y) coordinates. (0, 0) is solar disk center.
    rad : scalar
        The radius of sphere.
    B0 : scalar
        Heliographic latitude of disk center.
    L0 : scalar
        Carringon longitude of central meridian.
    deg : bool
        If True, all angles are in degrees. Default to True.

    Returns
    -------
    carr : ndarray
        Carrington coordinates (Long, Lat).
    """
    xyz = xy_to_xyz(xy, rad=rad)
    xyz = rotate_B0(xyz, B0, deg=deg)
    carr = xyz_to_sp(xyz, deg=deg)
    carr = carr + np.array([L0, 0])
    carr[:, 0] = carr[:, 0] % (360 if deg else 2*np.pi)
    return carr

def rotate_sphere_B0(coords, B0, deg=True): #pylint: disable=invalid-name
    """Rotate long, lat so that lat=0 corresponds to actual solar equator.

    Parameters
    ----------
    coords : ndarray
        Long and lat coordinates. Both in (-pi/2 to pi/2).
        Latitude is positive to the North.
    B0 : scalar
        Heliographic latitude of disk center.
    deg : bool
        If True, all angles are in degrees. Default to True.

    Returns
    -------
    coords : ndarray
        Rotated coordinates.
    """
    xyz = sp_to_xyz(coords, deg=deg)
    xyz = rotate_B0(xyz, B0, deg=deg)
    return xyz_to_sp(xyz, deg=deg)

def rotate_sphere_L0(coords, L0, deg=True): #pylint: disable=invalid-name
    """Rotate long, lat so that central meridian is on L0 longitude.

    Parameters
    ----------
    coords : ndarray
        Long and lat coordinates. Both in (-pi/2 to pi/2) or (-90, 90).
        Latitude is positive to the North.
    L0 : scalar
        Carringon longitude of central meridian.
    deg : bool
        If True, all angles are in degrees. Default to True.

    Returns
    -------
    coords : ndarray
        Rotated coordinates.
    """
    coords = coords - np.array([L0, 0])
    coords[:, 0] = coords[:, 0] % (360 if deg else 2*np.pi)
    return coords
