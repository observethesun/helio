"""Methods for synoptic map construction."""
import numpy as np
import pandas as pd
from skimage.measure import label

from .geometry_utils import rotate_sphere_B0, rotate_sphere_L0, sp_to_xy, xy_to_carr

SOLAR_SURFACE_AREA = 6.09

def map_syn_to_disk(i_cen, j_cen, rad, B0, L0, bins, deg=True):
    """Map synoptic map to solar disk.

    |
    i
    |
    ----j----

    Parameters
    ----------
    i_cen : int
        Row index of solar disk center.
    j_cen : int
        Column index of solar disk center.
    rad : int
        Solar disk radius in pixels.
    B0 : scalar
        B0 angle of solar equator.
    L0 : scalar
        Carringon longitude of central meridian.
    bins : (nx, ny)
        Resolution of synoptic map.
    deg : bool
        If True, all angles are in degrees. Default to True.

    Returns
    -------
    (image_ind, synoptic_ind) : tuple
        Image indices and corresponding synoptic indices.
    """
    nx, ny = bins
    xedges = np.linspace(0, 360, nx + 1)
    xc = (xedges[:-1] + xedges[1:]) / 2
    yedges = np.linspace(90, -90, ny + 1)
    yc = (yedges[:-1] + yedges[1:]) / 2

    centres = np.array(np.meshgrid(xc, yc)).reshape(2, -1).T
    syn_ind = np.moveaxis(np.indices((nx, ny)), [0, 1], [-1, 1]).reshape(-1, 2)

    centres = rotate_sphere_L0(centres, L0, deg=deg)
    centres = rotate_sphere_B0(centres, -B0, deg=deg)
    xy, valid = sp_to_xy(centres, deg=deg)

    xy_proj = xy[valid] * rad

    ind_i = np.floor(-xy_proj[:, 1] + i_cen).astype(int)
    ind_j = np.floor(xy_proj[:, 0] + j_cen).astype(int)

    syn_ind = syn_ind[valid]
    syn_ind = np.array([syn_ind[:, 1], syn_ind[:, 0]]).T

    return np.array([ind_i, ind_j]).T, syn_ind

def map_disk_to_syn(i_cen, j_cen, rad, B0, L0, bins, deg=True):
    """Map solar disk to synoptic map.

    |
    i
    |
    ----j----

    Parameters
    ----------
    i_cen : int
        Row index of solar disk center.
    j_cen : int
        Column index of solar disk center.
    rad : int
        Solar disk radius in pixels.
    B0 : scalar
        B0 angle of solar equator.
    L0 : scalar
        Carringon longitude of central meridian.
    bins : (nx, ny)
        Resolution of synoptic map.
    deg : bool
        If True, all angles are in degrees. Default to True.

    Returns
    -------
    (image_ind, synoptic_ind) : tuple
        Image indices and corresponding synoptic indices.
    """
    nx, ny = bins
    ind = np.moveaxis(np.indices((2*rad + 1, 2*rad + 1)), 0, -1)
    r = np.linalg.norm(ind - np.array([rad, rad]), axis=-1)
    disk = np.vstack(np.where(r < rad)).T
    xy = np.array([disk[:, 1] - rad, -disk[:, 0] + rad]).T

    if not deg:
        B0 = np.rad2deg(B0)
        L0 = np.rad2deg(L0)

    carr = xy_to_carr(xy, rad, B0=B0, L0=L0, deg=True)

    row_ind = np.digitize(carr[:, 1], np.linspace(-90, 90, ny + 1)) - 1
    row_ind = ny - 1 - row_ind
    col_ind = np.digitize(carr[:, 0], np.linspace(0, 360, nx + 1)) - 1

    return disk + np.array([i_cen - rad, j_cen - rad]), np.array([row_ind, col_ind]).T

def disk_and_syn_mapping(i_cen, j_cen, rad, B0, L0, bins, deg=True):
    """Bijective map of solar disk image and synoptic map.

    |
    i
    |
    ----j----

    Parameters
    ----------
    i_cen : int
        Row index of solar disk center.
    j_cen : int
        Column index of solar disk center.
    rad : int
        Solar disk radius in pixels.
    B0 : scalar
        B0 angle of solar equator.
    L0 : scalar
        Carringon longitude of central meridian.
    bins : (nx, ny)
        Resolution of synoptic map.
    deg : bool
        If True, all angles are in degrees. Default to True.

    Returns
    -------
    (image_ind, synoptic_ind) : tuple
        Solar disk indices and corresponding synoptic indices.
    """
    arr = np.vstack([np.hstack(map_disk_to_syn(i_cen, j_cen, rad, B0, L0, bins, deg)),
                     np.hstack(map_syn_to_disk(i_cen, j_cen, rad, B0, L0, bins, deg))])
    return pd.DataFrame(arr, columns=['disk_i', 'disk_j', 'syn_i', 'syn_j']).drop_duplicates()

def make_synoptic_map(img, i_cen, j_cen, rad, B0, L0, bins, average=None, deg=True):
    """Make a synoptic map from a solar disk image.

    |
    i
    |
    ----j----

    Parameters
    ----------
    img : 2d array
        Solar disk image.
    i_cen : int
        Row index of solar disk center.
    j_cen : int
        Column index of solar disk center.
    rad : int
        Solar disk radius in pixels.
    B0 : scalar
        B0 angle of solar equator.
    L0 : scalar
        Carringon longitude of central meridian.
    bins : (nx, ny)
        Resolution of synoptic map.
    average : callable
        Average function. Default is a mean function.
    deg : bool
        If True, all angles are in degrees. Default to True.

    Returns
    -------
    syn : 2d array of shape (ny, nx)
        Synoptic map.
    """
    df = disk_and_syn_mapping(i_cen, j_cen, rad, B0, L0, bins, deg)
    df = df.sort_values(['syn_i', 'syn_j'])
    pix = img[df['disk_i'], df['disk_j']]
    unq, counts = np.unique(df[['syn_i', 'syn_j']], axis=0, return_counts=True)
    arrs = np.split(pix, np.cumsum(counts))[:-1]
    if average is None:
        average = np.mean
    vals = np.array([average(arr) for arr in arrs])

    syn = np.full((bins[1], bins[0]), np.nan)
    syn[unq[:, 0], unq[:, 1]] = vals
    return syn

def label360(mask):
    """Label regions in a binary sinoptic map. Takes into account connections
    at 360 and 0 longitude."""
    if mask.dtype != np.bool:
        raise ValueError('Synoptic map should have bool dtype.')
    dfi = mask.shape[1]
    labeled = label(np.pad(mask, ((0, 0), (dfi, 0)), mode='wrap'), background=0)
    x = np.dstack((labeled[:, :dfi], labeled[:, dfi:]))
    unq = np.unique(x.reshape(-1, 2), axis=0)
    split = set(unq[:, 0]).intersection(unq[:, 1]) - {0}
    for p in split:
        p2 = unq[:, 1][unq[:, 0] == p]
        x[x == p2] = p
        p2 = unq[:, 0][unq[:, 1] == p]
        x[x == p2] = p
    return x.min(axis=-1)

def region_statistics(mask, sin, hmi=None):
    """Calculates statistics for regions in a binary synoptic map:
        - area ($10^{12}$ $km^2$)
        - mean latitude (in degrees, North at $90^{\circ}$)
        - largest Carrington longitude (in degrees)
        - positive magnetic flux ($10^{22}$ Mx)
        - negative magnetic flux ($10^{22}$ Mx)
        - maximal positive magnetic field value ($10^{22}$ Mx)
        - minimal negative magnetic field value ($10^{22}$ Mx)

    Parameters
    ----------
    mask : boolean ndarray
        Binary synoptic mask.
    sin : bool
        Sine Latitude synoptic map and magnetic synoptic map.
    hmi : ndarray
        Magnetic synoptic map for flux calculation. Should be same shape as mask.

    Returns
    -------
    df : pandas.DataFrame
        Regions statistics.
    """
    if sin:
        x = np.linspace(-1, 1, mask.shape[0] + 1)
        x = (x[:-1] + x[1:]) / 2.
        factor = np.ones(mask.shape[0])
    else:
        q = np.linspace(0, np.pi, mask.shape[0] + 1)
        q = (q[:-1] + q[1:]) / 2.
        factor = np.sin(q)
    factor = factor.reshape(-1, 1)
    labeled = label360(mask)
    ngroups = list(set(np.unique(labeled)) - {0})
    pixel_areas = np.ones(mask.shape, dtype=float) * factor
    pixel_areas = SOLAR_SURFACE_AREA * pixel_areas / pixel_areas.sum()
    if hmi is not None:
        pos_field = np.clip(hmi, 0, None)
        neg_field = -np.clip(-hmi, 0, None) #pylint: disable=invalid-unary-operand-type
        pos_flux = pos_field * pixel_areas
        neg_flux = neg_field * pixel_areas #pylint: disable=invalid-unary-operand-type
    if sin:
        lats = np.ones(mask.shape) * np.rad2deg(np.arcsin(-x)).reshape(-1, 1)
    else:
        lats = 90 - np.ones(mask.shape) * np.rad2deg(q).reshape(-1, 1)
    res = []
    for group_id in ngroups:
        report = {}
        reg_mask = labeled == group_id
        if hmi is not None:
            report['Flux_p'] = pos_flux[reg_mask].sum()
            report['Flux_n'] = neg_flux[reg_mask].sum()
            report['mx'] = pos_field[reg_mask].max()
            report['mn'] = neg_field[reg_mask].min()
        report['Area'] = pixel_areas[reg_mask].sum()
        if sin:
            report['Latitude'] = lats[reg_mask].mean()
        else:
            weights = pixel_areas[reg_mask]
            report['Latitude'] = np.average(lats[reg_mask], weights=weights/weights.sum())
        report['Longitude'] = max(np.where(reg_mask)[1]) * 360 / mask.shape[1]
        res.append(report)
    if res:
        return pd.DataFrame(res)
    if hmi is not None:
        return pd.DataFrame(columns=['Area', 'Latitude', 'Longitude', 'Flux_p', 'Flux_n', 'mx', 'mn'])
    return pd.DataFrame(columns=['Area', 'Latitude', 'Longitude'])
