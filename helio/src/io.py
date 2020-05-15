"""IO utils."""
import numpy as np
import pandas as pd
from astropy.io import fits
import sunpy
from skimage.measure import label

from .utils import detect_edges


def load_fits(path, verify='fix', smap=False):
    """Read fits file as `sunpy.map.Map` object or `ndarray`."""
    if smap:
        return sunpy.map.Map(path)
    hdul = fits.open(path)
    hdul.verify(verify)
    return hdul[0].data

def load_abp_mask(path, shape, sunspot_observation=False):
    """Builds segmentation mask from `abp` file.

    Parameters
    ----------
    path : str
        Path to abp file.
    shape : tuple
        Shape of the source image.
    sunspot_observation : bool
        If False, all active regions will be put in a single mask.
        If True, active regions will be separated into sunspots, cores and
        pores and put into separate masks.

    Returns
    -------
    mask : ndarray
        Segmentation mask.
    """
    with open(path, 'r') as fin:
        fread = fin.readlines()
        n_lines = len(fread)
        n_skip = int(fread[1].split()[0])
        num_objects = (n_lines - 3 - n_skip) // 2
        obj_meta = np.array([fread[3 + n_skip + 2 * i].split() for i in range(num_objects)]).astype(int)
        data = [np.array(fread[4 + n_skip +  2 * i].split()).astype(int) for i in range(num_objects)]

        df = pd.DataFrame(columns=['obj_num', 'core_num', 'pts'])
        if num_objects:
            df['obj_num'] = obj_meta[:, -2]
            df['core_num'] = obj_meta[:, -1]
            df['pts'] = [arr.reshape((-1, 3))[:, [1, 0]].astype('int') for arr in data]

    if sunspot_observation:
        mask = np.zeros(shape + (3,), dtype='bool')
        if df.empty:
            return mask
        #spots
        tdf = df.groupby('obj_num').filter(lambda x: len(x) > 1)
        pts = tdf.loc[tdf['core_num'] == 0, 'pts']
        if not pts.empty:
            pts = np.hstack(pts)
            mask[pts[:, 0], pts[:, 1], 0] = True
        #cores
        tdf = df.groupby('obj_num').filter(lambda x: len(x) > 1)
        pts = tdf.loc[tdf['core_num'] > 0, 'pts']
        if not pts.empty:
            pts = np.hstack(pts)
            mask[pts[:, 0], pts[:, 1], 1] = True
        #pores
        pts = df.groupby('obj_num').filter(lambda x: len(x) == 1)['pts']
        if not pts.empty:
            pts = np.hstack(pts)
            mask[pts[:, 0], pts[:, 1], 2] = True
        return mask

    mask = np.zeros(shape, dtype='bool')
    if df.empty:
        return mask
    pts = np.hstack(df['pts'])
    mask[pts[:, 0], pts[:, 1]] = True
    return mask

def write_syn_abp_file(fname, binary_mask, neighbors=None):
    """Write synoptic map to `abp` file."""
    binary_mask = binary_mask.astype(int)
    shape = binary_mask.shape[:2]
    header = [shape[1] // 2, shape[0] // 2, np.min(shape) // 2, 0, 0, 0, 1,
              shape[1], shape[0], 0, shape[1], 0, shape[0], 'Syn_map']

    labeled, num = label(binary_mask, neighbors=neighbors, background=0, return_num=True)
    edges = detect_edges(binary_mask)

    labeled_with_edges = 2 * labeled + edges

    buff = []
    buff.append(' '.join(map(str, header)) + '\n')
    buff.append('0 0\n')
    buff.append(str(num) + '\n')

    for i in range(1, num + 1):
        obj = np.where(labeled == i)
        pxl = np.array(obj).T
        arr = np.hstack([pxl[:, 1].reshape(-1, 1),
                         pxl[:, 0].reshape(-1, 1),
                         2 - ((labeled_with_edges[obj] % 2)).reshape(-1, 1)]).ravel()

        buff.append(' '.join(map(str, [i, len(obj[0])] + [0] * 5)) + '\n')
        buff.append(' '.join(map(str, arr)) + '\n')

    with open(fname, 'w') as fout:
        for line in buff:
            fout.writelines(line)
