"""IO utils."""
import datetime
import numpy as np
import pandas as pd
from astropy.io import fits
import sunpy
from sunpy.coordinates.sun import B0
from skimage.measure import label

from .utils import detect_edges


def load_fits(path, verify='fix', unit=0, as_smap=False):
    """Read fits file as `sunpy.map.Map` object or `ndarray`."""
    if as_smap:
        return sunpy.map.Map(path)
    hdul = fits.open(path)
    hdul.verify(verify)
    return hdul[unit].data

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
            pts = np.vstack(pts)
            mask[pts[:, 0], pts[:, 1], 0] = True
        #cores
        tdf = df.groupby('obj_num').filter(lambda x: len(x) > 1)
        pts = tdf.loc[tdf['core_num'] > 0, 'pts']
        if not pts.empty:
            pts = np.vstack(pts)
            mask[pts[:, 0], pts[:, 1], 1] = True
        #pores
        pts = df.groupby('obj_num').filter(lambda x: len(x) == 1)['pts']
        if not pts.empty:
            pts = np.vstack(pts)
            mask[pts[:, 0], pts[:, 1], 2] = True
        return mask

    mask = np.zeros(shape, dtype='bool')
    if df.empty:
        return mask
    pts = np.vstack(df['pts'])
    mask[pts[:, 0], pts[:, 1]] = True
    return mask

def write_fits(fname, data, index, meta, kind=None, **kwargs):
    """Write data to FITS file."""
    if kind is None:
        return write_simple_fits(fname, data, index, meta, **kwargs)
    if kind == 'synoptic':
        return write_syn_fits(fname, data, index, meta, **kwargs)
    if kind == 'synoptic_mask':
        return write_syn_mask_fits(fname, data, index, meta, **kwargs)
    if kind == 'aia_mask':
        return write_aia_mask_fits(fname, data, index, meta, **kwargs)
    raise NotImplementedError('FITS files for data of type {} are not implemented.'.format(kind))

def write_syn_fits(fname, data, index, meta, **headers):
    """Write synoptic map to FITS file."""
    _ = meta
    hdr = fits.Header()
    index = index.reset_index()
    hdr['DATE'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    car_rot = np.unique(index.loc[0, 'CR'])
    assert len(car_rot) == 1, 'CR numbers are not unique'
    hdr['CAR_ROT'] = int(car_rot[0])
    a = index.loc[0, 'DateTime'][0]
    b = index.loc[0, 'DateTime'][-1]
    rot = a + (b - a) / 2
    hdr['DATE-OBS'] = rot.strftime("%Y-%m-%dT%H:%M:%S")
    hdr['T_OBS'] = rot.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    hdr['T_ROT'] = rot.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    hdr['T_START'] = a.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    hdr['T_STOP'] = b.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    hdr['B0_ROT'] = B0(rot).deg
    hdr['B0_FIRST'] = index.loc[0, 'B0'][0]
    hdr['B0_LAST'] = index.loc[0, 'B0'][-1]
    hdr['CRPIX1'] = (data.shape[1] + 1.) / 2
    hdr['CRPIX2'] = (data.shape[0] + 1.) / 2
    hdr['CRVAL1'] = 180.
    hdr['CRVAL2'] = 0.
    hdr['CDELT1'] = 360. / data.shape[1]
    hdr['CDELT2'] = 180. / data.shape[0]
    hdr['CUNIT1'] = 'deg'
    hdr['CUNIT2'] = 'deg'
    hdr['CTYPE1'] = 'Carrington Longitude'
    hdr['CTYPE2'] = 'Latitude'
    hdr['BSCALE'] = 1.
    hdr['BZERO'] = 0.
    hdr['BUNIT'] = 'counts / pixel'
    hdr['WCSNAME'] = 'Carrington Heliographic'
    hdr['MISSVALS'] = np.isnan(data).sum()
    for k, v in headers.items():
        hdr[k] = v
    hdu = fits.PrimaryHDU(header=hdr, data=data[::-1])
    hdul = fits.HDUList([hdu])
    hdul.writeto(fname, overwrite=True)

def write_syn_mask_fits(fname, data, index, meta, **headers):
    """Write mask for synoptic map to FITS file."""
    _ = index
    hdr = fits.Header()
    hdr['DATE'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    for k in ['CAR_ROT', 'DATE-OBS', 'T_OBS', 'T_ROT', 'T_START', 'T_STOP',
              'B0_ROT', 'B0_FIRST', 'B0_LAST']:
        hdr[k] = meta[k]
    hdr['CRPIX1'] = (data.shape[1] + 1.) / 2
    hdr['CRPIX2'] = (data.shape[0] + 1.) / 2
    hdr['CRVAL1'] = 180.
    hdr['CRVAL2'] = 0.
    hdr['CDELT1'] = 360. / data.shape[1]
    hdr['CDELT2'] = 180. / data.shape[0]
    hdr['CUNIT1'] = 'deg'
    hdr['CUNIT2'] = 'deg'
    hdr['CTYPE1'] = 'Carrington Longitude'
    hdr['CTYPE2'] = 'Latitude'
    hdr['BSCALE'] = 1.
    hdr['BZERO'] = 0.
    hdr['BUNIT'] = 'counts / pixel'
    hdr['WCSNAME'] = 'Carrington Heliographic'
    hdr['MISSVALS'] = np.isnan(data).sum()
    for k, v in headers.items():
        hdr[k] = v
    hdu = fits.PrimaryHDU(header=hdr, data=data[::-1])
    hdul = fits.HDUList([hdu])
    hdul.writeto(fname, overwrite=True)

def write_aia_mask_fits(fname, data, index, meta, **headers):
    """Write mask for AIA disk image to FITS file."""
    hdr = fits.Header()
    hdr['DATE'] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    index = index.reset_index()
    hdr['CAR_ROT'] = int(index.loc[0, 'CR'])
    hdr['DATE-OBS'] = index.loc[0, 'DateTime'].strftime("%Y-%m-%dT%H:%M:%S")
    hdr['T_OBS'] = index.loc[0, 'DateTime'].strftime("%Y.%m.%d_%H:%M:%S_TAI")
    hdr['B0_OBS'] = index.loc[0, 'B0']
    hdr['L0_OBS'] = index.loc[0, 'L0']
    hdr['R_SUN'] = meta['r']
    hdr['X0_MP'] = meta['j_cen']
    hdr['Y0_MP'] = meta['i_cen']
    hdr['CRPIX1'] = (data.shape[1] + 1.) / 2
    hdr['CRPIX2'] = (data.shape[0] + 1.) / 2
    hdr['CRVAL1'] = 0.
    hdr['CRVAL2'] = 0.
    hdr['CDELT1'] = 0.6 * 4096 / data.shape[1]
    hdr['CDELT2'] = 0.6 * 4096 / data.shape[0]
    hdr['CUNIT1'] = 'arcsec'
    hdr['CUNIT2'] = 'arcsec'
    hdr['CTYPE1'] = 'HPLN-TAN'
    hdr['CTYPE2'] = 'HPLN-TAN'
    hdr['BSCALE'] = 1.
    hdr['BZERO'] = 0.
    hdr['BUNIT'] = 'counts / pixel'
    hdr['MISSVALS'] = np.isnan(data).sum()
    for k, v in headers.items():
        hdr[k] = v
    hdu = fits.PrimaryHDU(header=hdr, data=data[::-1])
    hdul = fits.HDUList([hdu])
    hdul.writeto(fname, overwrite=True)

def write_simple_fits(fname, data, index, meta, **headers):
    """Write array to FITS file."""
    _ = index, meta
    hdr = fits.Header()
    for k, v in headers.items():
        hdr[k] = v
    hdu = fits.PrimaryHDU(header=hdr, data=data)
    hdul = fits.HDUList([hdu])
    hdul.writeto(fname, overwrite=True)

def write_abp_file(fname, binary_mask, neighbors=None, meta=None):
    """Write binary map to `abp` file."""
    binary_mask = binary_mask.astype(int)
    if 'r' in meta:
        header = [meta['j_cen'], meta['i_cen'], meta['r'], 0, 0, 0, 0]
    else:
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
