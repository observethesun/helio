"""IO utils."""
import os
import warnings
from string import Template
import urllib.request
import datetime
import json
import numpy as np
import pandas as pd
from astropy.io import fits
import sunpy
from sunpy.coordinates.sun import B0
from skimage.measure import label

from .polygon import SphericalPolygon, PlanePolygon
from .utils import detect_edges
from .templates import CH_XML


def load_fits(path, verify='fix', unit=0, as_smap=False):
    """Read fits file as `sunpy.map.Map` object or `ndarray`."""
    if as_smap:
        return sunpy.map.Map(path)
    hdul = fits.open(path)
    hdul.verify(verify)
    return hdul[unit].data

def load_abp_mask(path, shape=None, spot_layers=False, group_layer=False):
    """Builds mask from `abp` file.

    Parameters
    ----------
    path : str
        Path to abp file.
    shape : tuple, optional
        Shape of the source image.
    spot_layers : bool
        If False, all active regions will be put in a single mask.
        If True, active regions will be separated into sunspots, cores and
        pores and put into separate masks. Default False.
    group_layer : bool
        If True, create a mask with group numbers. Default False.

    Returns
    -------
    mask : ndarray
        Segmentation mask.
    """
    with open(path, 'r') if os.path.exists(path) else urllib.request.urlopen(path) as fin:
        fread = fin.readlines()
        n_lines = len(fread)
        if shape is None:
            header = np.array(fread[0].split())
            i_cen = int(header[1])
            j_cen = int(header[0])
            r = int(header[2])
            shape = (i_cen + r + 1, j_cen + r + 1)
        n_skip = int(fread[1].split()[0])
        num_objects = (n_lines - 3 - n_skip) // 2
        obj_meta = np.array([fread[3 + n_skip + 2 * i].split() for i in range(num_objects)]).astype(int)
        data = [np.array(fread[4 + n_skip + 2 * i].split()).astype(int) for i in range(num_objects)]

        df = pd.DataFrame(columns=['obj_num', 'core_num', 'pts'])
        if num_objects:
            df['group_num'] = obj_meta[:, 2]
            df['obj_num'] = obj_meta[:, -2]
            df['core_num'] = obj_meta[:, -1]
            df['pts'] = [arr.reshape((-1, 3))[:, [1, 0]].astype(int) for arr in data]

    if group_layer:
        group_mask = np.zeros(shape, dtype='int')
        for _, row in df.iterrows():
            pts = row['pts']
            group_mask[pts[:, 0], pts[:, 1]] = row['group_num'] #assume positive group numbers
        if spot_layers is False:
            return group_mask

    if spot_layers:
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
        if group_layer:
            return np.concatenate([mask, group_mask[..., np.newaxis]], axis=-1)
        return mask

    mask = np.zeros(shape, dtype='bool')
    if df.empty:
        return mask
    pts = np.vstack(df['pts'])
    mask[pts[:, 0], pts[:, 1]] = True
    return mask

def load_cnt(path, shape=None):
    """Builds segmentation mask from `abp` file.

    Parameters
    ----------
    path : str
        Path to abp file.
    shape : tuple, optional
        Shape of the source image.

    Returns
    -------
    mask : ndarray
        Segmentation mask.
    """
    with open(path, 'r') if os.path.exists(path) else urllib.request.urlopen(path) as fin:
        fread = fin.readlines()
        n_lines = len(fread)
        if shape is None:
            header = np.array(fread[0].split())
            i_cen = int(header[1])
            j_cen = int(header[0])
            r = int(header[2])
            shape = (i_cen + r + 1, j_cen + r + 1)
        num_objects = (n_lines - 2) // 2
        data = [np.array(fread[3 + 2 * i].split()).astype(float).reshape(-1, 2) for i in range(num_objects)]
        data = [SphericalPolygon(arr, deg=True) for arr in data]
    return data

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

def write_abp_file(fname, binary_mask, connectivity=None, meta=None):
    """Write binary map to `abp` file."""
    binary_mask = binary_mask.astype(int)
    if 'r' in meta:
        header = [meta['j_cen'], meta['i_cen'], meta['r'], 0, 0, 0, 0]
    else:
        shape = binary_mask.shape[:2]
        header = [shape[1] // 2, shape[0] // 2, np.min(shape) // 2, 0, 0, 0, 1,
                  shape[1], shape[0], 0, shape[1], 0, shape[0], 'Syn_map']

    labeled, num = label(binary_mask, connectivity=connectivity, background=0, return_num=True)
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

def _make_chs_xml(ivorn, props, index, stop_datetime=None, **kwargs):
    """Fill CHs props into template XML."""
    tmp = Template(CH_XML)
    fill_values = {}
    fill_values['ivorn'] = ivorn
    fill_values['datetime_now'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    fill_values['datetime'] = index.loc[0, 'DateTime'].strftime('%Y-%m-%dT%H:%M:%S')
    fill_values['start_datetime'] = fill_values['datetime']
    fill_values['stop_datetime'] = (fill_values['start_datetime'] if stop_datetime is None
                                    else stop_datetime.strftime('%Y-%m-%dT%H:%M:%S'))
    fill_values['area'] = '%.3f' % props.area_msh
    fill_values['area_unit'] = 'microhemisphere'
    fill_values['lat_mean'] = '%.3f' % props.centroid_hpc[0]
    fill_values['long_mean'] = '%.3f' % props.centroid_hpc[1]
    fill_values['bbox_lat'] = '%.3f' % props.bbox_hpc[0]
    fill_values['bbox_long'] = '%.3f' % props.bbox_hpc[1]
    fill_values['bbox_lat_size'] = '%.3f' % (props.bbox_hpc[2] - props.bbox_hpc[0])
    fill_values['bbox_long_size'] = '%.3f' % (props.bbox_hpc[3] - props.bbox_hpc[1])
    fill_values['chaincode_size'] = len(props.approx_contour_hpc)
    fill_values['chaincode_lat_start'] = '%.3f' % props.approx_contour_hpc[0, 0]
    fill_values['chaincode_long_start'] = '%.3f' % props.approx_contour_hpc[0, 1]
    fill_values['chaincode'] = ','.join(['%.3f' % x for x in props.approx_contour_hpc.ravel()])
    tmp = Template(tmp.safe_substitute(fill_values))
    tmp = Template(tmp.safe_substitute(kwargs))
    out = tmp.safe_substitute()
    return out

def write_xml(path, data, index, kind, basename, **kwargs):
    """Write HEK xml file."""
    index = index.reset_index()
    ivorn_base = basename + '_' + index.loc[0, 'DateTime'].strftime('%Y%m%d_%H%M%S')
    if kind.lower() == 'chs':
        make_xml = _make_chs_xml
    else:
        raise NotImplementedError(kind)
    for i, props in enumerate(data):
        ivorn = ivorn_base + '_' + str(i)
        xml = make_xml(ivorn, props, index, **kwargs)
        missing = Template.pattern.findall(xml)
        if missing:
            warnings.warn('Missed values for ' + ', '.join([k[1] for k in missing]))
        with open(os.path.join(path, ivorn + '.xml'), 'w') as f:
            f.writelines(xml)

def write_json(path, data, index, decimals=2):
    """Write polygons to json file."""
    def round_floats(x):
        if isinstance(x, float):
            return np.round(x, decimals=decimals)
        if isinstance(x, dict):
            return {k: round_floats(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [round_floats(i) for i in x]
        return x
    out = {'data': []}
    for poly in data:
        out['data'].append(round_floats(poly.summary))
    with open(os.path.join(path, index.iloc[0].name + '.json'), 'w') as f:
        json.dump(out, f)

def load_json(path):
    """Load polygons from a json file."""
    with open(path) if os.path.exists(path) else urllib.request.urlopen(path) as f:
        data = json.load(f)['data']
    polygons = []
    for arr in data:
        if arr['type'] == 'SphericalPolygon':
            poly = SphericalPolygon(arr['vertices'], deg=arr['deg'])
        elif arr['type'] == 'PlanePolygon':
            poly = PlanePolygon(arr['vertices'])
        else:
            raise NotImplementedError('Unknown polygon type {}'.format(arr['type']))
        polygons.append(poly)
    return polygons
 