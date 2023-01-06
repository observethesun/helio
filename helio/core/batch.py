#pylint: disable=too-many-lines
"""HelioBatch class."""
import os
import warnings
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import dill
from astropy.io import fits
from aiapy.calibrate import correct_degradation
from scipy import interpolate
from scipy import sparse
from scipy.ndimage.morphology import binary_fill_holes
from skimage.io import imread
import skimage
import skimage.transform
from skimage.measure import label, regionprops, find_contours, approximate_polygon
from skimage.transform import resize, rescale, hough_circle, hough_circle_peaks
from skimage.feature import canny
from sklearn.metrics.pairwise import haversine_distances
import drms

from .decorators import execute, add_actions, extract_actions, TEMPLATE_DOCSTRING
from .index import BaseIndex
from .synoptic_maps import make_synoptic_map, label360, region_statistics
from .geometry_utils import xy_to_xyz, xyz_to_sp, xy_to_carr, sp_to_xy, rotate_at_center
from .io import (load_fits, load_abp_mask, load_cnt, load_json,
                 write_abp_file, write_fits, write_xml, write_json)
from .polygon import SphericalPolygon, PlanePolygon
from .utils import detect_edges

def softmax(x):
    """Softmax function."""
    z = np.exp(x)
    return z / sum(z)

def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


@add_actions(extract_actions(skimage, 'image'), TEMPLATE_DOCSTRING) # pylint: disable=too-many-public-methods
@add_actions(extract_actions(skimage.transform, 'image'), TEMPLATE_DOCSTRING)
class HelioBatch():
    """Batch class for solar observations processing.

    Attributes
    ----------
    index : FilesIndex
        Unique identifiers of batch items."""
    def __init__(self, index):
        self._index = index
        self._data = {}
        self._meta = {}

    def __getattr__(self, attr):
        return self._data[attr]

    def __setattr__(self, attr, value):
        if attr.startswith('_') or attr in dir(self):
            return super().__setattr__(attr, value)
        if attr in self:
            raise KeyError('Attribute already exists.')
        assert len(value) == len(self), 'Length mismatch.'
        self._data[attr] = np.array(list(value) + [None])[:-1]
        self._meta[attr] = np.array([None] * len(self))
        return self

    def __contains__(self, x):
        return x in self._data

    @property
    def meta(self):
        """Meta information on observations."""
        return self._meta

    @property
    def data(self):
        """Observational data."""
        return self._data

    @property
    def attributes(self):
        """List of data keys."""
        return list(self._data.keys())

    @property
    def indices(self):
        """Batch items identifiers."""
        return self._index.indices

    @property
    def index(self):
        """Batch index."""
        return self._index

    def __len__(self):
        return len(self._index)

    @execute(how='threads')
    def apply(self, i, func, src, dst, **kwargs):
        """Apply a ``func`` to each item in ``src`` and write results in ``dst``.

        Parameters
        ----------
        src : str, tuple of str
            A source for data.
        dst : same type as src
            A destination for results.
        func : callable
            A function to apply.
        kwargs : misc
            Any additional named arguments to ``func``.

        Returns
        -------
        batch : HelioBatch
            Processed batch.
        """
        data = self.data[src][i]
        self.data[dst][i] = func(data, **kwargs)
        return self

    @execute(how='threads')
    def apply_meta(self, i, func, src, dst, **kwargs):
        """Apply a ``func`` to each item in meta.

        Parameters
        ----------
        src : str, tuple of str
            A source for meta.
        dst : same type as src
            A destination for results.
        func : callable
            A function to apply.
        kwargs : misc
            Any additional named arguments to ``func``.

        Returns
        -------
        batch : HelioBatch
            Processed batch.
        """
        meta = self.meta[src][i]
        self.meta[dst][i] = func(meta, **kwargs)
        return self

    def load(self, src, dtype=None, meta=None, **kwargs):
        """Load batch data from source.

        Parameters
        ----------
        src : str, tuple of str
            Index column labels with data sources.
        dtype : dtype
            The type of output arrays.
        meta : str
            Index column label with data meta information.
        kwargs : misc
            Any additional named arguments to data and meta loaders.

        Returns
        -------
        batch : HelioBatch
            Batch with loaded data.
        """
        self._load_data(src=src, dtype=dtype, **kwargs)
        if meta is None:
            return self
        src = np.atleast_1d(src)
        self._load_meta(src=[meta]*len(src), dst=src, **kwargs)
        return self

    @execute(how='threads')
    def _load_data(self, i, src, dst, dtype=None, **kwargs):
        """Load arrays with observational data from various formats."""
        path = self.index.iloc[i, self.index.columns.get_loc(src)]
        fmt = Path(path).suffix.lower()[1:]
        if fmt == 'npz':
            if kwargs.get('sparse', False):
                data = sparse.load_npz(path)
            else:
                f = np.load(path)
                keys = list(f.keys())
                data = f[src] if len(keys) != 1 else f[keys[0]]
        elif fmt == 'abp':
            if isinstance(kwargs.get('shape', None), str):
                kwargs['shape'] = self.data[kwargs['shape']][i].shape
            data = load_abp_mask(path, **kwargs)
        elif fmt == 'cnt':
            data = load_cnt(path, **kwargs)
        elif fmt == 'json':
            data = load_json(path, **kwargs)
        elif fmt in ['fts', 'fits']:
            data = load_fits(path, **kwargs)
        else:
            data = imread(path, **kwargs)

        if dtype:
            data = data.astype(dtype)

        self.data[dst][i] = data
        return self

    @execute(how='threads')
    def _load_meta(self, i, src, dst, verify='fix', unit=0, **kwargs):
        """Load additional meta information on observations.
        """
        _ = kwargs
        path = self.index.iloc[i, self.index.columns.get_loc(src)]
        fmt = Path(path).suffix.lower()[1:]
        if fmt in ['abp', 'cnt']:
            with open(path, 'r') if os.path.exists(path) else urllib.request.urlopen(path) as fin:
                fread = fin.readlines()
                header = np.array(fread[0].split())
                #i_cen enumerates rows, j_cen enumerates columns
                meta = dict(i_cen=int(header[1]),
                            j_cen=int(header[0]),
                            r=int(header[2]),
                            P=float(header[3]),
                            B0=float(header[4]),
                            L0=float(header[5]))
        elif fmt in ['fts', 'fits']:
            hdul = fits.open(path)
            hdul.verify(verify)
            hdr = hdul[unit].header
            meta = dict(hdr.items())
            #i_cen enumerates rows, j_cen enumerates columns
            if 'X0_MP' in meta:
                meta['j_cen'] = int(meta['X0_MP'])
            if 'Y0_MP' in meta:
                meta['i_cen'] = int(meta['Y0_MP'])
            if 'R_SUN' in meta:
                meta['r'] = int(meta['R_SUN'])
        else:
            raise NotImplementedError('Format {} is not supported.'.format(fmt))

        self.meta[dst][i] = meta
        return self

    def jsoc_load(self, path, series, email, method='url', protocol='fits', verbose=True):
        """Download data from JSOC based on DateTime column in the batch index.

        Parameters
        ----------
        path : str
            Directory to save files.
        series : str
            JSOC series name, e.g. hmi.M_720s.
        email : str
            User email.
        method : str
            `drms` export method. Default fits.
        protocol: str
            `drms` export protocol. Default url.
        verbose : bool
            `drms` export and download verbose option. Default True.

        Returns
        -------
        batch : HelioBatch
            Batch unchanged.
        """
        client = drms.Client(email=email, verbose=verbose)
        arr = []
        for i in range(len(self)):
            dt = self.index.iloc[i]['DateTime']
            arr.append('{}[{}_TAI]'.format(series, dt.strftime('%Y.%m.%d_%H:%M')))
        query = ','.join(arr)
        k = client.export(query, method=method, protocol=protocol)
        k.wait()
        if k.status:
            return self
        return self._jsoc_load(client=k, path=path, verbose=verbose)

    @execute(how='threads')
    def _jsoc_load(self, i, src, dst, client, path, verbose=True):
        """Download data from JSOC."""
        _ = src, dst
        if i < len(client.data):
            client.download(path, i, verbose=verbose)
        return self

    def url_load(self, src, path, rename=False, fmt='%Y-%m-%dT%H%M%S'):
        """Download data from URL address and save to local directory.

        Parameters
        ----------
        src : str
            Index colum label with data sources.
        path : str
            Directory to save files. Directory should exist.
        rename : bool
            Leave file names unchanged or make new names based on datetime and the source name.
            Default False.
        fmt : str, optional
            Datetime format of the renamed files. Used if rename=True.

        Returns
        -------
        batch : HelioBatch
            Batch unchanged.
        """
        return self._url_load(src=src, dst=path, rename=rename, fmt=fmt)

    @execute(how='threads')
    def _url_load(self, i, src, dst, rename, fmt):
        """Load data from URL."""
        url = self.index.iloc[i][src]
        if rename:
            file = (src +
                    '_' +
                    self.index.iloc[i]['DateTime'].strftime(fmt) +
                    Path(url).suffix)
        else:
            file = Path(url).name
        urllib.request.urlretrieve(url, Path(dst, file))
        return self

    def deepcopy(self):
        """Make a deep copy of batch."""
        batch_dump = dill.dumps(self)
        return dill.loads(batch_dump)

    def dump(self, src, path, format, **kwargs): #pylint: disable=redefined-builtin
        """Dump data in various formats.

        Supported formats: npz, txt, binary, fits, abp, blosc and any of
        `matplotlib.pyplot.imsave` supported formats.

        Parameters
        ----------
        src : str
            A source for data to save.
        path : str
            Path where to write output files.
        format : str
            Output file format.
        kwargs : misc
            Any additional named arguments to dump method.

        Returns
        -------
        batch : HelioBatch
            Batch unchanged.
        """
        if len(np.atleast_1d(src)) > 1:
            raise ValueError('Only single attribute allowed.')
        if 'scatter' in kwargs:
            return self._dump_scatter_image(src=src, path=path, format=format, **kwargs)
        return self._dump(src=src, path=path, format=format, **kwargs)

    @execute(how='loop')
    def _dump_scatter_image(self, i, src, dst, scatter, path, format='jpg', #pylint: disable=redefined-builtin
                            dpi=None, figsize=None, cmap=None, **scatter_kw):
        """Dump image with scatterplot."""
        _ = dst
        img = self.data[src][i]
        pts = self.data[scatter][i]
        fname = os.path.join(path, str(self.indices[i])) + '.' + format
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img, cmap=cmap, aspect='auto')
        ax.scatter(pts[1], pts[0], **scatter_kw)
        ax.axis('off')
        ax.set_xlim([0, img.shape[1]])
        ax.set_ylim([img.shape[0], 0])
        plt.savefig(fname, dpi=dpi)
        plt.close(fig)
        return self

    @execute(how='threads')
    def _dump(self, i, src, dst, path, format, meta=None, **kwargs): #pylint: disable=redefined-builtin
        """Dump data in various formats."""
        _ = dst
        fname = os.path.join(path, str(self.indices[i])) + '.' + format
        data = self.data[src][i]
        meta = self.meta[src if meta is None else meta][i]
        if format == 'npz':
            np.savez(fname, data, **kwargs)
        elif format == 'txt':
            np.savetxt(fname, data, **kwargs)
        elif format == 'binary':
            data.tofile(fname)
        elif format == 'abp':
            write_abp_file(fname, data, meta=meta)
        elif format == 'fits':
            write_fits(fname, data, index=self.index.iloc[[i]], meta=meta, **kwargs)
        elif format == 'xml':
            write_xml(path, data, index=self.index.iloc[[i]], meta=meta, **kwargs)
        elif format == 'json':
            write_json(path, data, index=self.index.iloc[[i]], **kwargs)
        else:
            plt.imsave(fname, data, format=format, **kwargs)
        return self

    @execute(how='threads')
    def dump_group_patches(self, i, src, dst, meta=None, min_area=0):
        """Dump group pathes into separate `.npz` files."""
        ind = self.indices[i]
        data = self.data[src][i]
        meta = self.meta[src if meta is None else meta][i]
        labels = np.unique(data)
        for n in labels:
            if n == 0:
                continue
            group = np.where(data == n)
            imin = group[0].min()
            imax = group[0].max()
            jmin = group[1].min()
            jmax = group[1].max()
            patch = data[imin:imax, jmin:jmax]
            if patch.sum() < min_area:
                continue
            path = os.path.join(dst, ind + '_' + str(n) + '.npz')
            np.savez(path, patch=patch, r=meta['r'])
        return self

    @execute(how='threads')
    def fillna(self, i, src, dst, value=0.):
        """Replace NaN with a given value.

        Parameters
        ----------
        src : str
            A source for array.
        dst : str
            A destination for results.
        value : scalar
            Value to be used to fill NaN values. Default to 0.

        Returns
        -------
        batch : HelioBatch
            Batch NaN replaced.
        """
        data = self.data[src][i]
        if src != dst:
            data = data.copy()
        data[np.isnan(data)] = value
        self.data[dst][i] = data

    @execute(how='threads')
    def disk_center_crop(self, i, src, dst):
        """Crop disk to radius.

        Parameters
        ----------
        src : str
            A source for data.
        dst : str
            A destination for results.

        Returns
        -------
        batch : HelioBatch
            Batch cropped data.
        """
        data = self.data[src][i]
        i_cen = self.meta[src][i]['i_cen']
        j_cen = self.meta[src][i]['j_cen']
        r = self.meta[src][i]['r']
        self.data[dst][i] = data[i_cen-r:i_cen+r+1, j_cen-r:j_cen+r+1]
        self.meta[dst][i]['i_cen'] = r
        self.meta[dst][i]['j_cen'] = r
        self.meta[dst][i]['r'] = r
        return self

    @execute(how='loop')
    def correct_degradation(self, i, src, dst, **kwargs):
        """Apply ``aiapy.calibrate.correct_degradation`` procedure to AIA data.

        Parameters
        ----------
        src : str
            A source for AIA map.
        dst : str
            A destination for results.
        kwargs : misc
            Any additional named arguments to dump method.

        Returns
        -------
        batch : HelioBatch
            Batch with corrected AIA maps.
        """
        smap = self.data[src][i]
        self.data[dst][i] = correct_degradation(smap, **kwargs)
        return self

    @execute(how='threads')
    def rotate_p_angle(self, i, src, dst, deg=True, labels=False, background=0, **kwargs):
        """Rotate disk image to P=0 around disk center.

        Parameters
        ----------
        src : str
            A source for disk images.
        dst : str
            A destination for results.
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
        batch : HelioBatch
            Batch with rotated disk.
        """
        data = self.data[src][i]
        meta = self.meta[src][i]
        data = rotate_at_center(data, angle=meta["P"], center=(meta['j_cen'], meta['i_cen']),
                                deg=deg, labels=labels, background=background, **kwargs)
        self.data[dst][i] = data
        self.meta[dst][i] = meta.copy()
        self.meta[dst][i]['P'] = 0
        return self

    @execute(how='threads')
    def mask_disk(self, i, src, dst, fill_value=np.nan):
        """Set dummy value to pixels outside the solar disk.

        Parameters
        ----------
        src : str
            A source for disk images.
        dst : str
            A destination for results.
        fill_value : scalar
            A value to be set to pixels outside the solar disk. Default to np.nan.

        Returns
        -------
        batch : HelioBatch
            Batch with masked disk.
        """
        img = self.data[src][i]
        if src != dst:
            img = img.copy()
        meta = self.meta[src][i]
        ind = np.transpose(np.indices(img.shape[:2]), axes=(1, 2, 0))
        outer = np.where(np.linalg.norm(ind - np.array([meta['i_cen'], meta['j_cen']]), axis=-1) > meta['r'])
        img[outer] = fill_value
        self.data[dst][i] = img
        return self

    def drop_empty_days(self, mask):
        """Drop batch items without active regions.

        Parameters
        ----------
        mask : str
            A source for active regions masks.

        Returns
        -------
        batch : HelioBatch
            Batch without empty observations.
        """
        valid = [np.any(x) for x in self.data[mask]]
        self.index = self.index.loc[valid]
        for k, v in self.data.items():
            self.data[k] = v[valid]
        for k, v in self.meta.items():
            self.meta[k] = v[valid]
        return self

    @execute(how='threads')
    def get_radius(self, i, src, dst, hough_radii, sigma=2, raise_limits=False, logger=None):
        """Estimate solar disk center and radius.

        Parameters
        ----------
        src : str
            A source for solar disk images.
        hough_radii : tuple
            Mininal and maximal radius to search.
        sigma : scalar, optional
            Canny filter parameter. Default 2.
        raise_limits : bool, optional
            Raise error if radius found is in the end of search interval. Default False.
        logger : logger, optional
            Logger for messages. Default None.

        Returns
        -------
        batch : HelioBatch
            Batch with updated meta.
        """
        img = self.data[src][i]
        meta = self.meta[src][i]

        edges = canny(img, sigma=sigma)
        hough_res = hough_circle(edges, hough_radii)
        _, c_x, c_y, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
        if radii in [hough_radii[0], hough_radii[-1]]:
            logger = warnings if logger is None else logger
            msg = 'At index {}. Estimated radius {} is in the end of hough_radii.\
            Try to extend hough_radii to verify radius found.'.format(self.index.indices[i], radii)
            if raise_limits:
                raise ValueError(msg)
            logger.warn(msg)

        meta['i_cen'] = int(c_y)
        meta['j_cen'] = int(c_x)
        meta['r'] = int(radii)
        self.meta[dst][i] = meta
        return self

    @execute(how='threads')
    def disk_resize(self, i, src, dst, output_shape, **kwargs):
        """Resize solar disk image and update center location and radius.

        Parameters
        ----------
        src : str
            A source for solar disk images.
        output_shape : tuple
            Shape of output images. Axes ratio should be the same as for source image.
        kwargs : misc
            Any additional named arguments to ``skimage.transform.resize`` method.

        Returns
        -------
        batch : HelioBatch
            Batch with resized images and adjusted meta.
        """
        img = self.data[src][i]

        input_shape = img.shape[:2]
        assert input_shape[0] / output_shape[0] == input_shape[1] / output_shape[1]

        img = resize(img, output_shape=output_shape, **kwargs)
        self.data[dst][i] = img

        ratio = input_shape[0] / output_shape[0]
        meta = self.meta[src][i]
        if src != dst:
            meta = meta.copy()
        meta['i_cen'] = int(meta['i_cen'] / ratio)
        meta['j_cen'] = int(meta['j_cen'] / ratio)
        meta['r'] = int(meta['r'] / ratio)
        if src != dst:
            self.meta[dst][i] = meta
        return self

    @execute(how='threads')
    def fit_mask(self, i, src, dst, target):
        """Fit mask to new disk center and radius.

        Parameters
        ----------
        src : str
            A source for mask.
        dst : str
            A destination for new mask.
        target : str
            A disk image to fit to.

        Returns
        -------
        batch : HelioBatch
            Batch with new masks.
        """
        img = self.data[target][i]
        i_cen, j_cen = self.meta[target][i]['i_cen'], self.meta[target][i]['j_cen']
        rad = self.meta[target][i]['r']

        mask = self.data[src][i]
        has_channels = mask.ndim == 3
        mask = np.atleast_3d(mask)
        i_cen_mask, j_cen_mask = self.meta[src][i]['i_cen'], self.meta[src][i]['j_cen']
        rad_mask = self.meta[src][i]['r']

        scale = rad / rad_mask
        rmask = rescale(mask, (scale, scale), preserve_range=True, multichannel=True)
        new_mask = np.full(img.shape[:2] + mask.shape[2:], False)
        iarr, jarr, zarr = np.where(rmask > 0.5)
        new_mask[iarr - int(i_cen_mask * scale) + i_cen,
                 jarr - int(j_cen_mask * scale) + j_cen,
                 zarr] = True

        self.data[dst][i] = new_mask if has_channels else new_mask[..., 0]
        self.meta[dst][i] = {'i_cen': i_cen, 'j_cen': j_cen, 'r': rad}
        return self

    @execute(how='threads')
    def fit_scale(self, i, src, dst, target, labels=False, background=0):
        """Fit data scale to the target disk radius.

        Parameters
        ----------
        src : str
            A source for data.
        dst : str
            A destination for new image.
        target : str or int
            Target radius. If str, then get radius from the target meta.
        labels : bool
            Data contains labels. Default False.
        background : scalar
            Background label. Default 0.

        Returns
        -------
        batch : HelioBatch
            Batch with rescaled images.
        """
        data = self.data[src][i]
        rad = self.meta[src][i]['r']
        i_cen = self.meta[src][i]['i_cen']
        j_cen = self.meta[src][i]['j_cen']

        has_channels = data.ndim == 3
        data = np.atleast_3d(data)
        is_bool = data.dtype == np.bool

        rad_target = self.meta[target][i]['r'] if isinstance(target, str) else target
        scale = rad_target / rad
        if labels:
            new_data = None
            lbs = np.unique(data)
            for k in lbs:
                if (k == background) and (len(lbs) > 1):
                    continue
                mask = data == k
                rmask = rescale(mask, (scale, scale), preserve_range=True, multichannel=True) > 0.5
                if new_data is None:
                    new_data = np.full(rmask.shape, background)
                new_data[rmask] = k
        else:
            new_data = rescale(data, (scale, scale), preserve_range=True, multichannel=True)
        if not has_channels:
            new_data = new_data[..., 0]
        if is_bool:
            new_data = new_data > 0.5

        self.data[dst][i] = new_data
        self.meta[dst][i].update({'i_cen': int(i_cen * scale + 0.5),
                                  'j_cen': int(j_cen * scale + 0.5),
                                  'r': int(rad * scale + 0.5)})
        return self

    @execute(how='threads')
    def get_region_props(self, i, src, dst, level=0.5, tolerance=3, cache=True):
        """Get region properties.

        Parameters
        ----------
        src : str
            A source for binary mask.
        dst : str
            A destination for props.
        level : float
            Value along which to find contours in the array. Default 0.5.
        tolerance : int, optional
            A tolerance for contour approximation. Default 3.
        cache : bool, optional
            Use cache in regionprops.

        Returns
        -------
        batch : HelioBatch
            Batch with props calculated.
        """
        mask = binary_fill_holes(self.data[src][i])
        labeled, num = label(mask, background=0, return_num=True)
        if not num:
            self.data[dst][i] = []
            return self
        props = regionprops(labeled, cache=cache)
        for prop in props:
            cnt = find_contours(labeled == prop.label, level=level, fully_connected='high')[0]
            prop.contour = cnt
            prop.approx_contour = approximate_polygon(cnt, tolerance=tolerance)
        self.data[dst][i] = props
        self.meta[dst][i] = self.meta[src][i].copy()
        return self

    def props2hgc(self, src, meta):
        """Map props in i,j coordinates to HGC (Heliographic-Carrington) coordinates.

        Parameters
        ----------
        src : str
            A source for props.
        meta : str
            A source for disk meta information.

        Returns
        -------
        batch : HelioBatch
            Batch with props calculated.
        """
        return self._props_mapping(src=src, meta=meta, name='HGC')

    def props2hpc(self, src, meta, resolution):
        """Map props in i,j coordinates to HPC (Helioprojective-Cartesian) coordinates.

        Parameters
        ----------
        src : str
            A source for props.
        meta : str
            A source for disk meta information.
        resolution : float
            Pixel resolution in arcsec.

        Returns
        -------
        batch : HelioBatch
            Batch with props calculated.
        """
        return self._props_mapping(src=src, meta=meta, name='HPC', resolution=resolution)

    @execute(how='threads')
    def _props_mapping(self, i, src, dst, meta, name, resolution=None, get_areas=True):
        """Props mapping."""
        _ = dst
        props = self.data[src][i]
        meta = self.meta[meta][i]
        rad = meta['r']
        i_cen = meta['i_cen']
        j_cen = meta['j_cen']
        if 'B0' in meta:
            B0 = meta['B0']
        elif 'B0' in self.index:
            B0 = self.index.iloc[i]['B0']
        else:
            raise ValueError('B0 is missing.')
        if 'L0' in meta:
            L0 = meta['L0']
        elif 'L0' in self.index:
            L0 = self.index.iloc[i]['L0']
        else:
            raise ValueError('L0 is missing.')
        if meta.get('P', 0) != 0:
            raise ValueError('P angle is not zero.')

        def map2hpc(ij_arr):
            '''Cartesian x, y in arcsec.'''
            ij_arr = np.atleast_2d(ij_arr)
            xy = np.array([ij_arr[:, 1] - j_cen, -ij_arr[:, 0] + i_cen]).T
            return xy * resolution

        def map2hgc(ij_arr):
            '''Carrington lat and long.'''
            ij_arr = np.atleast_2d(ij_arr)
            xy = np.array([ij_arr[:, 1] - j_cen, -ij_arr[:, 0] + i_cen]).T
            carr = xy_to_carr(xy, rad, B0=B0, L0=L0, deg=True)
            return carr[:, ::-1] #lat, long

        def pixel_areas(ij_arr):
            '''Pixel areas in MSH.'''
            ij_arr = np.atleast_2d(ij_arr)
            xy = np.array([ij_arr[:, 1] - j_cen, -ij_arr[:, 0] + i_cen]).T
            spp = xyz_to_sp(xy_to_xyz(xy, rad=rad), deg=False)
            dist = haversine_distances(spp[:, ::-1], np.zeros((1, 2))).ravel()
            return 10**6 / (2 * np.cos(dist) * np.pi * rad**2)

        for prop in props:
            if name.lower() == 'hpc':
                bbox = map2hpc(np.array(prop.bbox).reshape(2, 2))
                bbox = np.hstack([bbox.min(axis=0), bbox.max(axis=0)])
                setattr(prop, 'bbox_hpc', bbox)
                setattr(prop, 'centroid_hpc', map2hpc(prop.centroid).ravel())
                setattr(prop, 'coords_hpc', map2hpc(prop.coords))
                setattr(prop, 'contour_hpc', map2hgc(prop.contour))
                setattr(prop, 'approx_contour_hpc', map2hpc(prop.approx_contour))
            elif name.lower() == 'hgc':
                setattr(prop, 'coords_hgc', map2hgc(prop.coords))
                setattr(prop, 'contour_hgc', map2hgc(prop.contour))
                setattr(prop, 'approx_contour_hgc', map2hgc(prop.approx_contour))
            if get_areas:
                areas = pixel_areas(prop.coords)
                setattr(prop, 'pixel_area_msh', areas)
                setattr(prop, 'area_msh', areas.sum())
        return self

    def get_polygons(self, src, dst, coords='hgc', tolerance=3, cache=True, ):
        """Get polygons from binary mask.

        Parameters
        ----------
        src : str
            A source for binary masks.
        dst : str
            A destination for polygons.
        coords : str
            Coordinates for polygons: 'hgc' for Carrington heliographic, 'plane' for pixel coordinates.

        Returns
        -------
        batch : HelioBatch
            Batch with polygons calculated.
        """
        self.rotate_p_angle(src=src)
        self.get_region_props(src=src, dst=dst, tolerance=tolerance, cache=cache)
        if coords == 'hgc':
            self.props2hgc(src=dst, meta=src)
        self._get_polygons(src=dst, dst=dst, coords=coords)
        return self

    @execute(how='threads')
    def _get_polygons(self, i, src, dst, coords):
        data = self.data[src][i]
        if coords == 'plane':
            self.data[dst][i] = [PlanePolygon(arr.contour[:, ::-1]) for arr in data]
        elif coords == 'hgc':
            self.data[dst][i] = [SphericalPolygon(arr.contour_hgc, deg=True) for arr in data]
        return self

    @execute(how='threads')
    def get_pixel_params(self, i, src, dst, meta=None):
        """Get coordinates and area of individual pixels for regions identified in solar disk image.

        Parameters
        ----------
        src : str
            A source for binary mask.
        dst : str
            A destination for results.
        meta : str, optional
            An optional source for meta information.

        Returns
        -------
        batch : HelioBatch
            Batch with parameters calculated.
        """
        mask = self.data[src][i]
        meta = self.meta[src if meta is None else meta][i]
        rad = meta['r']
        i_cen = meta['i_cen']
        j_cen = meta['j_cen']
        B0 = meta.get('B0', self.index.iloc[i]['B0'])
        L0 = meta.get('L0', self.index.iloc[i]['L0'])
        if meta.get('P', 0) != 0:
            mask = rotate_at_center(mask, meta["P"], center=(meta['j_cen'], meta['i_cen']))
        mask_ij = np.vstack(np.where(mask)).T
        if not mask_ij.size:
            df = pd.DataFrame(columns=['i', 'j', 'Lat', 'Long', 'q', 'phi', 'area'])
            self.data[dst][i] = df
            return self
        mask_xy = np.array([mask_ij[:, 1] - j_cen, -mask_ij[:, 0] + i_cen]).T
        spp = xyz_to_sp(xy_to_xyz(mask_xy, rad=rad), deg=False)
        dist = haversine_distances(spp[:, ::-1], np.zeros((1, 2))).ravel()
        areas = 10**6 / (2 * np.cos(dist) * np.pi * rad**2)
        carr = xy_to_carr(mask_xy, rad, B0=B0, L0=L0, deg=True)
        df = pd.DataFrame({'i': mask_ij[:, 0],
                           'j': mask_ij[:, 1],
                           'Lat': carr[:, 1],
                           'Long': carr[:, 0],
                           'q': np.rad2deg(spp[:, 1]),
                           'phi': np.rad2deg(spp[:, 0]),
                           'area': areas})
        self.data[dst][i] = df
        return self

    @execute(how='threads')
    def filter_regions(self, i, src, dst, min_area, **kwargs):
        """Filter regions with pixel area less than npix.

        Parameters
        ----------
        src : str
            A source for binary mask.
        dst : str
            A destination for results.
        min_area : int
            Minimal area in pixels.
        kwargs : misc
            Additional label keywords.

        Returns
        -------
        batch : HelioBatch
            Batch with filtered masks.
        """
        mask = self.data[src][i]
        if dst != src:
            mask = mask.copy()

        labeled, num = label(mask.astype(int), return_num=True, **kwargs)
        for k in range(1, num + 1):
            obj = np.where(labeled == k)
            if len(obj[0]) < min_area:
                mask[obj] = False

        self.data[dst][i] = mask
        return self

    def show(self, i, image, mask=None, figsize=None, cmap=None, s=None, color=None, **kwargs):
        """Show image data with optional mask countours overlayed.

        Parameters
        ----------
        i : int
            Integer data index.
        image : str
            Image data source.
        mask : str, optional
            Mask to be overlayed.
        figsize : tuple, optional
            Size of figure.
        cmap : cmap
            Matplotlib color map for image.
        s : int
            Point size for contours.
        color : color
            Matplotlib color for contours.
        kwargs : misc
            Additional imshow keywords.
        """
        plt.figure(figsize=figsize)
        img = self.data[image][i]
        plt.imshow(img, cmap=cmap, extent=(0, img.shape[1], img.shape[0], 0), **kwargs)
        if mask is not None:
            mask = np.atleast_1d(mask)
            s = np.atleast_1d(s)
            if len(s) < len(mask):
                assert len(s) == 1
                s = np.full(len(mask), s[0])
            color = np.atleast_1d(color)
            if len(color) < len(mask):
                assert len(color) == 1
                color = np.full(len(mask), color[0])
            for src, size, c in zip(mask, s, color):
                binary = np.rint(self.data[src][i]) == 1
                if binary.shape != img.shape:
                    raise ValueError('Image and mask should have equal shape.')
                cnt = np.where(detect_edges(binary))
                plt.scatter(cnt[1], cnt[0], s=size, color=c)
        plt.xlim([0, img.shape[1]])
        plt.ylim([img.shape[0], 0])
        plt.show()

    def show_sun(self, src, i, figsize=None, ax=None, c='blue', c2='red', c3='green',#pylint:disable=too-many-arguments,too-many-statements,too-many-branches
                 grid=True, grid_lw=0.5, deg=True):
        """Show active regions on the solar disk.

        Parameters
        ----------
        src : str
            Data source.
        i : int
            Integer data index.
        figsize : tuple, optional
            Size of figure.
        kwargs : misc
            Additional imshow keywords.
        """
        p0 = self.meta[src][i].get('P', 0)
        if deg:
            p0 = np.deg2rad(p0)
        if 'L0' in self.meta[src][i]:
            L0 = self.meta[src][i]['L0']
        elif 'L0' in self.index:
            L0 = self.index.iloc[i]['L0']
        else:
            print('L0 angle is not specified in metadata and is set to zero.')
            L0 = 0
        try:
            ndim = self.data[src][i].ndim
        except AttributeError:
            ndim = 1
        if ndim == 1: #data is a list of contours
            if ax is None:
                _, ax = plt.subplots(1, figsize=figsize)
                ax.add_patch(plt.Circle((0, 0), 1, color='#f5ed76'))
            for k, arr in enumerate(self.data[src][i]):
                if isinstance(arr, SphericalPolygon):
                    arr = np.stack([arr.lons, arr.lats]).T #TODO: check for degrees
                    p, v = sp_to_xy(arr - np.array([L0, 0]), deg=True)
                    if not v.all():
                        print('Invalid coordinates at object {}.'.format(k))
                    ax.plot(*p.T, c=c)
                elif isinstance(arr, PlanePolygon):
                    i_cen = self.meta[src][i]['i_cen']
                    j_cen = self.meta[src][i]['j_cen']
                    r = self.meta[src][i]['r']
                    x = (arr.x - j_cen) / r
                    y = (-arr.y + i_cen) / r
                    ax.plot(x, y, c=c)
        else:
            i_cen = self.meta[src][i]['i_cen']
            j_cen = self.meta[src][i]['j_cen']
            r = self.meta[src][i]['r']
            if ax is None:
                _, ax = plt.subplots(1, figsize=figsize)
                ax.add_patch(plt.Circle((0, 0), 1, color='#f5ed76'))
            if ndim == 2: #data is a 2d binary mask
                arr = self.data[src][i][i_cen-r:i_cen + r + 1, j_cen-r:j_cen + r + 1]
                c = matplotlib.colors.colorConverter.to_rgba(c)
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list('ar', [(0, 0, 0, 0), c], 2)
                ax.imshow(arr, extent=[-1, 1, -1, 1], interpolation='none',
                          vmin=0, vmax=1, cmap=cmap, zorder=2)
            if ndim == 3: #data is a 3d binary mask (spots, cores, pores, ...)
                arr = self.data[src][i][i_cen-r:i_cen + r+1, j_cen-r:j_cen + r+1]
                c = matplotlib.colors.colorConverter.to_rgba(c)
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list('ar', [(0, 0, 0, 0), c], 2)
                ax.imshow(arr[..., 0], extent=[-1, 1, -1, 1], interpolation='none',
                          vmin=0, vmax=1, cmap=cmap, zorder=2)
                c2 = matplotlib.colors.colorConverter.to_rgba(c2)
                cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('ar2', [(0, 0, 0, 0), c2], 2)
                ax.imshow(arr[..., 1], extent=[-1, 1, -1, 1], interpolation='none',
                          vmin=0, vmax=1, cmap=cmap2, zorder=2)
                c3 = matplotlib.colors.colorConverter.to_rgba(c3)
                cmap3 = matplotlib.colors.LinearSegmentedColormap.from_list('ar3', [(0, 0, 0, 0), c3], 2)
                ax.imshow(arr[..., 2], extent=[-1, 1, -1, 1], interpolation='none',
                          vmin=0, vmax=1, cmap=cmap3, zorder=2)
        if grid:
            ax.plot([-np.sin(p0), np.sin(p0)], [-np.cos(p0), np.cos(p0)], c='k', lw=grid_lw)
            ax.plot([np.cos(p0), -np.cos(p0)], [-np.sin(p0), np.sin(p0)], c='k', lw=grid_lw)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        ax.axis('equal')
        plt.show()
        return ax

    @execute(how='threads')
    def flip(self, i, src, axis, dst=None, p=True, update_meta=False):
        """Apply axis reversing. Accepts random parameter `p` for aurgemntation.

        Parameters
        ----------
        src : str, tuple of str
            A source for images.
        dst : same type as src
            A destination for results.
        axis : int
            Axis in array, which entries are reversed.
        p : bool or R
            Probabilistic parameter for augmentation, e.g. p = R('choice', a=[True, False], p=[0.5, 0.5])
            will flip images with probability 0.5.
        update_meta : bool
            Update meta with new image orientation. Default to False.

        Returns
        -------
        batch : ImageBatch
            Batch with flipped images.
        """
        if not p:
            return self
        img = self.data[src][i]
        input_shape = img.shape
        self.data[dst][i] = np.flip(img, axis)
        if not update_meta:
            return self

        meta = self.meta[src][i]
        if src != dst:
            meta = meta.copy()
        if axis == 0:
            meta['i_cen'] = input_shape[0] - meta['i_cen']
        elif axis == 1:
            meta['j_cen'] = input_shape[1] - meta['j_cen']
        else:
            raise ValueError('Axis can be only 0 or 1.')
        if src != dst:
            self.meta[dst][i] = meta
        return self

    @execute(how='threads')
    def rot90(self, i, src, dst, axes=(0, 1), k=1, update_meta=False):
        """Apply rotation of an image by 90 degrees given number of times.

        Parameters
        ----------
        src : str, tuple of str
            A source for images.
        dst : same type as src
            A destination for results.
        axes : (2,) array_like
            The array is rotated in the plane defined by the axes.
            Axes must be different. Default to (0, 1).
        k : int or R
            Probabilistic parameter for augmentation, e.g. k = R('choice', a=np.arange(4))
            will rotate images random number of times.
        update_meta : bool
            Update meta with new image orientation. Default to False.

        Returns
        -------
        batch : HelioBatch
            Batch with rotated images.
        """
        if not k:
            return self
        img = self.data[src][i]
        input_shape = img.shape
        self.data[dst][i] = np.rot90(img, k, axes=axes)
        if not update_meta:
            return self

        meta = self.meta[src][i]
        if src != dst:
            meta = meta.copy()
        if k % 4 == 1:
            x = input_shape[1] - meta['j_cen']
            y = meta['i_cen']
        elif k % 4 == 2:
            x = input_shape[0] - meta['i_cen']
            y = input_shape[1] - meta['j_cen']
        elif k % 4 == 3:
            x = meta['i_cen']
            y = input_shape[1] - meta['j_cen']
        else:
            raise ValueError('Unexpected value k = {}.'.format(k))
        meta['i_cen'], meta['j_cen'] = x, y
        if src != dst:
            self.meta[dst][i] = meta
        return self

    def group_by_index(self):
        """Stack batch items according to batch index.

        Returns
        -------
        batch : HelioBatch
            A new batch with stacked items.

        Notes
        -----
        Meta will be lost.
        """
        df = self.index
        index = df.groupby(df.index.name).agg(list)
        batch = self.__class__(BaseIndex(index))
        group_ids = [np.where(df.index.values == i)[0] for i in batch.indices]
        for k, v in self.data.items():
            data = np.array([np.stack(v[ids]) for ids in group_ids] + [None])[:-1]
            batch.data[k] = data
            batch.meta[k] = np.array([None] * len(data))
        return batch

    @execute(how='threads')
    def map_to_synoptic(self, i, src, dst, bins, average=None, deg=True):
        """Make a synoptic map from a solar disk image.

        Parameters
        ----------
        src : str
            A source for images.
        dst : str
            A destination for results.
        bins: (nx, ny)
            Longitudinal and latitudinal resolution of synoptic map.
        average : callable
            Average function. Default is a mean function.
        deg : bool
            If True, all angles are in degrees. Default to True.

        Returns
        -------
        batch : HelioBatch
            Batch with synoptic maps.
        """
        img = self.data[src][i]
        meta = self.meta[src][i]
        B0 = self.index.iloc[i, self.index.columns.get_loc('B0')]
        L0 = self.index.iloc[i, self.index.columns.get_loc('L0')]
        res = make_synoptic_map(img, i_cen=meta['i_cen'], j_cen=meta['j_cen'],
                                rad=meta['r'], B0=B0, L0=L0,
                                bins=bins, average=average, deg=deg)
        self.data[dst][i] = res
        return self

    @execute(how='threads')
    def stack_synoptic_maps(self, i, src, dst, shift, scale, weight_decay=None, deg=True):
        """Stack synoptic maps corresponding to disk images into a single map.

        Parameters
        ----------
        src : str
            A source for synoptic maps corresponding to disk images.
        dst : str
            A destination for results.
        shift : scalar
            A parameter for pixel weights according to `weight_decay(-(dist - shift) / scale)`.
        scale : scalar
            A parameter for pixel weights according to `weight_decay(-(dist - shift) / scale)`.
        weight_decay: callable
            Function to get a pixel weight based on its distance from central meridian.
            Distance unit should be degree. Default is sigmoid function.
        deg : bool
            If True, all angles are in degrees. Default to True.

        Returns
        -------
        batch : HelioBatch
            Batch with synoptic maps.
        """
        maps = self.data[src][i]
        xedges = np.linspace(0, 360, maps.shape[-1] + 1)
        lng = (xedges[1:]  + xedges[:-1]) / 2
        L0 = self.index.iloc[i, self.index.columns.get_loc('L0')]
        L0 = np.asarray(L0) if deg else np.rad2deg(L0)

        dist = abs((lng - L0.reshape((-1, 1)) + 180) % 360 - 180)
        dist = np.zeros_like(maps) + np.expand_dims(dist, 1)

        if weight_decay is None:
            weight_decay = sigmoid

        dist = weight_decay(-(dist - shift) / scale)
        dist[np.isnan(maps)] = 0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = np.nansum(dist * maps, axis=0) / dist.sum(axis=0)

        self.data[dst][i] = res
        return self

    @execute(how='loop')
    def make_polar_plots(self, i, src, dst, path, figsize=None, axes=False,
                         labelsize=14, pad=None, **kwargs):
        """Make polar projections from a synoptic map.

        Parameters
        ----------
        src : str
            A source for synoptic maps.
        path : str
            Path where to write output files.
        figsize : tuple
            Output size of figure.
        kwargs : misc
            Additional positional argumets to `pcolormesh`.

        Returns
        -------
        batch : HelioBatch
            Batch unchanged.
        """
        _ = dst
        fname = os.path.join(path, str(self.indices[i]))
        data = self.data[src][i]
        n_size = data.shape[0] // 2

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)
        theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, data.shape[1]),
                                 np.linspace(0, 90, n_size))
        ax.pcolormesh(theta, phi, data[:n_size], **kwargs)
        if not axes:
            plt.axis('off')
        else:
            ax.tick_params(axis='both', which='major', labelsize=labelsize, pad=pad)
            plt.yticks(np.arange(0, 91, 15),
                       [str(i) + str('$^\circ$') for i in 90-np.arange(0, 91, 15)],
                       color='w')
        plt.savefig(fname + '_N.jpg')
        plt.close(fig)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)
        theta, phi = np.meshgrid(np.linspace(0, 2*np.pi, data.shape[1]),
                                 np.linspace(-90, 0, data.shape[0] - n_size))
        ax.pcolormesh(theta, phi, data[n_size:][::-1], **kwargs)
        if not axes:
            plt.axis('off')
        else:
            ax.tick_params(axis='both', which='major', labelsize=labelsize, pad=pad)
            plt.yticks(np.arange(-90, 1, 15),
                       [str(-i) + str('$^\circ$') for i in 90-np.arange(0, 91, 15)],
                       color='w')
        plt.savefig(fname + '_S.jpg')
        plt.close(fig)

    @execute(how='threads')
    def aia_intscale(self, i, src, dst, wavelength):
        """Adjust intensity to AIA standard.

        Parameters
        ----------
        src : str
            A source for image.
        dst : str
            A destination for results.
        wavelength : int
            Image wavelength.

        Returns
        -------
        batch : HelioBatch
            Batch with adjusted images.
        """
        img = self.data[src][i]
        exptime = self.meta[src][i]['EXPTIME']
        if wavelength == 193:
            img = np.log10(np.clip(img*(2.99950 / exptime), 120.0, 6000))
            minv = img.min()
            img = (img - minv) / (img.max() - minv)
        else:
            raise ValueError('Unknown wavelength {}.'.format(wavelength))
        self.data[dst][i] = img
        return self

    @execute(how='threads')
    def match_histogram(self, i, src, dst, reference):
        """Histogram matching.

        Parameters
        ----------
        src : str
            A source for image.
        dst : str
            A destination for results.
        reference : str
            Reference image to match histogram with.

        Returns
        -------
        batch : HelioBatch
            Batch with adjusted images.
        """
        source = self.data[src][i]
        ref = self.data[reference][i]
        res = np.full(source.shape, np.nan)
        mask = np.isnan(source)
        source = source[~mask]
        ref = ref[~np.isnan(ref)]
        _, src_ind, src_counts = np.unique(source, return_inverse=True, return_counts=True)
        ref_vals, ref_counts = np.unique(ref, return_counts=True)

        src_q = np.cumsum(src_counts) / len(source)
        ref_q = np.cumsum(ref_counts) / len(ref)

        res[~mask] = np.interp(src_q, ref_q, ref_vals)[src_ind]
        self.data[dst][i] = res
        return self

    @execute(how='threads')
    def deg2sin(self, i, src, dst):
        """Transform a Latitude synoptic map to Sine Latitude.

        Parameters
        ----------
        src : str
            A source for synoptic map.
        dst : str
            A destination for results.

        Returns
        -------
        batch : HelioBatch
            Batch with transformed synoptic maps.
        """
        img = self.data[src][i]
        q = np.linspace(-np.pi/2, np.pi/2, img.shape[0] + 1)
        q = (q[:-1] + q[1:]) / 2
        x = np.linspace(-1, 1, img.shape[0] + 1)
        x = (x[:-1] + x[1:]) / 2
        q2 = np.arcsin(x)
        self.data[dst][i] = np.array([interpolate.interp1d(q, y)(q2) for y in img.T]).T
        return self

    @execute(how='threads')
    def sin2deg(self, i, src, dst):
        """Transform a Sine Latitude synoptic map to Latitude.

        Parameters
        ----------
        src : str
            A source for synoptic map.
        dst : str
            A destination for results.

        Returns
        -------
        batch : HelioBatch
            Batch with transformed synoptic maps.
        """
        img = self.data[src][i]
        q = np.linspace(-np.pi/2, np.pi/2, img.shape[0] + 1)
        q = (q[:-1] + q[1:]) / 2
        x = np.linspace(-1, 1, img.shape[0] + 1)
        x = (x[:-1] + x[1:]) / 2
        x2 = np.sin(q)
        self.data[dst][i] = np.array([interpolate.interp1d(q, y)(x2) for y in img.T]).T
        return self

    def label_synoptic_map(self, src, dst):
        """Label regions in a binary synoptic map. Takes into account connections
        at 360 and 0 longitude.

        Parameters
        ----------
        src : str
            A source for binary synoptic map.
        dst : str
            A destination for labeled map.

        Returns
        -------
        batch : HelioBatch
            Batch with labeled synoptic maps.
        """
        return self.apply(func=label360, src=src, dst=dst)

    @execute(how='threads')
    def region_statistics(self, i, src, dst, sin, hmi=None):
        """Calculates statistics for regions in a binary synoptic map:
            - area (:math:`10^{12}` :math:`km^2`)
            - mean latitude (in degrees, North at :math:`90^{\circ}`)
            - largest Carrington longitude (in degrees)
            - positive flux (:math:`10^{22}` Mx, if hmi is not None and in Gauss)
            - negative flux (:math:`10^{22}` Mx, if hmi is not None and in Gauss)

        Parameters
        ----------
        src : str
            A source for binary synoptic map.
        dst : str
            A destination for statistics.
        sin : bool
            Sine Latitude synoptic map.
        hmi : str, optional
            Magnetic synoptic map for flux calculation. Default to None.

        Returns
        -------
        batch : HelioBatch
            Batch with statistics.
        """
        mask = self.data[src][i]
        if hmi is not None:
            hmi = self.data[hmi][i]
        self.data[dst][i] = region_statistics(mask, sin=sin, hmi=hmi)
        return self
