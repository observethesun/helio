"""HelioBatch class."""
import os
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import dill
from astropy.io import fits
from aiapy.calibrate import correct_degradation
from scipy import interpolate
from skimage.io import imread
import skimage
import skimage.transform
from skimage.transform import resize
try:
    import blosc
except ImportError:
    warnings.warn('blosc module not found, this file format will be not available.', ImportWarning)

from .decorators import execute, add_actions, extract_actions, TEMPLATE_DOCSTRING
from .index import BaseIndex
from .synoptic_maps import make_synoptic_map, label360, region_statistics
from .io import load_fits, load_abp_mask, write_syn_abp_file, write_fits
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
            Index column labels with data source specifications.
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
        if fmt == 'blosc':
            with open(path, 'rb') as f:
                data = dill.loads(blosc.decompress(f.read()))
        elif fmt == 'npz':
            f = np.load(path)
            keys = list(f.keys())
            if len(keys) != 1:
                raise ValueError('Expected single key, found {}.'.format(len(keys)))
            data = f[keys[0]]
        elif fmt == 'abp':
            data = load_abp_mask(path, **kwargs)
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
        |
        i
        |
        ----j----
        """
        _ = kwargs
        path = self.index.iloc[i, self.index.columns.get_loc(src)]
        fmt = Path(path).suffix.lower()[1:]
        if fmt == 'abp':
            with open(path, 'r') as fin:
                fread = fin.readlines()
                header = np.array(fread[0].split())
                meta = dict(i_cen=int(header[1]),
                            j_cen=int(header[0]),
                            r=int(header[2]))
        elif fmt in ['fts', 'fits']:
            hdul = fits.open(path)
            hdul.verify(verify)
            hdr = hdul[unit].header
            meta = dict(hdr.items())
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
        fname = os.path.join(path, str(self.indices[i])) + '.' + format
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_position([0, 0, 1, 1])
        ax.imshow(img, cmap=cmap)
        pts = self.data[scatter][i]
        ax.scatter(pts[1], pts[0], **scatter_kw)
        plt.axis('off')
        plt.savefig(fname, dpi=dpi)
        plt.close(fig)
        return self

    @execute(how='threads')
    def _dump(self, i, src, dst, path, format, **kwargs): #pylint: disable=redefined-builtin
        """Dump data in various formats."""
        _ = dst
        fname = os.path.join(path, str(self.indices[i])) + '.' + format
        data = self.data[src][i]
        if format == 'blosc':
            with open(fname, 'w+b') as f:
                f.write(blosc.compress(dill.dumps(data)))
        elif format == 'npz':
            np.savez(fname, data, **kwargs)
        elif format == 'txt':
            np.savetxt(fname, data, **kwargs)
        elif format == 'binary':
            data.tofile(fname)
        elif format == 'abp':
            write_syn_abp_file(fname, data)
        elif format == 'fits':
            meta = self.meta[kwargs.pop('meta', src)][i]
            write_fits(fname, data, index=self.index.iloc[[i]], meta=meta, **kwargs)
        else:
            plt.imsave(fname, data, format=format, **kwargs)
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
    def disk_resize(self, i, src, dst, output_shape, **kwargs):
        """Resize solar disk image and update center location and radius.

        Parameters
        ----------
        src : str
            A source for solar disk images.
        output_shape : tuple
            Shape of output images. Axes ratio should be the same as for source image.
        kwargs : misc
            Any additional named arguments to resize method.

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
            binary = np.rint(self.data[mask][i]) == 1
            if binary.shape != img.shape:
                raise ValueError('Image and mask should have equal shape.')
            cnt = np.where(detect_edges(binary))
            plt.scatter(cnt[1], cnt[0], s=s, color=color)
        plt.xlim([0, img.shape[1]])
        plt.ylim([img.shape[0], 0])
        plt.show()

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
