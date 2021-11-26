"""Index classes."""
import re
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import dateutil.parser as dparser
try:
    from sunpy.coordinates.sun import carrington_rotation_number, L0, B0
except ImportError:
    pass

class BaseIndex(pd.DataFrame): #pylint: disable=abstract-method
    """Base index class."""

    @property
    def _constructor(self):
        return self.__class__

    @property
    def indices(self):
        """Unique indices."""
        return self.index.unique().values

    def train_test_split(self, train_ratio=0.8, shuffle=True, seed=None):
        """Splits index into train and test subsets.

        Parameters
        ----------
        train_ratio : float, in [0, 1]
            Ratio of the train subset to the whole dataset. Default to 0.8.
        shuffle : bool
            If True, index will be shuffled before splitting into train and test.
            Default to True.
        seed : int
            Seed for the random number generator.

        Returns
        -------
        train, test : BaseIndexer
            Train and test indices.
        """
        indices = self.index.unique().values
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        train_size = int(train_ratio * len(indices))
        train = self.loc[indices[:train_size]]
        test = self.loc[indices[train_size:]]
        return train, test

    def shuffle(self, seed=None):
        """Randomly shuffle indices.

        Parameters
        ----------
        seed : int
            Seed for the random number generator.

        Returns
        -------
        index : BaseIndex
            BaseIndex with shuffled indices.
        """
        indices = self.index.unique().values
        np.random.seed(seed)
        np.random.shuffle(indices)
        return self.loc[indices]

    def index_merge(self, x):
        """Merge on left and right indexes."""
        return self.merge(x, left_index=True, right_index=True)


class FilesIndex(BaseIndex): #pylint: disable=abstract-method,too-many-ancestors
    """Index of files.

    Builds a DataFrame containing path to files in columns. By default
    the DataFrame is indexed by filnames.

    Paramerers
    ----------
    kwargs : dict of type {name: path}
        Path to files that should be indexed and reference name for files collection.
        Path can contain shell-style wildcards.
    """
    def __init__(self, *args, **kwargs):
        if not kwargs:
            super().__init__(*args, **kwargs)
        elif len(kwargs) > 1:
            raise ValueError('Only a single keyword in allowed, got {}'.format(len(kwargs)))
        else:
            name, path = next(iter(kwargs.items()))
            files = glob.glob(path)
            index = pd.Index([Path(f).stem for f in files], name=self.__class__.__name__)
            super().__init__({name: files}, index=index)

    def parse_datetime(self, regex=None):
        """Extract date and time from index.

        Parameters
        ----------
        regex : str, optional
            Datetime pattern to search in index.

        Returns
        -------
        index : FilesIndex
            FilesIndex with a column DateTime added.
        """
        if regex is None:
            dt = [dparser.parse(i, fuzzy=True) for i in self.index]
        else:
            dt = [dparser.parse(re.search(regex, i).group(0), fuzzy=True) for i in self.index]
        self['DateTime'] = dt
        return self

    def get_sun_params(self):
        """"Get L0, B0 and carrington rotation number corresponding to DateTime column.

        Returns
        -------
        index : FilesIndex
            FilesIndex with columns L0, B0, CR added.
        """
        t = self['DateTime']
        self['L0'] = L0(t)
        self['B0'] = B0(t)
        self['CR'] = carrington_rotation_number(t).astype(int)
        return self
