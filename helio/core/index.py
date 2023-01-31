"""Index classes."""
import re
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import dateutil.parser as dparser
from bs4 import BeautifulSoup
import requests
try:
    from sunpy.coordinates.sun import carrington_rotation_number, L0, B0
except ImportError:
    pass

KISL_URL = 'http://158.250.29.123:8000/web/obs/{}/{}/'
EXT_LIST = dict(ch='cnt', spot='abp', fil='abp', ca='abp')

def list_files(url, ext=''):
    """Get list of remote files with given extension.
    """
    page = requests.get(url).text #pylint:disable=missing-timeout
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if
            node.get('href').endswith(ext)]


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
        (train, test) : tuple
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
    """Index of local files.

    Builds a DataFrame containing file paths and indexed by filnames.

    Parameters
    ----------
    path : str
        Path to files that should be indexed. Path can contain shell-style wildcards.
    name : str, optional
        Name for files collection.
    """
    def __init__(self, *args, path=None, name=None):
        if path is None:
            super().__init__(*args)
        else:
            files = glob.glob(path)
            index = pd.Index([Path(f).stem for f in files], name=self.__class__.__name__)
            super().__init__(pd.DataFrame({name: files}, index=index))

    def parse_datetime(self, regex=None, format=None, **kwargs): #pylint:disable=redefined-builtin
        """Extract date and time from index.

        Parameters
        ----------
        regex : str, optional
            Datetime pattern to search in index.
        format : str, optional
            The strftime to parse time.
        kwargs : dict
            Additional keywords for pandas.to_datetime if format is not None.

        Returns
        -------
        index : FilesIndex
            FilesIndex with a column DateTime added.
        """
        if format is not None:
            dt = pd.to_datetime(self.index, format=format, **kwargs)
        else:
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

class RemoteFilesIndex(FilesIndex): #pylint:disable=too-many-ancestors
    """Index of remote files.

    Builds a DataFrame containing urls and indexed by filnames.

    Parameters
    ----------
    url : str
        URL where remote files are located.
    ext : str
        Files extention.
    name : str, optional
        Name for files collection.
    """
    def __init__(self, *args, url=None, ext=None, name=None):
        if url is None:
            super().__init__(*args)
        else:
            files = list_files(url, ext)
            index = pd.Index([Path(f).stem for f in files], name=self.__class__.__name__)
            super().__init__(pd.DataFrame({name: files}, index=index))


class KislovodskFilesIndex(RemoteFilesIndex): #pylint:disable=too-many-ancestors
    """Index for the archive of the Kislovodsk Mountain Astonomical Station.

    Builds a DataFrame containing urls and indexed by filnames.

    Parameters
    ----------
    series : str
        Data series. 'spot' for sunspots, 'ca' for plages, 'fil' for filaments, 'ch' for coronal holes.
    start_date : str
        Start date, YYYY-MM-DD.
    end_date : str
        End date, YYYY-MM-DD.
    ext : str, optional
        Files extention.
    """
    def __init__(self, *args, series=None, start_date=None, end_date=None, ext=None):
        if series is None:
            super().__init__(*args)
        else:
            if start_date is None:
                raise ValueError('Start date is not specified.')
            if end_date is None:
                raise ValueError('End date is not specified.')
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            series = series.lower()
            if series not in ['ca', 'ch', 'fil', 'spot']:
                raise ValueError("Series shoud be one of 'CA', 'CH', 'FIL', 'SPOT'.")
            ids = []
            for year in range(start_date.year, end_date.year + 1):
                url = KISL_URL.format(str(year), series)
                index = RemoteFilesIndex(url=url,
                                         ext=EXT_LIST[series] if ext is None else ext,
                                         name=series.upper())
                if series == 'ca':
                    index = index.parse_datetime(format='%Y%m%d%H%M' if year < 2000
                                                 else '%y%m%d%H%M', errors='coerce')
                elif series == 'ch':
                    index = index.parse_datetime()
                if series == 'spot':
                    index = index.loc[(index.index.map(len) == 12) |
                                      (index.index.map(len) == 25)]
                    index = index.parse_datetime()
                elif series == 'fil':
                    index['DateTime'] = index.index.map(lambda x: re.sub("[^0-9]", "", x))
                    index = index.loc[(index.DateTime.map(len) == 12) |
                                      (index.DateTime.map(len) == 14)]
                    dt = pd.to_datetime(index.DateTime, format='%Y%m%d%H%M%S', errors='coerce')
                    index['DateTime'] = dt
                ids.append(index)
            index = pd.concat(ids)
            index.index.name = self.__class__.__name__
            index = index.loc[(index.DateTime.dt.date >= start_date) & (index.DateTime.dt.date <= end_date)]
            super().__init__(index)
