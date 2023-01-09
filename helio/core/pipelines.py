"""Pipelines."""
import os
from tqdm import tqdm

from .batch_sampler import BatchSampler
from .batch import HelioBatch
from .index import KislovodskFilesIndex

def get_kislovodsk_data(series, start_date, end_date, path,
                        batch_size=10, rename=True, progress_bar=True):
    """Get countours of active regions from archive of the Kislovodsk Mountain Astronomical Station.

    Paramerets:
    ----------
    series : str
        Type of active regions. Possible options are 'CA' for plages,
        'CH' for coronal holes, 'FIL' for filaments, 'SPOT' for sunspots.
    start_date : str
        First date of the time series, for example, '2022-01-01'.
    end_date : str
        Last date (included) of the time series, for example, '2022-12-31'.
    path : str
        Path to the local directory where to save the files.
    batch_size : int, optional
        Number of files processed in parallel. Default 10.
    rename : bool, optional
        If True, files will be renamed uniformely. Otherwise, keep original filenames. Default True.
    progress_bar : bool, optional
        Show the progress bar. Default True.
    """
    if not os.path.exists(path):
        raise ValueError("Path {} does not exists.".format(path))
    series = series.upper()
    index = KislovodskFilesIndex(series=series, start_date=start_date, end_date=end_date)
    if rename:
        fmt = '%Y-%m-%dT%H%M%S'
        new_ids = series + '_' + index['DateTime'].apply(lambda x: x.strftime(fmt))
        index.set_index(new_ids, inplace=True)
    sampler = BatchSampler(index, batch_size=batch_size)
    for ids in (tqdm(sampler) if progress_bar else sampler):
        batch = HelioBatch(ids)
        batch.load(src=series, meta=series, raise_errors=False)
        if series != 'CH':
            batch.get_polygons(src=series, dst=series, coords='hgc')
        batch.dump(src=series, path=path, format='json')
