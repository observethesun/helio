"""Contains BatchSampler class."""
import numpy as np


class BatchSampler: #pylint: disable=too-many-instance-attributes
    """Get next batch of indices.

    Parameters
    ----------
    batch_size : int
        Size of target batch.
    n_epochs : int or None
        Maximal number of epochs. If None then loops are infinitely. Default to 1.
    suffle : bool
        Shuffle indices after each epoch. Default to False.
    drop_incomplete : bool
        Defines handling of incomplete batches in the end of epochs.
        If drop_incomplete is False we complete the batch with items from the beginning
        of the current epoch and return incomplete batch if current epoch
        is the last epoch. If drop_incomplete is True we go to the next epoch.
        Default to False.

    Returns
    -------
    index : index
        Next indices.
    """
    def __init__(self, index, batch_size, n_epochs=1, shuffle=False, drop_incomplete=False,
                 seed=None):
        self._index = index
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._shuffle = shuffle
        self._drop_incomplete = drop_incomplete
        self._seed = seed
        self._batch_start = 0
        self._on_epoch = 0
        self._indices = index.index.unique().values
        self._order = self._indices.copy()

    @property
    def indices(self):
        """Indices to sample from."""
        return self._indices

    @property
    def params(self):
        """Sampler parameters."""
        return dict(batch_size=self._batch_size,
                    n_epochs=self._n_epochs,
                    shuffle=self._shuffle,
                    drop_incomplete=self._drop_incomplete,
                    seed=self._seed)

    def __len__(self):
        if self._drop_incomplete:
            return self._n_epochs * (len(self._indices) // self._batch_size)
        if len(self._indices) % self._batch_size == 0:
            return self._n_epochs * (len(self._indices) // self._batch_size)
        return self._n_epochs * (len(self._indices) // self._batch_size + 1)

    def reset(self):
        """Reset epochs counter to 0."""
        if self._shuffle:
            np.random.seed(self._seed)
            np.random.shuffle(self._order)
        self._batch_start = 0
        self._on_epoch = 0
        return self

    def __iter__(self):
        return self.reset()

    def __next__(self):
        if (self._n_epochs is not None) and (self._on_epoch >= self._n_epochs):
            raise StopIteration
        if self._batch_size + self._batch_start <= len(self.indices):
            a, b = self._batch_start, self._batch_start + self._batch_size
        else:
            if (self._n_epochs is not None) and (self._on_epoch == self._n_epochs - 1):
                self._on_epoch += 1
                if self._drop_incomplete:
                    return next(self)
                a, b = self._batch_start, len(self.indices)
            else:
                if self._drop_incomplete:
                    self._on_epoch += 1
                    self._batch_start = 0
                    if self._shuffle:
                        np.random.shuffle(self._order)
                    return next(self)
                self._on_epoch += 1
                a, b = self._batch_start, (self._batch_start + self._batch_size) % len(self.indices)
                next_items = np.hstack([self._order[a:], self._order[:b]])
                self._batch_start = b
                if self._shuffle:
                    np.random.shuffle(self._order)
                return self._index.loc[next_items]
        next_items = self._order[a: b]
        if b == len(self.indices):
            self._batch_start = 0
            self._on_epoch += 1
            if self._shuffle:
                np.random.shuffle(self._order)
        else:
            self._batch_start = b
        return self._index.loc[next_items]
