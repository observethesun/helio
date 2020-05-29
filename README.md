[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.4-orange.svg)](https://tensorflow.org)

# helio

Machine learning framework for solar data processing.

Key features:
* a variety of supported file formats (image files, fits, npz, txt)
* ndimage processing (resize, rotate, ``skimage.transforms``, etc)
* solar image processing (e.g. construction of synoptic maps)
* batch-by-batch processing of large datasets
* implemented neural networks modules and architectures

For more features see API [documentation](http://observethesun.github.io/helio/).

## Installation

Clone the repository
```
git clone https://github.com/observethesun/helio.git
```

## Quick start

Index files to be processed:

```python
index = FilesIndex(images='../aia193/*.fits')
```

Load all data at once (for small datasets):

```python
batch = HelioBatch(index).load('images')
```

Make some processing:

```python
batch.resize(src='images', dst='resized', output_shape=(256, 256), preserve_range=True)
```

Organize batch-by-batch processing of large datasets:

```python
batch_sampler = BatchSampler(index, batch_size=10)

for ids in batch_sampler:
    (HelioBatch(ids).load('images')
     .resize(src='images', output_shape=(256, 256), preserve_range=True))
```

For more examples see [tutorials](./tutorials).
