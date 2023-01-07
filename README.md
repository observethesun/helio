[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://python.org)

# helio

A framework for solar data processing.

Key features:
* batch-by-batch processing of large datasets
* a variety of supported file formats
* downloading data from data archives
* standard image processing tools (resize, rotate, ``skimage.transforms``)
* specific methods for processing of solar images (e.g. construction of synoptic maps)

For more features see the [documentation](http://observethesun.github.io/helio/).

## Installation

Clone the repository
```
git clone https://github.com/observethesun/helio.git
```
or download the ZIP [archive](https://github.com/observethesun/helio/archive/refs/heads/master.zip).

## Quick start

Index files to be processed:

```python
index = FilesIndex(path='./data/*.fits', name='img')
```

Load all data at once (for small datasets):

```python
batch = HelioBatch(index).load('img')
```

Make some processing:

```python
batch.resize(src='img', dst='resized', output_shape=(256, 256))
```

Organize batch-by-batch processing of large datasets:

```python
batch_sampler = BatchSampler(index, batch_size=10)

for ids in batch_sampler:
    (HelioBatch(ids).load('img')
     .resize(src='img', output_shape=(256, 256))
     .dump(src='img', path='./processed', format='jpg', cmap='gray'))
```

For more examples see [tutorials](./tutorials). 
