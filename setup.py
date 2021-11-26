"""
Helio
"""

from setuptools import setup, find_packages

setup(
    name='helio',
    packages=find_packages(exclude=['tutorials', 'docs']),
    version='0.1',
    url='https://github.com/observethesun/helio',
    license='Apache License 2.0',
    author='illarionovEA',
    description='A machine learning framework for solar data processing',
    zip_safe=False,
    platforms='any',
    install_requires=[
        'numpy>=1.13.1',
        'scipy>=0.19.1',
        'matplotlib>=2.1.0',
        'tqdm>=4.19.7',
        'sunpy>=1.1.1',
        'astropy>=4.0',
        'aiapy>=0.1.0',
        'scikit-image>=0.16.2',
        'scikit-learn>=0.21.2',
        'dill>=0.3.1.1'
    ],
    extras_require={
        'numpydoc': ['numpydoc>=0.8.0'],
        'tensorflow-gpu': ['tensorflow-gpu==2.5.2']
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering'
    ],
)
