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
    author='ObserveTheSun',
    description='A machine learning framework for solar data processing',
    zip_safe=False,
    platforms='any',
    install_requires=[],
    extras_require={},
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