from setuptools import setup, Command
from codecs import open
from os import path

currPath = path.abspath(path.dirname(__file__))
with open(path.join(currPath, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='tsfracdiff',
    version='1.0.0',
    description='Efficient and easy to use fractional differentiation transformations for stationarizing time series data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AdamWLabs/tsfracdiff',
    author='Adam Wu',
    author_email='adamwu1@outlook.com',
    packages=['tsfracdiff'],
    classifiers=[
	'Intended Audience :: Science/Research',
	'Topic :: Scientific/Engineering :: Information Analysis',
	'Programming Language :: Python',
	'Operating System :: OS Independent',
	'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'arch',
	'joblib'
    ],
    license_files = ('LICENSE',),
)