from codecs import open
from os import path
from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# Requirements
install_requires=['cython>=0.24.1',
                  'numpy>=1.6.1',
                  'scipy>=0.16',
                  'matplotlib>=1.5.1',
                  'scikit-learn>=0.17.1',
                  'nibabel>=2.0.2',
                  'nilearn>=0.2.4',
                  'GPy>=1.0.7']

setup(

    name='connectopic_mapping',

    version='0.3.0',

    description='Connectopic mapping',
    long_description=long_description,

    author='Michele Damian',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='neuroscience connectopic mapping research',

    packages=['connectopic_mapping'],

    install_requires=install_requires,

    cmdclass={'build_ext': build_ext},

    ext_modules=[Extension("connectopic_mapping.haak", ["connectopic_mapping/haak.pyx"], include_dirs=[numpy.get_include()])],

)
