from codecs import open
from os import path
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='connectopic_mapping',

    version='0.3.0',

    description='Connectopic mapping',
    long_description=long_description,

    author='Michele Damian',
    author_email='michele.damian@gmail.com',

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

    #packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    packages=['connectopic_mapping'],

    cmdclass={'build_ext': build_ext},

    ext_modules=[Extension("connectopic_mapping.haak", ["connectopic_mapping/haak.pyx"], include_dirs=[numpy.get_include()])]

)
