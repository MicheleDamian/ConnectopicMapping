Connectopic Mapping
===================

:Author: Michele Damian
:Contact: michele.damian@gmail.com
:Date: 2016-07-06
:Version: 0.1.0br1

This module contains the tools to extract a connectopic mapping of the brain
from resting state fMRI data. The pipeline used is based on Haak et al. paper
"Connectopic Mapping with resting state fMRI", 2016.

Dependencies
------------

The following are the dependencies' lower versions tested to work:

- `Python <http://www.python.org/>`_ (>= 3.5.1)
- `Numpy <http://www.numpy.org/>`_ (>= 1.6.1)
- `Scipy <http://www.scipy.org/>`_ (>= 0.16)
- `Matplotlib <http://www.matplotlib.org/>`_ (>= 1.5.1)
- `Scikit-learn <http://www.scikit-learn.org//>`_ (>= 0.17.1)
- `Nibabel <http://www.nipy.org/nibabel/>`_ (>= 2.2)
- `Nilearn <http://nilearn.github.io/>`_ (>= 0.2.4)
- `GPy <https://github.com/SheffieldML/GPy/>`_ (>= 1.0.7)

Even if it is recommended to install Python 3.5, it is possible that
Python 3.1, 3.2, 3.3 and 3.4 also work normally. Connectopic mapping doesn't
support Python 2.

Install
-------

To install the package save the folder `connectopic_mapping` on your hard drive
and run:
::

   pip install connectopic_mapping

Alternatively, it is possible to just add the same folder to the `PYTHONPATH`
without installing it.
