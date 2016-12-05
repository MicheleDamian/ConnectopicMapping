Run
===

To run the pipeline simply create an object of the class :py:class:`connectopic_mapping.Haak` and then call :py:meth:`connectopic_mapping.Haak.fit_transorm` to extract the connectopies.

Haak pipeline
-------------

The script `run.py` implements the pipeline developed in Haak and colleagues paper "Connectopic mapping with resting-state fMRI"; it provides an example of how to compute and visualize the connectopies from a resting state fMRI image provided by the `Human Connectome Project <https://www.humanconnectome.org>`_. As such, it expects as input two 15-minutes, 0.72-seconds temporal and 2mm spatial resolution, denoised and registered volume-based scans preprocessed with the HCP functional and ICA-FIX pipelines. The parameters of the script can be changed by editing the configuration file `config.json`. The following section contains a definition of these parameters.

It is possible to test the pipeline against the results provided in the Notes page. To do it run the pipeline for two sessions of the same subject. Then compute the normalized root mean squared error as follow:
::

   import numpy
   from sklearn.metrics import mean_squared_error
   from math import sqrt

   connectopies_session_1 = numpy.load('filename_connectopic_map_session_1')
   connectopies_session_2 = numpy.load('filename_connectopic_map_session_2')

   rmse = sqrt(mean_squared_error(connectopic_map_session_1, connectopic_map_session_2))
   nrmse = rmse / (numpy.max(connectopic_map_session_1) - numpy.min(connectopic_map_session_2))

   print("nRMSE = {0}".format(nrmse))


Parameters
----------

subject
   Number of the subject as in the filename of the Human Connectome Project's Nifti image.

session
   Name of the session as in the filename of the Human Connectome Project's Nifti image. Usually "REST1" or "REST2".

scans
   List of names of the scans as in the Human Connectome Project's Nifti images. This script concatenates two scans named "LR" and "RL".

hemisphere
   Name of the hemisphere as in the Human Connectome Project's Nifti image. Usually "RH" or "LH".

nifti_dir_path
   Path where the Human Connectome Project's Nifti image are stored.

atlas_name
   Name of the labeled atlas to use from the Harvard-Oxford dataset. Refer to :py:meth:`connectopic_mapping.utils.load_masks`

roi_name
   Name of the ROI to use as named in the atlas. Refer to :py:meth:`connectopic_mapping.utils.load_masks`.

num_lower_dim
   See parameter ``num_lower_dim`` in :py:meth:`connectopic_mapping.Haak`.

num_processes
   See parameter ``num_processes`` in :py:meth:`connectopic_mapping.Haak`.

manifold_learning
   See parameter ``manifold_learning`` in :py:meth:`connectopic_mapping.Haak`.

manifold_components
   See parameter ``manifold_components`` in :py:meth:`connectopic_mapping.Haak`.

out_path
   See parameter ``out_path`` in :py:meth:`connectopic_mapping.Haak`.

verbose
   See parameter ``verbose`` in :py:meth:`connectopic_mapping.Haak`.

figures
    Add a `json` list of the following parameters for each slice of the brain to be visualized:

   * axis_x
       Integer coordinate of the plane orthogonal to the X axis to visualize;

   * axis_y
       Integer coordinate of the plane orthogonal to the Y axis to visualize;

   * axis_z
       Integer coordinate of the plane orthogonal to the Z axis to visualize;

   * legend_location
       Location of the legend in the graph. 1 := North-East, 2 := North-West, 3 := South-West and 4 := South-East. If None the legend is not visualized.

   See :py:meth:`connectopic_mapping.utils.visualize_volume`.
