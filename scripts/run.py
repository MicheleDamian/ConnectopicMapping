#!/usr/bin/env python

""" Run the pipeline for Haak connectopic mapping.

    Consider changing the parameters contained in config.json to
    set input and output folder and experiment with different behaviors
    of the algorithm.

"""

import json
import os
import numpy
from connectopic_mapping import haak, utils
from matplotlib import pyplot

with open('config.json') as config_file:
    config = json.load(config_file)

#
# Define general parameters
#
subject = config["subject"]
session = config["session"]
scans = config["scans"]
hemisphere = config["hemisphere"]
atlas_name = config["atlas_name"]
roi_name = config["roi_name"]

#
# Define Haak parameters
#
num_lower_dim = config["num_lower_dim"]
num_processes = config["num_processes"]
manifold_learning = config["manifold_learning"]
manifold_components = config["manifold_components"]
out_path = config["out_path"]
verbose = config["verbose"]

#
# Define input/output locations
#
image_path = config["nifti_dir_path"]

image_path_0 = image_path + \
               '/rfMRI_{1}_{2}_hp2000_clean.nii.gz' \
               .format(subject, session, scans[0])

image_path_1 = image_path + \
               '/rfMRI_{1}_{2}_hp2000_clean.nii.gz' \
               .format(subject, session, scans[1])

out_path = config["out_path"] + \
           '/rfMRI_{0}_{1}_{2}' \
           .format(subject, session, hemisphere)

#
# Load ROI and brain masks
#

print("Loading brain and ROI masks from atlas...", end="", flush=True)

brain_mask, roi_mask = utils.load_masks(atlas_name, roi_name, hemisphere)

print("\rLoading brain and ROI masks from atlas... Done!", flush=True)

#
# Load Nifti images, smooth with FWHM=6, compute % temporal change
#

print("Loading Nifti images (1/2)...", end="", flush=True)

data_info_0 = utils.normalize_nifti_image(image_path_0, fwhm=6)

print("\rLoading Nifti images (2/2)...", end="", flush=True)

data_info_1 = utils.normalize_nifti_image(image_path_1, fwhm=6)

print("\rLoading Nifti images... Done!", flush=True)

#
# Concatenate data from the two scans along the temporal axis
#

print("Concatenating Nifti images...", end="", flush=True)

brain_mask, roi_mask, data = utils.concatenate_data(brain_mask, roi_mask,
                                                    *data_info_0, *data_info_1)

# Dereference unnecessary data
del data_info_0, data_info_1

print("\rConcatenating Nifti images... Done!", flush=True)

print('Number brain voxels = {0}'.format(numpy.sum(brain_mask)),
      flush=True)
print('Number ROI voxels = {0}'.format(numpy.sum(roi_mask)),
      flush=True)

os.makedirs(out_path, exist_ok=True)
numpy.save(out_path + '/roi_mask.npy', roi_mask)
numpy.save(out_path + '/brain_mask.npy', brain_mask)

#
# Compute Haak mapping
#
haak_mapping = haak.Haak(num_lower_dim=num_lower_dim,
                         num_processes=num_processes,
                         manifold_learning=manifold_learning,
                         manifold_components=manifold_components,
                         out_path=out_path,
                         verbose=1)

eta2_coef, embedding, connectopic_map, connectopic_var = haak_mapping.fit_transform(data, roi_mask)

#
# Visualize connectopic mapping
#

i_plot = 1

for config_figures in config['figures']:

    slice_indexes = [config_figures['axis_x'],
                     config_figures['axis_y'],
                     config_figures['axis_z']]

    if hemisphere == 'RH':
        slice_indexes[0] = brain_mask.shape[0] - slice_indexes[0]

    #
    # Display connectopy
    #
    fig = pyplot.figure(i_plot, tight_layout=True)
    utils.visualize_volume(connectopic_map, brain_mask, roi_mask, slice_indexes,
                           low_percentile=5, high_percentile=95,
                           num_fig=fig,
                           margin=2,
                           legend_location=config_figures['legend_location'])

    i_plot += 1

pyplot.show()
