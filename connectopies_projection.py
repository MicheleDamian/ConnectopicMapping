import cutils
import json
import numpy
import seaborn
import multiprocessing
import os
from connectopic_mapping import utils
from matplotlib import pyplot

with open('config.json') as config_file:
    config = json.load(config_file)

subject = config["subject"]
session = config["session"]
scans = config["scans"]
hemisphere = config["hemisphere"]
atlas_name = config["atlas_name"]
roi_name = config["roi_name"]

in_path_LR = config["out_path"] + \
           '/../../../Resources/rfMRI/{0}_{1}/rfMRI_{1}_{2}_hp2000_clean.nii.gz'.format(subject, session, scans[0])

in_path_RL = config["out_path"] + \
           '/../../../Resources/rfMRI/{0}_{1}/rfMRI_{1}_{2}_hp2000_clean.nii.gz'.format(subject, session, scans[1])

out_path = config["out_path"] + \
           '/rfMRI_{0}_{1}_{2}'.format(subject, session, hemisphere)

brain_mask = numpy.load(out_path + "/brain_mask.npy")
roi_mask = numpy.load(out_path + "/roi_mask.npy")

filename_data = out_path + "/data.npy"
filename_brain_mask = out_path + "/brain_mask_{0}.npy".format(hemisphere)
filename_roi_mask = out_path + "/roi_mask_{0}.npy".format(hemisphere)

if os.path.isfile(filename_data):
    data = numpy.load(filename_data)
    brain_mask = numpy.load(filename_brain_mask)
    roi_mask = numpy.load(filename_roi_mask)

else:
    print("Loading Nifti images (1/2)...", end="", flush=True)

    data_info_0 = utils.normalize_nifti_image(in_path_LR, fwhm=6)

    print("\rLoading Nifti images (2/2)...", end="", flush=True)

    data_info_1 = utils.normalize_nifti_image(in_path_RL, fwhm=6)

    print("\rLoading Nifti images... Done!", flush=True)

    half = int(brain_mask.shape[0] / 2) + 1

    print(numpy.sum(brain_mask), numpy.sum(roi_mask))

    if hemisphere == "LH":
        brain_mask[:half, :, :] = False
        brain_mask += roi_mask
    else:
        brain_mask[half:, :, :] = False
        brain_mask += roi_mask

    print(numpy.sum(brain_mask), numpy.sum(roi_mask))

    #
    # Concatenate data from the two scans along the temporal axis
    #

    print("Concatenating Nifti images...", end="", flush=True)

    brain_mask, roi_mask, data = utils.concatenate_data(brain_mask, roi_mask,
                                                        *data_info_0, *data_info_1)

    print(numpy.sum(brain_mask), numpy.sum(roi_mask))

    # Dereference unnecessary data
    del data_info_0, data_info_1

    numpy.save(filename_data, data)
    numpy.save(filename_brain_mask, brain_mask)
    numpy.save(filename_roi_mask, roi_mask)

    print("\rConcatenating Nifti images... Done!", flush=True)

#
# Compute correlations
#

coord_in_roi = numpy.where(roi_mask)
coord_out_roi = numpy.where(brain_mask ^ roi_mask)

assert((brain_mask * roi_mask == roi_mask).all())

num_voxels_in_roi = coord_in_roi[0].shape[0]
num_voxels_out_roi = coord_out_roi[0].shape[0]

print(numpy.sum(brain_mask), num_voxels_in_roi, num_voxels_out_roi, data.shape)
assert(data.shape[1] == num_voxels_out_roi + num_voxels_in_roi)

filename_pearson_coeff = out_path + "/pearson_coeff.npy"

if os.path.isfile(filename_pearson_coeff):
    pearson_coeff = numpy.load(filename_pearson_coeff)

else:

    print("Computing correlations...", end="", flush=True)

    # Distribute load equally among all CPUs
    num_cpu = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cpu)

    # Approximation of the optimal fraction of the dataset to
    # allocate to each CPU
    idxs_pool = [numpy.arange(int(idx/num_cpu * num_voxels_out_roi),
                              int((idx+1)/num_cpu * num_voxels_out_roi))
                 for idx in range(num_cpu)]

    starmap_input = [(data[:, :num_voxels_in_roi], data[:, num_voxels_in_roi+idx]) for idx in idxs_pool]

    # Run in parallel
    corr_chunks = pool.starmap(cutils.compute_pearson, starmap_input)

    # Merge together results
    pearson_coeff = numpy.zeros((num_voxels_in_roi, num_voxels_out_roi), dtype=numpy.float32)
    for i_cpu in range(num_cpu):
        pearson_coeff[:, idxs_pool[i_cpu]] = corr_chunks[i_cpu]

    numpy.save(filename_pearson_coeff, pearson_coeff)

#
pearson_coeff -= numpy.repeat(numpy.mean(pearson_coeff, axis=1)[:, numpy.newaxis], num_voxels_out_roi, axis=1)
pearson_std = numpy.std(pearson_coeff, axis=1)
pearson_coeff_norm = pearson_coeff / numpy.repeat(pearson_std[:, numpy.newaxis], num_voxels_out_roi, axis=1)

pearson_argmax = numpy.argmax(pearson_coeff, axis=0)
pearson_max = numpy.max(pearson_coeff, axis=0)
idx_pearson_coeff_norm = [pearson_argmax[i] for i in range(num_voxels_out_roi)]
pearson_max_norm = pearson_coeff_norm[(idx_pearson_coeff_norm, list(range(num_voxels_out_roi)))]

numpy.save(out_path + "/pearson_coeff_norm.npy", pearson_coeff_norm)
numpy.save(out_path + "/pearson_max_norm.npy", pearson_max_norm)
numpy.save(out_path + "/pearson_max.npy", pearson_max)

connectopic_map = numpy.load(out_path + "/connectopic_map_{0}.npy".format(config["manifold_learning"]))[:, 0]

pearson_mask = numpy.zeros(brain_mask.shape)

i_x = coord_out_roi[0]
i_y = coord_out_roi[1]
i_z = coord_out_roi[2]
pearson_mask[i_x, i_y, i_z] = True
pearson_map = connectopic_map[pearson_argmax]

# Remove low correlations
coord_low = numpy.where((pearson_max < 0.0) + (pearson_max_norm < 2))
pearson_map = numpy.delete(pearson_map, coord_low)
pearson_mask[i_x[coord_low], i_y[coord_low], i_z[coord_low]] = False

slice_indexes = [66, 55, 69]

fig = pyplot.figure(0, tight_layout=True)
utils.visualize_volume(pearson_map, brain_mask, pearson_mask, slice_indexes,
                       low_percentile=5, high_percentile=95,
                       num_fig=fig,
                       margin=2,
                       legend_location=1)

pyplot.show()
