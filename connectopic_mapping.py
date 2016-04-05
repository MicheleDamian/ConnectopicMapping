import os
import nibabel
import numpy
import multiprocessing
from nilearn import datasets
from sklearn import manifold
from scipy.sparse import csgraph

from matplotlib import pyplot


def compute_similarity_map(fingerprints, idx_chunk=None):
    """
    Compute the eta2 coefficients from the voxels' fingerprints as described
    in "Defining functional areas in individual human brains using resting
    functional connectivity MRI". The result is a symmetric square matrix
    where each element is a real value in the range 0.0 to 1.0 with 0
    indicating the pair of voxels is completely dissimilar and 1 equals.

    Parameters
    ----------
    fingerprints : numpy.ndarray, shape (n_voxels_in_roi, n_voxels_svd)
        Boolean 3-dimensional numpy array with the same dimensions of a
        single frame of the nifti image provided as input. Voxels
        marked as True are part of the same region of interest (ROI).

    idx_chunk : numpy.ndarray, shape (K, ), (default: None)
        Indexes of the in-ROI voxels to consider when computing the eta2
        coefficients. If None all n_voxels_in_roi are considered.
    """

    num_voxels_in_chunk = idx_chunk.shape[0]
    num_voxels_in_roi = fingerprints.shape[0]
    num_voxels_out_roi = fingerprints.shape[1]

    if idx_chunk is None:
        idx_chunk = numpy.arange(0, num_voxels_in_roi)

    # Similarity maps
    def sum_coeff(fp_0, fp_1, fp_mean):
        return numpy.sum((fp_0 - fp_mean)**2 + (fp_1 - fp_mean)**2, axis=1)

    progress_old = -1

    eta2_coef = numpy.zeros((num_voxels_in_chunk, num_voxels_in_roi))

    for i in range(num_voxels_in_chunk):

        i_chunk = idx_chunk[i]
        fingerprints_chunk = fingerprints[i_chunk:, :]
        num_voxels_from_chunk = num_voxels_in_roi - i_chunk

        fp_i = fingerprints[i_chunk, :]
        fp_i_mat = numpy.repeat(fp_i[numpy.newaxis, :], num_voxels_from_chunk, axis=0)

        fp_means = (fp_i_mat + fingerprints_chunk) / 2

        fp_means_row = numpy.mean(fp_means, axis=1)
        fp_means_row_mat = numpy.repeat(fp_means_row[:, numpy.newaxis], num_voxels_out_roi, axis=1)

        eta2_num = sum_coeff(fp_i_mat, fingerprints_chunk, fp_means)
        eta2_den = sum_coeff(fp_i_mat, fingerprints_chunk, fp_means_row_mat)

        eta2_coef[i, i_chunk:] = 1 - eta2_num / eta2_den

        progress_new = int(100 * i / num_voxels_in_chunk)

        if progress_new > progress_old:
            print('\rComputing similarity maps... {0}%'.format(progress_new),
                  end="", flush=True)
            progress_old = progress_new

    return eta2_coef


def haak_mapping(nifti_image, roi_mask, brain_mask=None):
    """
    Fully data-driven methods for mapping connectopies using
    functional magnetic resonance imaging (fMRI) data acquired at
    rest by combining spectral embedding of voxel-wise connectivity
    ‘fingerprints’ with a novel approach to spatial statistical
    inference.

    "Connectopic mapping with resting-state fMRI", Haak 2016

    Parameters
    ----------
    nifti_image : nibabel
        4D nifti image on which to compute the connectopies.

    roi_mask : numpy.ndarray,
               shape (n_voxels_x_axis, n_voxels_y_axis, n_voxels_z_axis)
        Boolean 3-dimensional numpy array with the same dimensions of a
        single frame of the nifti image provided as input. Voxels
        marked as True are part of the same region of interest (ROI).

    brain_mask : numpy.ndarray, shape (n_voxels_x_axis,
        n_voxels_y_axis, n_voxels_z_axis), (default: None)
        Boolean 3-dimensional numpy array with the same dimensions of a
        single frame of the nifti image provided as input. Voxels
        marked as False are considered not part of the brain and are
        not included in the analysis. If None the entire nifti image
        is considered for the analysis.

    """

    num_cpu = multiprocessing.cpu_count()

    nifti_data = nifti_image.get_data()

    assert(nifti_data.shape[:-1] == roi_mask.shape)
    assert(roi_mask.shape == brain_mask.shape)

    # Build a all-ones mask if brain_mask is None
    if brain_mask is None:
        brain_mask = numpy.ones(nifti_data.shape[:-1], dtype=bool)

    # Get (time, num_voxels)-shaped array and voxels' indexes in
    # the original Nifti image from the image and a mask
    def extract_data(nifti_image, mask):
        assert(len(nifti_image.shape) == 4)
        assert(len(mask.shape) == 3)
        idx = numpy.asarray(numpy.where(mask))
        data_roi = nifti_image[mask, :].T
        return data_roi, idx

    print("Extracting time-series from Nifti image...",
          end="", flush=True)

    # Voxels/Features in column order
    data_in_roi, idx_in_roi = extract_data(nifti_data, roi_mask * brain_mask)

    data_out_roi, idx_out_roi = extract_data(nifti_data, ~roi_mask * brain_mask)

    print("\rExtracting time-series from Nifti image... Done!",
          flush=True)

    zeros_in_roi = numpy.where(numpy.all(data_in_roi == 0, axis=0))[0]
    zeros_out_roi = numpy.where(numpy.all(data_out_roi == 0, axis=0))[0]

    print('Found {0} null time-series in ROI mask'.format(len(zeros_in_roi)),
          flush=True)
    print('Found {0} null time-series in brain mask'.format(len(zeros_out_roi)),
          flush=True)

    print("Removing null time-series from data...",
          end="", flush=True)

    idx_zeros_in_roi = idx_in_roi[:, zeros_in_roi]
    roi_mask[tuple(idx_zeros_in_roi)] = False
    idx_zeros_out_roi = idx_out_roi[:, zeros_out_roi]
    brain_mask[tuple(idx_zeros_out_roi)] = False

    data_in_roi = numpy.delete(data_in_roi, zeros_in_roi, axis=1)
    idx_in_roi = numpy.delete(idx_in_roi, zeros_in_roi, axis=1)

    data_out_roi = numpy.delete(data_out_roi, zeros_out_roi, axis=1)
    idx_out_roi = numpy.delete(idx_out_roi, zeros_out_roi, axis=1)

    numpy.save('data_in_roi.npy', data_in_roi)
    numpy.save('data_out_roi.npy', data_out_roi)

    print("\rRemoving null time-series from data... Done!",
          flush=True)

    print('Number brain voxels = {0}'.format(numpy.sum(brain_mask)),
          flush=True)
    print('Number ROI voxels = {0}'.format(numpy.sum(roi_mask)),
          flush=True)

    print("Dimensionality reduction...",
          end="", flush=True)

    ###
    # Get first t-1 components (where t is the number of time frames)
    ###

    num_voxels_svd = data_out_roi.shape[0] - 1

    num_voxels_in_roi = data_in_roi.shape[1]
    num_voxels_out_roi = data_out_roi.shape[1]

    filename_data_s = 'data_s.npy'
    filename_data_v = 'data_v.npy'

    if os.path.isfile(filename_data_v):
        data_v = numpy.load(filename_data_v)
        data_s = numpy.load(filename_data_s)
    else:

        # Demean data
        data_out_roi_mean = numpy.mean(data_out_roi, axis=0)
        data_out_roi_demean = data_out_roi - data_out_roi_mean

        data_in_roi_mean = numpy.mean(data_in_roi, axis=0)
        data_in_roi_demean = data_in_roi - data_in_roi_mean

        data = numpy.hstack((data_in_roi_demean, data_out_roi_demean))

        data_u, data_s, data_v = numpy.linalg.svd(data, full_matrices=False)

        # Store data
        numpy.save(filename_data_s, data_s)
        numpy.save(filename_data_v, data_v)

    print("\rDimensionality reduction... Done!",
          flush=True)

    # TEST
    numpy.save('data_in_roi.npy', data_in_roi)
    numpy.save('data_out_roi.npy', data_out_roi)

    ###
    # Compute voxels fingerprints as Pearson correlation between ROI
    # time-series and out-of-ROI reprojected time-series
    ###

    print("Computing fingerprints...",
          end="", flush=True)

    filename_fingerprints = 'fingerprints.npy'

    if os.path.isfile(filename_fingerprints):
        fingerprints = numpy.load(filename_fingerprints)
    else:

        data_s_diag = numpy.diag(data_s[:num_voxels_svd])
        data_reprojected = numpy.dot(data_s_diag,
                                     data_v[:num_voxels_svd, :num_voxels_in_roi])
        fingerprints = data_reprojected.T

        numpy.save(filename_fingerprints, fingerprints)

    print("\rComputing fingerprints... Done!",
          flush=True)

    print("Computing similarity maps...",
          end="", flush=True)

    filename_eta2_coef = 'eta2_coef.npy'

    if os.path.isfile(filename_eta2_coef):
        eta2_coef = numpy.load(filename_eta2_coef)
    else:
        # Distribute load equally among all CPUs
        pool = multiprocessing.Pool(num_cpu)

        # Approximation of the optimal fraction of the dataset to
        # allocate to each CPU
        frac_fingerprint = (numpy.arange(num_cpu+1)) * 2 / (num_cpu * (num_cpu + 1))
        idxs_fingerprint = numpy.cumsum(frac_fingerprint) * num_voxels_in_roi
        idxs_pool = [numpy.arange(int(idxs_fingerprint[idx]),
                                  int(idxs_fingerprint[idx+1]))
                     for idx in range(len(idxs_fingerprint)-1)]

        starmap_input = [(fingerprints, idx) for idx in idxs_pool]

        # Run compute_similarity_map in parallel
        eta2_chunks = pool.starmap(compute_similarity_map, starmap_input)

        # Merge together results
        eta2_coef = numpy.zeros((num_voxels_in_roi, num_voxels_in_roi))
        for i_cpu in range(num_cpu):
            for i_chunk in range(len(idxs_pool[i_cpu])):
                i_eta = idxs_pool[i_cpu][i_chunk]
                eta2_row = eta2_chunks[i_cpu][i_chunk, i_eta:]
                eta2_coef[i_eta, i_eta:] = eta2_row
                eta2_coef[i_eta:, i_eta] = eta2_row

        numpy.save(filename_eta2_coef, eta2_coef)

    print("\rComputing similarity maps... Done!",
          flush=True)

    ###
    # Similarity of connectivity between pairs of voxels
    ###

    print("Computing L2 distance between voxels...",
          end="", flush=True)

    filename_similarity_distance = 'similarity_distance.npy'

    if os.path.isfile(filename_similarity_distance):
        similarity_distance = numpy.load(filename_similarity_distance)
    else:

        progress_old = -1

        similarity_distance = numpy.zeros((num_voxels_in_roi, num_voxels_in_roi))

        for i in range(num_voxels_in_roi):
            for j in range(i+1, num_voxels_in_roi):
                similarity_distance[i, j] = similarity_distance[j, i] = \
                    numpy.linalg.norm(eta2_coef[i, :] - eta2_coef[j, :], ord=2)

                progress_new = int(100 * i / num_voxels_in_roi)

                if progress_new > progress_old:
                    print("\rComputing L2 distance between voxels... {0}%".format(progress_new),
                          end="", flush=True)
                    progress_old = progress_new

        numpy.save(filename_similarity_distance, similarity_distance)

    print("\rComputing L2 distance between voxels... Done!", flush=True)

    ###
    # Binary search a similarity threshold, that is the minimum value
    # of the L2-norm between pair of voxels' similarities required
    # such that all the voxels in the ROI are connected.
    ###
    print("Searching minimum threshold value for connected graph...",
          end="", flush=True)

    filename_adjacency = 'adjacency.npy'

    if os.path.isfile(filename_adjacency):
        adjacency = numpy.load(filename_adjacency)
    else:

        similarity_values = numpy.sort(similarity_distance, axis=None)
        high_index = num_voxels_in_roi**2 - 1
        low_index = 0
        min_threshold = similarity_values[-1]

        while True:

            similarity_index = int((high_index + low_index) / 2)
            similarity_threshold = similarity_values[similarity_index]

            # Transform similarity matrix into a connected graph
            adjacency = similarity_distance < similarity_threshold

            # Find connected components
            num_components, _ = csgraph.connected_components(adjacency,
                                                             directed=False)

            if num_components > 1:
                low_index = similarity_index + 1
            else:
                high_index = similarity_index - 1
                min_threshold = min(min_threshold, similarity_threshold)

            if high_index < low_index:
                break

        adjacency = similarity_distance < min_threshold

        numpy.save(filename_adjacency, adjacency)

    print("\rSearching minimum threshold value for connected graph... Done!",
          flush=True)

    # Disconnect distant voxels
    eta2_coef[~adjacency] = 0

    ###
    # Learn the manifold for this data and reproject the similarity
    # graph on the two most significant connectopies
    ###

    print("tSNE embedding...",
          end="", flush=True)

    filename_connectopic_map = 'connectopic_map.npy'

    if os.path.isfile(filename_connectopic_map):
        connectopic_map = numpy.load(filename_connectopic_map)
    else:

        manifold_tsne = manifold.TSNE(n_components=2,
                                      perplexity=30.0,
                                      early_exaggeration=4.0,
                                      learning_rate=1000.0,
                                      n_iter=1000,
                                      n_iter_without_progress=30,
                                      min_grad_norm=1e-07,
                                      metric='euclidean',
                                      init='random',
                                      verbose=0,
                                      random_state=None,
                                      method='barnes_hut',
                                      angle=0.5)

        connectopic_map = manifold_tsne.fit_transform(eta2_coef)

        numpy.save(filename_connectopic_map, connectopic_map)

    print("\rtSNE embedding... Done!",
          end="", flush=True)

    return connectopic_map, roi_mask


""" Run pipeline
"""
if __name__ == "__main__":

    image_path = '/Users/michele/Development/Workspaces/UpWork/' \
                 'Morgan_Hough/Resources/rfMRI/100307/MNINonLinear/' \
                 'Results/rfMRI_REST1_LR/' \
                 'rfMRI_REST1_LR_hp2000_clean.nii.gz'

    print("Loading Nifti image...")

    # Load Nifti image
    nifti_image = nibabel.load(image_path)

    print("Loading ROI from atlas...")

    # Load M1 region
    dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    harvard_oxford_labels = dataset.labels
    harvard_oxford_maps = nibabel.load(dataset.maps)
    harvard_oxford_data = harvard_oxford_maps.get_data()

    m1_indexes = numpy.where((dataset.labels == "Precentral Gyrus"))[0]

    # Build mask from ROI
    roi_mask = numpy.zeros(harvard_oxford_data.shape, dtype=bool)
    for index in m1_indexes:
        roi_mask[harvard_oxford_data == index] = True

    # Keep just left part
    roi_mask_width = roi_mask.shape[0]
    roi_mask[int(roi_mask_width/2):, :, :] = False

    print("Loading brain mask...")

    # Load brain mask
    brain_mask = numpy.zeros(harvard_oxford_data.shape, dtype=bool)
    brain_mask[numpy.nonzero(harvard_oxford_data)] = True

    # Display brain mask and ROI mask
    pyplot.figure(1)
    x_index = 21
    coords_brain = numpy.where(brain_mask[x_index, :, :])
    pyplot.scatter(coords_brain[0], coords_brain[1], c='b')
    pyplot.hold(True)
    coords_mask = numpy.where(roi_mask[x_index, :, :])
    pyplot.scatter(coords_mask[0], coords_mask[1], c='r')
    pyplot.title("Brain and ROI masks")
    pyplot.legend(("Brain mask", "ROI mask"))

    # Compute Haak mapping
    embedding, roi_mask = haak_mapping(nifti_image, roi_mask, brain_mask)

    numpy.save('embedding.npy', embedding)
    numpy.save('roi_mask.npy', roi_mask)

    # Display connectopy 0 dimension
    pyplot.figure(2)
    x_index = 21
    z_index = 30

    coords_brain = numpy.where(brain_mask[:, :, z_index])
    pyplot.scatter(coords_brain[0], coords_brain[1], c='w')

    pyplot.hold(True)

    coords_mask = numpy.where(roi_mask[:, :, z_index])
    cmap = pyplot.get_cmap('jet')
    min_mask = numpy.min(embedding[:, 0])
    max_mask = numpy.max(embedding[:, 0])
    idx_cmap = (embedding[:, 0] - min_mask) / (max_mask - min_mask)
    clrs_mask = cmap(idx_cmap)
    pyplot.scatter(coords_mask[0], coords_mask[1], c=clrs_mask)

    pyplot.title("ROI connectopies")
    pyplot.legend(("Brain mask", "ROI connectopies"))
    pyplot.show()
