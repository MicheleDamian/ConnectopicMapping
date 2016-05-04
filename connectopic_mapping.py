import os
import nibabel
import numpy
import multiprocessing
import GPy
from mpl_toolkits.axes_grid1 import Grid
from nilearn import datasets
from sklearn import manifold
from scipy.sparse import csgraph
from matplotlib import pyplot


def spatial_statistic(x, y, x_predictions):
    """ Approximate each connectopic map using a spatial model. The
        input is smoothed by a zero-mean Gaussian process where the
        kernel is given by a Marten covariance function added to Gaussian
        noise. This method estimate the parameters and return the values
        predicted at the locations in x_predictions.


    Parameters
    ----------
    x : numpy.ndarray, (n_voxels, 3)
        3D coordinates of the center of each voxel in the ROI.

    y : numpy.ndarray, (n_voxels, n_dim)
        Values of the voxels centered in x. Each dimension of y is
        estimated independently.

    x_predictions : numpy.ndarray, (n_predictions, 3)
        3D coordinates where to predict the function.

    Returns
    -------
    y_means : numpy.ndarray, (n_predictions, n_dim)
        A posterior mean from the Gaussian process for each dimension
        given as input.

    y_vars : numpy.ndarray, (n_predictions, )
        Variance at each x_predictions point.

    """

    # Matern kernel with v=5/2, the GPRegression model includes a
    # Gaussian noise by default.
    kernel = GPy.kern.Matern52(input_dim=3, variance=1., lengthscale=1.)

    # Estimate the model
    model = GPy.models.GPRegression(x, y, kernel)
    model.optimize(messages=True)

    print(model)

    y_means, y_vars = model.predict(x_predictions)

    return y_means, y_vars


def compute_fingerprints(data_in_roi, data, idxs_chunk=None):
    """ Compute fingerprints for each voxel time-series in the ROI
        and each connectopy in the reprojecetd out-of-ROI voxels.

    Parameters
    ----------
    data_in_roi: numpy.ndarray (n_timeseries, n_variables)
        Voxels which fingerprints have to be computed. Each voxel's
        time-serie is represented by a column in data_in_roi.

    data: numpy.ndarray (n_timeseries, n_svd_variables)
        Voxels out of the ROI used for computing the fingerprints.

    idxs_chunk: numpy.ndarray, default None
        Indexes of the voxel in the data_in_roi array at which
        to compute the fingerprints. If None it takes all
        voxels in data_in_roi.

    Returns
    -------
    fp_chunks: numpy.ndarray
        Computed fingerprints for this chunk of data.

    """

    if idxs_chunk is None:
        idxs_chunk = numpy.arange(data_in_roi.shape[1])

    num_voxels_chunk = len(idxs_chunk)
    fingerprints = numpy.zeros((num_voxels_chunk, data.shape[1]))

    progress_old = -1

    # Compute out of ROI data variance
    data_mus = numpy.mean(data, axis=0)
    data_demean = data - data_mus[numpy.newaxis, :]
    data_var = numpy.mean(data_demean * data_demean, axis=0)

    # Compute in ROI data variance
    data_in_roi_mus = numpy.mean(data_in_roi[:, idxs_chunk], axis=0)
    data_in_roi_demean = data_in_roi[:, idxs_chunk] - data_in_roi_mus[numpy.newaxis, :]
    data_in_roi_var = numpy.mean(data_in_roi_demean * data_in_roi_demean, axis=0)

    for i_chunk in range(num_voxels_chunk):

        # Compute covariance between one in ROI time-series
        # and out of ROI time-series
        data_voxel_demean = data_in_roi_demean[:, i_chunk]
        data_voxel_demean = numpy.repeat(data_voxel_demean[:, numpy.newaxis],
                                         data_demean.shape[1], axis=1)
        data_cov = numpy.mean(data_voxel_demean * data_demean, axis=0)

        data_data_in_roi_cov = numpy.sqrt(data_in_roi_var[i_chunk] * data_var)

        fingerprints[i_chunk, :] = data_cov / data_data_in_roi_cov

        progress = int(i_chunk * 100 / num_voxels_chunk)

        if progress > progress_old:
            print("\rComputing fingerprints... {0}%".format(progress),
                  end="", flush=True)
            progress_old = progress

    return fingerprints


def compute_similarity_map(idx_chunk, fingerprints):

    # Similarity maps
    def sum_coeff(fp_0, fp_1, fp_mean):
        return numpy.sum((fp_0 - fp_mean)**2 + (fp_1 - fp_mean)**2, axis=1)

    progress_old = -1

    num_voxels_in_chunk = idx_chunk.shape[0]
    num_voxels_in_roi = fingerprints.shape[0]
    num_voxels_out_roi = fingerprints.shape[1]

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


def haak_mapping(nifti_image, roi_mask, brain_mask=None, out_path='.'):
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

    os.makedirs(out_path, exist_ok=True)

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

    print("\rRemoving null time-series from data... Done!",
          flush=True)

    print('Number brain voxels = {0}'.format(numpy.sum(brain_mask)),
          flush=True)
    print('Number ROI voxels = {0}'.format(numpy.sum(roi_mask)),
          flush=True)

    print("Dimensionality reduction...",
          end="", flush=True)

    # Get first t-1 principal components (where t is the number of
    # time frames)
    num_features = data_out_roi.shape[0] - 1

    filename_data_svd = out_path + os.sep + 'data_svd.npy'
    if os.path.isfile(filename_data_svd):
        data_svd = numpy.load(filename_data_svd)
    else:

        # Demean data
        data_out_roi_mean = numpy.mean(data_out_roi, axis=0)
        data_out_roi_demean = data_out_roi - data_out_roi_mean

        u, s, v = numpy.linalg.svd(data_out_roi_demean, full_matrices=False)
        #data_svd = numpy.dot(u[:, :num_features], numpy.diag(s[:num_features]))
        data_svd = u[:, :num_features]
        numpy.save(filename_data_svd, data_svd)

    print("\rDimensionality reduction... Done!",
          flush=True)

    # Compute voxels fingerprints as Pearson correlation between ROI
    # time-series and out-of-ROI reprojected time-series
    num_voxels_in_roi = data_in_roi.shape[1]
    num_voxels_out_roi = data_out_roi.shape[1]
    num_voxels_svd = data_svd.shape[1]

    print("Computing fingerprints...",
          end="", flush=True)

    filename_fingerprints = out_path + os.sep + 'fingerprints.npy'
    if os.path.isfile(filename_fingerprints):
        fingerprints = numpy.load(filename_fingerprints)
    else:
        # Distribute load equally among all CPUs
        pool = multiprocessing.Pool(num_cpu)

        # Approximation of the optimal fraction of the dataset to
        # allocate to each CPU
        idxs_pool = [numpy.arange(int(idx/num_cpu * num_voxels_in_roi),
                                  int((idx+1)/num_cpu * num_voxels_in_roi))
                     for idx in range(num_cpu)]

        starmap_input = [(data_in_roi, data_svd, idx) for idx in idxs_pool]

        # Run compute_similarity_map in parallel
        fingerprints_chunks = pool.starmap(compute_fingerprints, starmap_input)

        # Merge together results
        fingerprints = numpy.zeros((num_voxels_in_roi, num_voxels_svd))
        for i_cpu in range(num_cpu):
            fingerprints[idxs_pool[i_cpu], :] = fingerprints_chunks[i_cpu]

        numpy.save(filename_fingerprints, fingerprints)

    print("\rComputing fingerprints... Done!",
          flush=True)

    print("Computing similarity maps...",
          end="", flush=True)

    filename_eta2_coef = out_path + os.sep + 'eta2_coef.npy'
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

        starmap_input = [(idx, fingerprints) for idx in idxs_pool]

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

    # Similarity of connectivity between pairs of voxels
    print("Computing L2 distance between voxels...",
          end="", flush=True)

    filename_similarity_distance = out_path + os.sep + 'similarity_distance.npy'
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

    print("Searching minimum threshold value for connected graph...",
          end="", flush=True)

    filename_adjacency = out_path + os.sep + 'adjacency.npy'
    if os.path.isfile(filename_adjacency):
        adjacency = numpy.load(filename_adjacency)
    else:
        # Binary search a similarity threshold, that is the minimum value
        # of the L2-norm between pair of voxels' similarities required
        # such that all the voxels in the ROI are connected.

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

    print("Spectral embedding...",
          end="", flush=True)

    # Learn the manifold for this data and reproject the similarity
    # graph on the two most significant connectopies
    embedding = manifold.spectral_embedding(eta2_coef, n_components=2, norm_laplacian=False)

    print("\rSpectral embedding... Done!", flush=True)

    print("Spatial statistics...",
          end="", flush=True)

    filename_connectopic_map = out_path + os.sep + 'connectopic_map.npy'
    if os.path.isfile(filename_connectopic_map):
        connectopic_map = numpy.load(filename_connectopic_map)
    else:

        # Run gaussian process on embedding to compute spatial statistics
        coord_x, coord_y, coord_z = numpy.where(roi_mask)
        coords = numpy.array([[x, y, z] for x, y, z in zip(coord_x, coord_y, coord_z)])
        connectopic_map, connectopic_var = spatial_statistic(coords, embedding, coords)
        numpy.save(filename_connectopic_map, connectopic_map)

    print("Spatial statistics... Done!", flush=True)

    return embedding, connectopic_map, roi_mask


def visualize_volume(data, fig, title, cmap, slice_indexes, brain_mask, roi_mask):

    # Get voxels color from embedding
    min_val = numpy.min(data, axis=0)
    max_val = numpy.max(data, axis=0)
    data_norm = (data - min_val) / (max_val - min_val)
    fun_cmap = pyplot.get_cmap(cmap)
    clr_rgb = fun_cmap(data_norm.flatten())

    def get_slice_coords(mask, plane):
        coords_mask = numpy.where(mask)
        return coords_mask, numpy.where(coords_mask[plane[3]] == plane[2])[0]

    #
    # Display embedding
    #
    axes = Grid(fig, rect=111, nrows_ncols=(2,2), label_mode='L')

    # Display in X, Y and Z subplot
    for i in range(3):

        dims = numpy.delete(numpy.arange(3), slice_indexes[i][3])

        # Plot brain
        coords_mask, coords_idx = get_slice_coords(brain_mask, slice_indexes[i])
        coords_brain_x = coords_mask[dims[0]][coords_idx]
        coords_brain_y = coords_mask[dims[1]][coords_idx]
        axes[i].scatter(coords_brain_x, coords_brain_y, c='k', s=15, edgecolors='face')

        axes[i].hold(True)

        # Plot ROI
        coords_mask, coords_idx = get_slice_coords(roi_mask, slice_indexes[i])
        coords_roi_x = coords_mask[dims[0]][coords_idx]
        coords_roi_y = coords_mask[dims[1]][coords_idx]
        axes[i].scatter(coords_roi_x, coords_roi_y, c=clr_rgb[coords_idx, :], s=15, edgecolors='face')

        axes[i].set_title("{0} at slice {2}={3}".format(title, *slice_indexes[i]))
        axes[i].legend(("Cortex", "ROI"), loc=2)
        axes[i].grid(True)

    # Set axis limits
    coords_3d = numpy.where(roi_mask)
    coords_x = numpy.sort(coords_3d[0])
    coords_y = numpy.sort(coords_3d[1])
    coords_z = numpy.sort(coords_3d[2])
    axes[0].set_xlim([coords_x[0] - 5, coords_x[-1] + 5])
    axes[0].set_ylim([coords_z[0] - 5, coords_z[-1] + 5])
    axes[1].set_xlim([coords_y[0] - 5, coords_y[-1] + 5])
    axes[2].set_ylim([coords_y[0] - 5, coords_y[-1] + 5])

    # Remove shared borders
    axes[0].spines['bottom'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[3].spines['top'].set_visible(False)
    axes[3].spines['left'].set_visible(False)

    # Name axes
    axes[0].set_ylabel("Z axis")
    axes[2].set_xlabel("X axis")
    axes[2].set_ylabel("Y axis")
    axes[3].set_xlabel("Y axis")

    # Backgroud color
    bg_color = [0.9, 0.9, 0.9]
    for i in range(4):
        axes[i].patch.set_facecolor(bg_color)

    # Add colorbar
    sm = pyplot.cm.ScalarMappable(cmap=cmap, norm=pyplot.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbaxes = fig.add_axes([0.95, 0.1, 0.01, 0.4])
    cbar = pyplot.colorbar(sm, cax=cbaxes)
    cbar.set_ticklabels([])


""" Run pipeline
"""
if __name__ == "__main__":

    image_path = '/Users/michele/Development/Workspaces/UpWork/' \
                 'Morgan_Hough/Resources/rfMRI/103414_REST2/MNINonLinear/' \
                 'Results/rfMRI_REST2_RL/' \
                 'rfMRI_REST2_RL_hp2000_clean.nii.gz'

    out_path = '/Users/michele/Development/Workspaces/UpWork/' \
               'Morgan_Hough/Results/connectopic_mapping/master/' \
               'rfMRI_103414_REST2_RL'

    print("Loading Nifti image...")

    # Load Nifti image
    nifti_image = nibabel.load(image_path)

    print("Loading ROI from atlas...")

    # Load M1 region
    dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    harvard_oxford_labels = numpy.array(dataset.labels)
    harvard_oxford_maps = nibabel.load(dataset.maps)
    harvard_oxford_data = harvard_oxford_maps.get_data()

    m1_indexes = numpy.where((harvard_oxford_labels == "Precentral Gyrus"))

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

    # Compute Haak mapping
    embedding, connectopy, roi_mask = haak_mapping(nifti_image, roi_mask, brain_mask, out_path)

    # Slice coordinates (plane, axis, value, axis_index)
    #slice_indexes = [('X-Z', 'Y', 65, 1), ('Y-Z', 'X', 18, 0), ('X-Y', 'Z', 50, 2)]
    slice_indexes = [('X-Z', 'Y', 55, 1), ('Y-Z', 'X', 25, 0), ('X-Y', 'Z', 69, 2)]

    #
    # Display embedding
    #
    fig = pyplot.figure(1, tight_layout=True)
    visualize_volume(embedding, fig, "Voxels after Manifold Learning", 'terrain', slice_indexes, brain_mask, roi_mask)

    #
    # Display connectopy
    #
    fig = pyplot.figure(2, tight_layout=True)
    visualize_volume(connectopy, fig, "Voxels after Gaussian Processes", 'terrain', slice_indexes, brain_mask, roi_mask)

    pyplot.show()
