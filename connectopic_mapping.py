import os
import nibabel
import numpy
import matplotlib
import multiprocessing
import GPy
from nilearn import datasets, image
from scipy.sparse import csgraph
from sklearn import manifold
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import Grid


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
    #model.optimize(messages=True)

    print(model)

    y_means, y_vars = model.predict(x_predictions)

    return y_means, y_vars


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
        fp_i_mat = numpy.repeat(fp_i[numpy.newaxis, :],
                                num_voxels_from_chunk,
                                axis=0)

        fp_means = (fp_i_mat + fingerprints_chunk) / 2

        fp_means_row = numpy.mean(fp_means, axis=1)
        fp_means_row_mat = numpy.repeat(fp_means_row[:, numpy.newaxis],
                                        num_voxels_out_roi,
                                        axis=1)

        eta2_num = sum_coeff(fp_i_mat, fingerprints_chunk, fp_means)
        eta2_den = sum_coeff(fp_i_mat, fingerprints_chunk, fp_means_row_mat)

        eta2_coef[i, i_chunk:] = 1 - eta2_num / eta2_den

        progress_new = int(100 * i / num_voxels_in_chunk)

        if progress_new > progress_old:
            print('\rComputing similarity maps... {0}%'.format(progress_new),
                  end="", flush=True)
            progress_old = progress_new

    return eta2_coef


def haak_mapping(data, num_voxels_in_roi, roi_mask, manifold_learning='tSNE', out_path='.'):
    """
    Fully data-driven methods for mapping connectopies using
    functional magnetic resonance imaging (fMRI) data acquired at
    rest by combining spectral embedding of voxel-wise connectivity
    ‘fingerprints’ with a novel approach to spatial statistical
    inference.

    "Connectopic mapping with resting-state fMRI", Haak 2016

    Parameters
    ----------
    data : numpy.ndarray, shape (n_timepoints, n_voxels)
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

    manifold_learning : string, (options: 'tSNE', 'spectral'),
        (default: 'tSNE')
        Manifold learning algorithm used to find  lower dimensional
        representation of the correlation map between voxels. The options
        are tSNE or spectral embedding ()

    out_path : string, (default: '.')
        folder where to save the intermediate results. If the folder
        doesn't exist the method tries to create one. By default the
        results are saved in the same folder from where the script is
        run. They are not recomputed if they are are already present.
        If out_path=None no intermediate result is saved and everything
        is recomputed.

    """

    assert(manifold_learning in ['tSNE', 'spectral'])

    # File names of intermediate results
    if out_path is not None:
        # Create folder and subfolders if they don't exist
        os.makedirs(out_path, exist_ok=True)
        # Instantiate filename variables
        filename_data_s = out_path + os.sep + \
                          'data_s.npy'
        filename_data_v = out_path + os.sep + \
                          'data_v.npy'
        filename_fingerprints = out_path + os.sep + \
                                'fingerprints.npy'
        filename_eta2_coef = out_path + os.sep + \
                             'eta2_coef.npy'
        filename_embedding = out_path + os.sep + \
                             'embedding_{0}.npy'.format(manifold_learning)
        filename_connectopic_map = out_path + os.sep + \
                                   'connectopic_map_{0}.npy'.format(manifold_learning)

    num_cpu = multiprocessing.cpu_count()

    ###
    #
    # Get first t-1 components (where t is the number of time frames)
    #
    ###

    print("Dimensionality reduction...", end="", flush=True)

    if out_path is not None and os.path.isfile(filename_data_v):
        data_v = numpy.load(filename_data_v)
        data_s = numpy.load(filename_data_s)
    else:

        data_u, data_s, data_v = numpy.linalg.svd(data, full_matrices=False)
        del data_u

        # Store data
        if out_path is not None:
            numpy.save(filename_data_s, data_s)
            numpy.save(filename_data_v, data_v)

    print("\rDimensionality reduction... Done!", flush=True)

    ###
    #
    # Compute voxels fingerprints as Pearson correlation between ROI
    # time-series and out-of-ROI reprojected time-series
    #
    ###

    print("Computing fingerprints...", end="", flush=True)

    if out_path is not None and os.path.isfile(filename_fingerprints):
        fingerprints = numpy.load(filename_fingerprints)
    else:

        num_voxels_svd = data.shape[0] - 1
        data_s_diag = numpy.diag(data_s[:num_voxels_svd])
        data_reprojected = numpy.dot(data_s_diag,
                                     data_v[:num_voxels_svd, :num_voxels_in_roi])
        fingerprints = data_reprojected.T

        if out_path is not None:
            numpy.save(filename_fingerprints, fingerprints)

    print("\rComputing fingerprints... Done!", flush=True)

    print("Computing similarity maps...", end="", flush=True)

    if out_path is not None and os.path.isfile(filename_eta2_coef):
        eta2_coef = numpy.load(filename_eta2_coef)
    else:
        # Distribute load equally among all CPUs
        pool = multiprocessing.Pool(num_cpu)

        # Approximation of the optimal fraction of the dataset to
        # allocate to each CPU
        frac_fingerprint = (numpy.arange(num_cpu+1)) * 2 / (num_cpu * (num_cpu+1))
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

        if out_path is not None:
            numpy.save(filename_eta2_coef, eta2_coef)

    print("\rComputing similarity maps... Done!", flush=True)

    ###
    #
    # Learn the manifold for this data and reproject the similarity
    # graph on the two most significant connectopies
    #
    ###

    print("Learning embedding...", end="", flush=True)

    if out_path is not None and os.path.isfile(filename_embedding):
        embedding = numpy.load(filename_embedding)
    else:

        distances = 2 * (1 - eta2_coef)

        if manifold_learning == 'tSNE':
            manifold_class = manifold.TSNE(n_components=1,
                                          perplexity=30.0,
                                          early_exaggeration=4.0,
                                          learning_rate=1000.0,
                                          n_iter=1000,
                                          n_iter_without_progress=30,
                                          min_grad_norm=1e-07,
                                          metric='precomputed',
                                          init='random',
                                          verbose=1,
                                          random_state=None,
                                          method='exact')

        elif manifold_learning == 'spectral':

            print("Searching connected graph's min threshold value...",
                  end="", flush=True)

            # Binary search a similarity threshold, that is the minimum value
            # of the L2-norm between pair of voxels' similarities required
            # such that all the voxels in the ROI are connected.

            similarity_values = numpy.sort(distances, axis=None)
            high_index = num_voxels_in_roi**2 - 1
            low_index = 0
            min_threshold = similarity_values[-1]

            while True:

                similarity_index = int((high_index + low_index) / 2)
                similarity_threshold = similarity_values[similarity_index]

                # Transform similarity matrix into a connected graph
                adjacency = distances < similarity_threshold

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

            adjacency = distances < min_threshold

            print("\rSearching connected graph's min threshold value... Done!",
                  flush=True)

            # Disconnect distant voxels
            eta2_coef[~adjacency] = 0
            distances = eta2_coef

            manifold_class = manifold.SpectralEmbedding(n_components=1,
                                                        affinity='precomputed',
                                                        eigen_solver=None)

        embedding = manifold_class.fit_transform(distances)

        if out_path is not None:
            numpy.save(filename_embedding, embedding)

    print("\rLearning embedding... Done!", flush=True)

    ###
    #
    # Estimate a smooth function generating the voxels.
    #
    ###

    if out_path is not None and os.path.isfile(filename_connectopic_map):
        connectopic_map = numpy.load(filename_connectopic_map)
    else:

        # Run gaussian process on embedding to compute spatial statistics
        coord_x, coord_y, coord_z = numpy.where(roi_mask)
        coords = numpy.array([[x, y, z] for x, y, z in zip(coord_x, coord_y, coord_z)])
        connectopic_map, connectopic_var = spatial_statistic(coords, embedding, coords)

        if out_path is not None:
            numpy.save(filename_connectopic_map, connectopic_map)

    print("Spatial statistics... Done!", flush=True)

    return embedding, connectopic_map

def visualize_volume(fig, data, brain_mask, roi_mask, title, cmap, slice_indexes):
    """ Visualize 2-dimensional views of the brain

    """

    # Get voxels color from embedding
    min_val = numpy.min(data, axis=0)
    max_val = numpy.max(data, axis=0)
    data_norm = (data - min_val) / (max_val - min_val)
    fun_cmap = pyplot.get_cmap(cmap)
    clr_rgb = fun_cmap(data_norm.flatten())

    def get_slice_coords(mask, plane):
        coords_mask = numpy.where(mask)
        coords_idx = numpy.where(coords_mask[plane[3]] == plane[2])[0]
        return coords_mask, coords_idx

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
        axes[i].scatter(coords_brain_x, coords_brain_y,
                        c='k', s=15, edgecolors='face')

        axes[i].hold(True)

        # Plot ROI
        coords_mask, coords_idx = get_slice_coords(roi_mask, slice_indexes[i])
        coords_roi_x = coords_mask[dims[0]][coords_idx]
        coords_roi_y = coords_mask[dims[1]][coords_idx]
        axes[i].scatter(coords_roi_x, coords_roi_y,
                        c=clr_rgb[coords_idx, :], s=15, edgecolors='face')

        axes[i].set_title("{0} at slice {2}={3}".format(title, *slice_indexes[i]))
        axes[i].legend(("Cortex", "ROI"), loc=2)
        axes[i].grid(True)

    #
    # Apply stylistic adjustments
    #

    # Set axis limits
    coords_3d = numpy.where(roi_mask)
    coords_x = numpy.sort(coords_3d[0])
    coords_y = numpy.sort(coords_3d[1])
    coords_z = numpy.sort(coords_3d[2])
    axes[0].set_xlim([coords_x[0] - 7, coords_x[-1] + 7])
    axes[0].set_ylim([coords_z[0] - 7, coords_z[-1] + 7])
    axes[1].set_xlim([coords_y[0] - 7, coords_y[-1] + 7])
    axes[2].set_ylim([coords_y[0] - 7, coords_y[-1] + 7])

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


def _normalize_nifti(image_path, fwhm=6):
        """
        Load a nifti 4D image, apply a spatial smooth and normalize it
        along the temporal axis. The normalization step subtract and
        divide the voxels' values by their means along the temporal axis.

        Parameters
        ----------
        image_path: string
            Filename of the 4D Nifti image.

        fwhm: float, (default: 6)
            Full width half maximum window dimension used to smooth the
            image. The same value will be used for all 3 dimensions.

        Returns
        -------
        nifti_data: numpy.ndarray, shape (N, )
            Voxels' values normalized that were active at least at one
            timepoint during the scan.

        idx_brain: tuple, shape (3)
            X,Y,Z-coordinates of the 3D region where the active voxels
            are located.
        """

        # Smooth
        nifti_image = image.smooth_img(image_path, 6)
        nifti_data = nifti_image.get_data()
        nifti_data_shape = nifti_data.shape

        # Keep just non-zero voxels
        is_zero = numpy.abs(nifti_data) < numpy.finfo(nifti_data.dtype).eps
        idx_brain = numpy.where(~numpy.all(is_zero, axis=-1))

        # Calc mean
        nifti_data = nifti_data[idx_brain]
        nifti_data_mean = numpy.mean(nifti_data, axis=-1)[..., numpy.newaxis]

        # Demean and normalize
        nifti_data = nifti_data - nifti_data_mean
        nifti_data = nifti_data / nifti_data_mean

        return nifti_data, idx_brain


""" Run pipeline
"""
if __name__ == "__main__":

    #
    # Define parameters
    #
    subject = '103414'
    session = 'REST1'
    scans = ['LR', 'RL']
    hemisphere = 'LH'  # LH or RH

    #
    # Define input/output locations
    #
    image_path = '/Users/michele/Development/Workspaces/' \
                 'UpWork/Morgan_Hough/Resources'
    image_path_0 = image_path + \
                   '/rfMRI/{0}_{1}/rfMRI_{1}_{2}_hp2000_clean.nii.gz' \
                   .format(subject, session, scans[0])
    image_path_1 = image_path + \
                   '/rfMRI/{0}_{1}/rfMRI_{1}_{2}_hp2000_clean.nii.gz' \
                   .format(subject, session, scans[1])

    out_path = '/Users/michele/Development/Workspaces/UpWork/' \
               'Morgan_Hough/Results/connectopic_mapping/manifold_v2/' \
               'rfMRI_{0}_{1}_{2}'.format(subject, session, hemisphere)

    #
    # Load ROI and brain masks
    #
    print("Loading ROI from atlas...", end="", flush=True)

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

    # Keep just one hemisphere
    roi_mask_width = roi_mask.shape[0]
    roi_mask_half_width = int(roi_mask_width/2)

    if hemisphere == 'LH':
        roi_mask[roi_mask_half_width:, :, :] = False
    else:
        roi_mask[:roi_mask_half_width:, :, :] = False

    print("\rLoading ROI from atlas... Done!", flush=True)

    print("Loading brain mask...", end="", flush=True)

    # Load brain mask
    brain_mask = numpy.zeros(harvard_oxford_data.shape, dtype=bool)
    brain_mask[numpy.nonzero(harvard_oxford_data)] = True

    print("\rLoading brain mask... Done!", flush=True)

    #
    # Load Nifti images, smooth with FWHM=6, compute % temporal change
    #
    print("Loading Nifti images (1/2)...", end="", flush=True)

    nifti_data_0, idx_brain_0 = _normalize_nifti(image_path_0)

    print("\rLoading Nifti images (2/2)...", end="", flush=True)

    nifti_data_1, idx_brain_1 = _normalize_nifti(image_path_1)

    print("\rLoading Nifti images... Done!", flush=True)

    print("Concatenating Nifti images...", end="", flush=True)

    def get_idx_unique(idx, dim):
        """ Transforms X,Y,Z coords to unique index.
        """
        return idx[0]*dim[1]*dim[2] + idx[1]*dim[2] + idx[2]

    # Keep just voxels that are non-zero in both scans and inside
    # brain mask
    brain_0 = numpy.zeros(brain_mask.shape, dtype=bool)
    brain_1 = numpy.zeros(brain_mask.shape, dtype=bool)

    brain_0[idx_brain_0] = True
    brain_1[idx_brain_1] = True

    # Update masks
    brain_mask = brain_mask * brain_0 * brain_1
    roi_mask = roi_mask * brain_mask
    nonroi_mask = ~roi_mask * brain_mask

    # Generate brain indexes
    idx_unique_brain_0 = get_idx_unique(idx_brain_0, brain_mask.shape)
    idx_unique_brain_1 = get_idx_unique(idx_brain_1, brain_mask.shape)

    # Generate ROI indexes
    idx_roi_mask = numpy.nonzero(roi_mask)

    idx_unique_roi = get_idx_unique(idx_roi_mask, brain_mask.shape)

    is_roi_0 = numpy.in1d(idx_unique_brain_0, idx_unique_roi)
    is_roi_1 = numpy.in1d(idx_unique_brain_1, idx_unique_roi)

    # Generate nonROI indexes
    idx_nonroi_mask = numpy.nonzero(nonroi_mask)

    idx_unique_nonroi = get_idx_unique(idx_nonroi_mask, brain_mask.shape)

    is_nonroi_0 = numpy.in1d(idx_unique_brain_0, idx_unique_nonroi)
    is_nonroi_1 = numpy.in1d(idx_unique_brain_1, idx_unique_nonroi)

    # Merge data
    num_voxels_in_roi = numpy.sum(is_roi_0)
    data = numpy.concatenate((
           numpy.concatenate((nifti_data_0[is_roi_0],
                              nifti_data_1[is_roi_1]), axis=-1),
           numpy.concatenate((nifti_data_0[is_nonroi_0],
                              nifti_data_1[is_nonroi_1]), axis=-1)), axis=0).T

    print(data.shape)
    assert(numpy.sum(is_roi_0) + numpy.sum(is_nonroi_0) == numpy.sum(brain_mask, axis=None))
    assert(numpy.sum(is_roi_1) + numpy.sum(is_nonroi_1) == numpy.sum(brain_mask, axis=None))

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
    embedding, connectopy = haak_mapping(data,
                                         num_voxels_in_roi,
                                         roi_mask,
                                         manifold_learning='spectral',
                                         out_path=out_path)

    # Slice coordinates (plane, axis, value, axis_index)

    x = 25  # 18 or 25

    if hemisphere == 'RH':
        bx = brain_mask.shape[0] - x
    else:
        bx = x

    if x == 18:
        slice_indexes = [('X-Z', 'Y', 65, 1),
                         ('Y-Z', 'X', bx, 0),
                         ('X-Y', 'Z', 50, 2)]
    elif x == 25:
        slice_indexes = [('X-Z', 'Y', 55, 1),
                         ('Y-Z', 'X', bx, 0),
                         ('X-Y', 'Z', 69, 2)]

    #
    # Display embedding
    #
    fig = pyplot.figure(1, tight_layout=True)
    visualize_volume(fig, embedding,
                     brain_mask, roi_mask,
                     "Voxels after Manifold Learning",
                     'terrain', slice_indexes)

    #
    # Display connectopy
    #
    fig = pyplot.figure(2, tight_layout=True)
    visualize_volume(fig, connectopy,
                     brain_mask, roi_mask,
                     "Voxels after Gaussian Processes",
                     'terrain', slice_indexes)

    pyplot.show()
