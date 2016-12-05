"""Utility methods to manipulate and visualize voxels."""

import numpy
import nibabel
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import Grid
from nilearn import datasets, image

def load_masks(dataset, roi_label, hemisphere, remove_white_matter=True):
    """Load brain and region of interest (ROI) masks.

    Construct brain and ROI masks from a labeled Harvard-Oxford's `dataset`
    (e.g. 'cort-maxprob-thr25-2mm').

    Parameters
    ----------
    dataset : string
        Name of the labeled dataset from the Harvard-Oxford's atlas
        from where to extract the region of interest. The voxels must
        have a resolution of 2mm.

    roi_label : string
        Name of the label from the dataset corresponding to the region
        of interest.

    hemisphere : string, (options: 'LH', 'RH', 'both')
        Hemisphere on which the region of interest is located
        (left 'LH', right 'RH' or 'both')

    remove_white_matter : bool, (default: True)
        Remove white matter voxels from masks.

    Returns
    -------
    brain_mask : numpy.ndarray, shape (x_dim, y_dim, z_dim)
        Boolean matrix where each cell is True if the voxel is part
        of the brain, False otherwise.

    roi_mask : numpy.ndarray, shape (x_dim, y_dim, z_dim)
        Boolean matrix where each cell is True if the voxel is part
        of the region of interest, False otherwise.

    Examples
    --------
    This is the Harvard-Oxford dataset:

    >>> import numpy, nilearn, nibabel
    >>> cortex_dataset = nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    >>> cortex_labels = numpy.array(cortex_dataset.labels)
    >>> 'Precentral Gyrus' in cortex_dataset.labels
    True
    >>> cortex_data = nibabel.load(cortex_dataset.maps).get_data()
    >>> cortex_data.shape
    (91, 109, 91)

    This is how to load the masks:

    >>> brain_mask, roi_mask = utils.load_masks('cort-maxprob-thr25-2mm', 'Precentral Gyrus', 'both', False)
    >>> brain_mask.shape
    (91, 109, 91)
    >>> roi_mask.shape
    (91, 109, 91)

    And these are the definitions of the masks
    (with `remove_white_matter` == False):

    >>> idx_background_label = numpy.where((cortex_labels == 'Background'))[0][0]
    >>> numpy.all(brain_mask == (cortex_data != idx_background_label))
    True
    >>> idx_pregyrus_label = numpy.where((cortex_labels == 'Precentral Gyrus'))[0][0]
    >>> numpy.all(roi_mask == (cortex_data == idx_pregyrus_label))
    True

    """

    # Load region
    cortex_dataset = datasets.fetch_atlas_harvard_oxford(dataset)
    cortex_labels = numpy.array(cortex_dataset.labels)
    cortex_maps = nibabel.load(cortex_dataset.maps)
    cortex_data = cortex_maps.get_data()

    roi_indexes = numpy.where((cortex_labels == roi_label))[0]

    # Build mask from ROI
    roi_mask = numpy.zeros(cortex_data.shape, dtype=bool)
    for index in roi_indexes:
        roi_mask[cortex_data == index] = True

    # Keep just one hemisphere
    roi_mask_width = roi_mask.shape[0]
    roi_mask_half_width = int(roi_mask_width/2)

    if hemisphere == 'LH':
        roi_mask[roi_mask_half_width:, :, :] = False
    elif hemisphere == 'RH':
        roi_mask[:roi_mask_half_width:, :, :] = False

    # Load brain mask
    brain_mask = numpy.zeros(cortex_data.shape, dtype=bool)
    brain_mask[numpy.nonzero(cortex_data)] = True

    # Load white matter mask
    if remove_white_matter:

        subcortex_dataset = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
        subcortex_labels = numpy.array(subcortex_dataset.labels)
        subcortex_maps = nibabel.load(subcortex_dataset.maps)
        subcortex_data = subcortex_maps.get_data()

        white_indexes = numpy.where((subcortex_labels == "Left Cerebral White Matter") +
                                    (subcortex_labels == "Right Cerebral White Matter"))[0]

        white_mask = numpy.zeros(subcortex_data.shape, dtype=bool)
        for index in white_indexes:
            white_mask[subcortex_data == index] = True

        # Remove white matter from masks
        brain_mask = brain_mask * ~white_mask
        roi_mask = roi_mask * ~white_mask

    return brain_mask, roi_mask


def visualize_volume(data, brain_mask, roi_mask,
                     slice_indexes,
                     low_percentile=10, high_percentile=90,
                     num_fig=0,
                     title="Projection", margin=1,
                     legend_location=1, cmap='gist_rainbow'):
    """Visualize the projections of the brain onto the XY, XZ and YZ planes.

    Parameters
    ----------
    data : numpy.ndarray, shape (num_voxels, )
        1-dimensional data containing the voxels' values inside the ROI
        as defined by `roi_mask`.

    brain_mask : numpy.ndarray, shape (x_dim, y_dim, z_dim)
        Boolean 3-dimensional numpy ndarray where voxels marked as True
        are part of the brain.

    roi_mask : numpy.ndarray, shape (x_dim, y_dim, z_dim)
        Boolean 3-dimensional numpy ndarray where voxels marked as True
        are part of the same region of interest.

    low_percentile : int, (default: 10)
        Lower percentile at which to start normalizing `data`. Lower values
        are encoded as `low_percentile`. Consider increasing (decreasing)
        it if the distribution of `data` has a long (short) left tail.

    high_percentile : int, (default: 90)
        Higher percentile at which to stop normalizing `data`. Higher values
        are encoded as `high_percentile`. Consider decreasing (increasing)
        it if the distribution of `data` has a long (short) left tail.

    slice_indexes : numpy.ndarray, shape (3, )
        List of int representing the X, Y and Z indexes where to cut the
        YZ, XZ and XY planes respectively.

    num_fig : int, (default: 0)
        Number of the figure where to plot the graphs.

    title : str, (default: "Projection")
        Title for the three graphs to visualize. It is followed by the
        plane the graph corresponds to and the coordinates of the plane.

    margin : int, (default: 1)
        Number of voxels between the ROI and the axes in the visualization.

    legend_location : int, (default: 1)
        Location of the legend in the graph. 1 := North-East, 2 :=
        North-West, 3 := South-West and 4 := South-East. If None the
        legend is not visualized.

    cmap : str, (default: 'gist_rainbow')
        String that identifies a matplotlib.colors.Colormap instance.

    Examples
    --------
    >>> slice_indexes = [18, 65, 50]
    >>> visualize_volume(data, brain_mask, roi_mask, slice_indexes)

    """

    slice_config = [('X-Z', 'Y', slice_indexes[1], 1),
                    ('Y-Z', 'X', slice_indexes[0], 0),
                    ('X-Y', 'Z', slice_indexes[2], 2)]

    # Get voxels color from embedding
    min_val = numpy.percentile(data, low_percentile)
    max_val = numpy.percentile(data, high_percentile)
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
    axes = Grid(num_fig, rect=111, nrows_ncols=(2, 2), label_mode='L')

    # Display in X, Y and Z subplot
    for i in range(3):

        dims = numpy.delete(numpy.arange(3), slice_config[i][3])

        # Plot brain
        coords_mask, coords_idx = get_slice_coords(brain_mask, slice_config[i])
        coords_brain_x = coords_mask[dims[0]][coords_idx]
        coords_brain_y = coords_mask[dims[1]][coords_idx]
        axes[i].scatter(coords_brain_x, coords_brain_y,
                        c=[0.5, 0.5, 0.5], s=15, edgecolors='face')

        axes[i].hold(True)

        # Plot ROI
        coords_mask, coords_idx = get_slice_coords(roi_mask, slice_config[i])
        coords_roi_x = coords_mask[dims[0]][coords_idx]
        coords_roi_y = coords_mask[dims[1]][coords_idx]
        axes[i].scatter(coords_roi_x, coords_roi_y,
                        c=clr_rgb[coords_idx, :], s=15, edgecolors='face')

        axes[i].set_title("{0} at coord {2}={3}".format(title, *slice_config[i]))

        if legend_location is not None:
            axes[i].legend(("Cortex", "ROI"), loc=legend_location)

        axes[i].grid(True)

    #
    # Apply stylistic adjustments
    #

    # Set axis limits
    coords_3d = numpy.where(roi_mask)
    coords_x = numpy.sort(coords_3d[0])
    coords_y = numpy.sort(coords_3d[1])
    coords_z = numpy.sort(coords_3d[2])

    axes[0].set_xlim([coords_x[0] - margin, coords_x[-1] + margin])
    axes[0].set_ylim([coords_z[0] - margin, coords_z[-1] + margin])
    axes[1].set_xlim([coords_y[0] - margin, coords_y[-1] + margin])
    axes[2].set_ylim([coords_y[0] - margin, coords_y[-1] + margin])

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
    cbaxes = num_fig.add_axes([0.95, 0.1, 0.01, 0.4])
    cbar = pyplot.colorbar(sm, cax=cbaxes)
    cbar.set_ticklabels([])


def normalize_nifti_image(image_path, fwhm=6):
        """Normalize a 4D Nifti image.

        Apply a spatial smooth of dimension `fwhm` on a 4D nifti image
        located at `image_path` and normalize it along the temporal axis.
        The normalization step subtracts and divides the voxels' values
        by their means. The method returns just the active voxels and
        their XYZ-coordinates to reduce memory usage.

        Parameters
        ----------
        image_path : string
            Path of the 4D Nifti image.

        fwhm : float, (default: 6)
            Full width half maximum window dimension used to smooth the
            image. The same value will be used for all 3 spatial dimensions.

        Returns
        -------
        nifti_data : numpy.ndarray, shape (num_timepoints, num_voxels)
            Voxels' values normalized that were active at least at one
            timepoint during the scan.

        idx_active : tuple, shape (3)
            X,Y,Z-coordinates of the 3D brain where the active voxels
            are located.

        """

        # Smooth
        nifti_image = image.smooth_img(image_path, fwhm)
        nifti_data = nifti_image.get_data()

        # Keep just non-zero voxels
        is_zero = numpy.abs(nifti_data) < numpy.finfo(nifti_data.dtype).eps
        idx_active = numpy.where(~numpy.all(is_zero, axis=-1))

        # Calc mean
        nifti_data = nifti_data[idx_active]
        nifti_data_mean = numpy.mean(nifti_data, axis=-1)[..., numpy.newaxis]

        # Demean and normalize
        nifti_data = nifti_data - nifti_data_mean
        nifti_data = nifti_data / nifti_data_mean

        return nifti_data.T, idx_active


def concatenate_data(brain_mask, roi_mask, data_0, data_xyz_0, data_1, data_xyz_1):
    """Concatenate data.

    Concatenate `data_0` and `data_1` along their first dimension such that
    the first part of the result represents ROI voxels and the second part
    represents outside of ROI voxels, as defined by `brain_mask` and `roi_mask`.

    Parameters
    ----------
    brain_mask : numpy.ndarray, shape (x_dim, y_dim, z_dim)
        A 3-dimensional boolean matrix where each cell represent a
        voxel and is True if it belongs to the brain, False otherwise.

    roi_mask : numpy.ndarray, shape (x_dim, y_dim, z_dim)
        A 3-dimensional boolean matrix of the same size of `brain_mask`
        where each cell is True if the voxel belongs to the region of
        interest, False otherwise.

    data_0, data_1 : numpy.ndarray, shape (num_timepoints, num_voxels)
        Voxels' time-series inside the ROI.

    data_xyz_0, data_xyz_1 : tuple, shape (3, num_voxels)
        X,Y,Z-coordinates of `data_0`/`data_1`. Each element of the triplet
        is an array-like list of voxels' coordinates along an axis, stored
        in the same order of the voxels' values in `data_0`/`data_1`.

    Returns
    -------
    brain_mask : numpy.ndarray, shape (x_dim, y_dim, z_dim)
        As the homonymous input parameter, but updated such that just the
        voxels present in `data_0` and `data_1` are set to True.

    roi_mask : numpy.ndarray, shape (x_dim, y_dim, z_dim)
        As the homonymous input parameter, but updated such that just the
        voxels present in `data_0` and `data_1` are set to True.

    data : numpy.ndarray, shape (num_timepoints, num_voxels)
        Input data concatenated along the num_timepoints axe such that
        the first ``numpy.sum(roi_mask)`` of `num_voxels` are voxels inside
        the region of interest, while the other voxels are outside.

    Examples
    --------

    Concatenate two scans:

    >>> brain_mask, roi_mask = utils.load_masks('cort-maxprob-thr25-2mm', 'Precentral Gyrus', 'both') # Load masks
    >>> data_info_0 = utils.normalize_nifti_image(image_path_0) # Load scan 0
    >>> data_info_1 = utils.normalize_nifti_image(image_path_1) # Load scan 1
    >>> brain_mask, roi_mask, data = utils.concatenate_data(brain_mask, roi_mask, *data_info_0, *data_info_1) # Concatenate scans

    """

    def get_idx(xyz, dim):
        """ Transforms X,Y,Z coords to a unique index.
        """
        return xyz[0]*dim[1]*dim[2] + xyz[1]*dim[2] + xyz[2]

    # Keep just voxels that are non-zero in both scans and inside
    # brain mask
    brain_data_0 = numpy.zeros(brain_mask.shape, dtype=bool)
    brain_data_1 = numpy.zeros(brain_mask.shape, dtype=bool)

    brain_data_0[data_xyz_0] = True
    brain_data_1[data_xyz_1] = True

    # Remove non active voxels from masks (i.e., voxels not present
    # in the data)
    brain_mask = brain_mask * brain_data_0 * brain_data_1
    roi_mask = roi_mask * brain_mask
    nonroi_mask = ~roi_mask * brain_mask

    # Generate brain indexes
    data_idx_0 = get_idx(data_xyz_0, brain_mask.shape)
    data_idx_1 = get_idx(data_xyz_1, brain_mask.shape)

    # Generate inside ROI indexes
    roi_mask_xyz = numpy.nonzero(roi_mask)
    roi_mask_idx = get_idx(roi_mask_xyz, brain_mask.shape)

    # Generate outside ROI indexes
    nonroi_mask_xyz = numpy.nonzero(nonroi_mask)
    nonroi_mask_idx = get_idx(nonroi_mask_xyz, brain_mask.shape)

    # Test if index is inside ROI
    is_roi_0 = numpy.in1d(data_idx_0, roi_mask_idx)
    is_roi_1 = numpy.in1d(data_idx_1, roi_mask_idx)

    # Test if index is outside ROI
    is_nonroi_0 = numpy.in1d(data_idx_0, nonroi_mask_idx)
    is_nonroi_1 = numpy.in1d(data_idx_1, nonroi_mask_idx)

    # Merge data
    data = numpy.concatenate((
           numpy.concatenate((data_0[:, is_roi_0], data_1[:, is_roi_1]), axis=0),
           numpy.concatenate((data_0[:, is_nonroi_0], data_1[:, is_nonroi_1]), axis=0)),
                             axis=1)

    assert(numpy.sum(is_roi_0) + numpy.sum(is_nonroi_0) == numpy.sum(brain_mask, axis=None))
    assert(numpy.sum(is_roi_1) + numpy.sum(is_nonroi_1) == numpy.sum(brain_mask, axis=None))

    return brain_mask, roi_mask, data


def save_nifti(file_path, data, mask, affine=None, zooms=[2,2,2]):
    """Save a nifti image.

    Save the voxels that are set True in `mask` and which values are contained
    in `data` in a 3D nifti image.

    Parameters
    ----------
    file_path : str
        File path of the nifti image.

    data : numpy.ndarray, shape (num_voxels, )
        Values of the voxels. The order of the voxels must be the same as
        their coordinates in `numpy.where(mask)`.

    mask : numpy.ndarray, shape (x_dim, y_dim, z_dim)
        3-dimensional boolean numpy array representing brain voxels, where
        an element is set to True if and only if it is present in data.

    affine : numpy.ndarray, shape (4, 4), (default=None)
        Nifti affine matrix.

    zooms : array_like, shape(3, ), (default: [2,2,2])
        Nifti zooms in millimeters.

    """

    if affine is None:
        affine = [[-2, 0, 0,  90],
                  [0, 2, 0, -126],
                  [0, 0, 2, -72],
                  [0, 0, 0, 0]]

    # Set voxels values
    coords = numpy.where(mask)
    out = numpy.zeros(mask.shape)
    out[coords] = data

    # Transform connectopic map into Nifti image
    nifti = nibabel.Nifti1Image(out, affine)
    nifti.header.set_data_dtype(numpy.float64)
    nifti.header.set_zooms(zooms)
    nifti.to_filename(file_path)
