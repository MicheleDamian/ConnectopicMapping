import numpy
cimport numpy

DTYPE = numpy.float32
ctypedef numpy.float32_t DTYPE_t

def compute_pearson(numpy.ndarray[DTYPE_t, ndim=2] data_in_roi,
                    numpy.ndarray[DTYPE_t, ndim=2] data_out_roi):
    """ Compute pearson correlations for each voxel time-series in
        the ROI and each voxel in the brain.

    Parameters
    ----------
    data_in_roi: numpy.ndarray (n_timeseries, n_in_roi_voxels)
        Voxels in the ROI on which to compute the correlation. Each voxel's
        time-serie is represented by a column in data_in_roi.

    data_out_roi: numpy.ndarray (n_timeseries, n_out_roi_voxels)
        Voxels out of the ROI on which to compute the correlations.

    Returns
    -------
    corr: numpy.ndarray (n_in_roi_voxels, n_out_roi_voxels)
        Pearson correlations coefficients.

    """

    cdef numpy.ndarray[DTYPE_t, ndim=2] corr = numpy.zeros((data_in_roi.shape[1], data_out_roi.shape[1]), dtype=DTYPE)

    progress_old = -1

    # Compute out of ROI data variance
    cdef numpy.ndarray[DTYPE_t] data_out_roi_mus = numpy.mean(data_out_roi, axis=0)
    cdef numpy.ndarray[DTYPE_t, ndim=2] data_out_roi_demean = data_out_roi - data_out_roi_mus[numpy.newaxis, :]
    cdef numpy.ndarray[DTYPE_t] data_out_roi_var = numpy.mean(data_out_roi_demean * data_out_roi_demean, axis=0)

    # Compute in ROI data variance
    cdef numpy.ndarray[DTYPE_t] data_in_roi_mus = numpy.mean(data_in_roi, axis=0)
    cdef numpy.ndarray[DTYPE_t, ndim=2] data_in_roi_demean = data_in_roi - data_in_roi_mus[numpy.newaxis, :]
    cdef numpy.ndarray[DTYPE_t] data_in_roi_var = numpy.mean(data_in_roi_demean * data_in_roi_demean, axis=0)

    cdef numpy.ndarray[DTYPE_t] data_i_voxel_demean = numpy.zeros((data_out_roi_demean.shape[0], ), dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t, ndim=2] data_voxel_demean = numpy.zeros((data_out_roi_demean.shape[0], data_in_roi_demean.shape[1]), dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t] data_cov = numpy.zeros((data_voxel_demean.shape[1], ), dtype=DTYPE)
    cdef numpy.ndarray[DTYPE_t] data_in_out_roi_cov = numpy.zeros((data_in_roi_demean.shape[1], ), dtype=DTYPE)

    cdef int i_voxel

    for i_voxel in range(data_out_roi.shape[1]):

        # Compute covariance between one in ROI time-series
        # and out of ROI time-series
        data_i_voxel_demean = data_out_roi_demean[:, i_voxel]
        data_voxel_demean = numpy.repeat(data_i_voxel_demean[:, numpy.newaxis],
                                         data_in_roi_demean.shape[1], axis=1)
        data_cov = numpy.mean(data_voxel_demean * data_in_roi_demean, axis=0)

        data_in_out_roi_cov = numpy.sqrt(data_out_roi_var[i_voxel] * data_in_roi_var)

        corr[:, i_voxel] = data_cov / data_in_out_roi_cov

        progress = int(i_voxel * 100 / data_out_roi.shape[1])

        if progress > progress_old:
            print("\rComputing fingerprints... {0}%".format(progress))
            progress_old = progress

    return corr
