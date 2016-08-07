from __future__ import print_function
import numpy
cimport numpy
cimport cython

DTYPE = numpy.float32
DDTYPE = numpy.float64
ctypedef numpy.float32_t DTYPE_t
ctypedef numpy.float64_t DDTYPE_t

@cython.boundscheck(False)
@cython.cdivision(True)
def compute_similarity_map(numpy.ndarray[DTYPE_t, ndim=2] fingerprints, numpy.ndarray[long] idx_chunk):
    """Compute the eta2 coefficients from the voxels' fingerprints.

    The result is a symmetric square matrix where each element is a
    real value in the range 0.0 to 1.0 with 0 indicating the pair of
    voxels is completely dissimilar and 1 equals.

    Parameters
    ----------
    fingerprints : numpy.ndarray, shape (num_fingerprints, num_lower_dim)
        The correlation between the voxels and the lower dimensional
        embedding.

    idx_chunk : numpy.ndarray, shape (K, )
        Indexes of the num_fingerprints voxels to consider when
        computing the eta2 coefficients. This parameter can be used
        to split the computation among several CPUs.

    Returns
    -------
    eta2_coef : numpy.ndarray, shape (num_fingerprints, num_fingerprints)
        The symmetric matrix containing the eta square coefficients
        between pairs of voxels.

    """

    cdef:
        DDTYPE_t eta2_numerator, eta2_denominator
        DDTYPE_t fingerprints_ij_mean, fingerprints_ijk_mean
        DDTYPE_t fingerprints_i, fingerprints_j
        int num_fingerprints = fingerprints.shape[0]
        int num_lower_dim = fingerprints.shape[1]
        int i, j, k
        int i_chunk, j_chunk
        int progress, progress_old

    cdef int num_chunk = idx_chunk.shape[0]
    cdef numpy.ndarray[DTYPE_t, ndim=2] eta2_coef = numpy.zeros((num_chunk, num_fingerprints), dtype=DTYPE)

    progress_old = -1

    for i in range(num_chunk):

        i_chunk = idx_chunk[i]

        for j in range(i_chunk, num_fingerprints):

            fingerprints_ij_mean = 0.0

            for k in range(num_lower_dim):
                fingerprints_ij_mean += (fingerprints[i_chunk, k] + fingerprints[j, k]) / 2

            fingerprints_ij_mean /= num_lower_dim

            eta2_numerator = 0.0
            eta2_denominator = 0.0

            for k in range(num_lower_dim):
                fingerprints_i = fingerprints[i_chunk, k]
                fingerprints_j = fingerprints[j, k]

                fingerprints_ijk_mean = (fingerprints_i + fingerprints_j) / 2

                eta2_numerator += (fingerprints_i - fingerprints_ijk_mean)**2 + \
                                  (fingerprints_j - fingerprints_ijk_mean)**2

                eta2_denominator += (fingerprints_i - fingerprints_ij_mean)**2 + \
                                    (fingerprints_j - fingerprints_ij_mean)**2

            eta2_coef[i, j] = 1 - eta2_numerator / eta2_denominator

        progress = (i * 100) / num_chunk

        if progress > progress_old:
            print("\rCompute similarity map... {0}".format(progress), end="", flush=True)
            progress_old = progress

    return eta2_coef


@cython.boundscheck(False)
@cython.cdivision(True)
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

    cdef:

        int num_time = data_in_roi.shape[0]
        int num_in_voxels = data_in_roi.shape[1]
        int num_out_voxels = data_out_roi.shape[1]

        numpy.ndarray[DTYPE_t, ndim=2] corr = numpy.zeros((num_in_voxels, num_out_voxels), dtype=DTYPE)

        # Compute out of ROI data variance
        numpy.ndarray[DDTYPE_t] data_out_roi_mus = numpy.mean(data_out_roi, axis=0)
        numpy.ndarray[DDTYPE_t, ndim=2] data_out_roi_demean = data_out_roi - data_out_roi_mus[numpy.newaxis, :]
        numpy.ndarray[DDTYPE_t] data_out_roi_var = numpy.mean(data_out_roi_demean * data_out_roi_demean, axis=0)

        # Compute in ROI data variance
        numpy.ndarray[DDTYPE_t] data_in_roi_mus = numpy.mean(data_in_roi, axis=0)
        numpy.ndarray[DDTYPE_t, ndim=2] data_in_roi_demean = data_in_roi - data_in_roi_mus[numpy.newaxis, :]
        numpy.ndarray[DDTYPE_t] data_in_roi_var = numpy.mean(data_in_roi_demean * data_in_roi_demean, axis=0)

        # Squared variance
        numpy.ndarray[DDTYPE_t, ndim=2] data_sqrt_var = numpy.sqrt(numpy.repeat(data_in_roi_var[:, numpy.newaxis], num_out_voxels, axis=1) *
                                                                   numpy.repeat(data_out_roi_var[:, numpy.newaxis].T, num_in_voxels, axis=0))

        DDTYPE_t data_cov, data_corr

        int id_in_voxel, id_out_voxel, id_time

    progress_old = -1

    for id_out_voxel in range(num_out_voxels):
        for id_in_voxel in range(num_in_voxels):

            # Compute covariance between one in ROI time-series
            # and out of ROI time-series
            data_cov = 0

            for id_time in range(num_time):
                data_cov += data_out_roi_demean[id_time, id_out_voxel] * \
                            data_in_roi_demean[id_time, id_in_voxel]

            data_corr = data_cov / num_time / data_sqrt_var[id_in_voxel, id_out_voxel]

            corr[id_in_voxel, id_out_voxel] = data_corr

        progress = int(id_out_voxel * 100 / num_out_voxels)

        if progress > progress_old:
            print("\rComputing fingerprints... {0}%".format(progress), end="", flush=True)
            progress_old = progress

    return corr

def compute_similarity_map_old(fingerprints, idx_chunk=None):
    """Compute the eta2 coefficients from the voxels' fingerprints.

    The result is a symmetric square matrix where each element is a
    real value in the range 0.0 to 1.0 with 0 indicating the pair of
    voxels is completely dissimilar and 1 equals.

    Parameters
    ----------
    fingerprints : numpy.ndarray, shape (num_fingerprints, num_lower_dim)
        The correlation between the voxels and the lower dimensional
        embedding.

    idx_chunk : numpy.ndarray, shape (K, ), (default: None)
        Indexes of the num_fingerprints voxels to consider when
        computing the eta2 coefficients. If idx_chunk=None all
        num_fingerprints are considered. This parameter can be used
        to split the computation among several CPUs.

    Returns
    -------
    eta2_coef : numpy.ndarray, shape (num_fingerprints, num_fingerprints)
        The symmetric matrix containing the eta square coefficients
        between pairs of voxels.

    """

    num_chunk = idx_chunk.shape[0]
    num_fingerprints = fingerprints.shape[0]
    num_lower_dim = fingerprints.shape[1]

    if idx_chunk is None:
        idx_chunk = numpy.arange(0, num_fingerprints)

    # Utility method to compute eta square
    def sum_coeff(fingerprints_0, fingerprints_1, fingerprints_mean):
        addend_0 = (fingerprints_0 - fingerprints_mean)**2
        addend_1 = (fingerprints_1 - fingerprints_mean)**2
        return numpy.sum(addend_0 + addend_1, axis=1)

    # Allocate memory where to store eta square coefficients
    eta2_coef = numpy.zeros((num_chunk, num_fingerprints))

    for i in range(num_chunk):

        # Get fingerprint's index
        chunk_i = idx_chunk[i]

        # Fingerprints left to compute
        num_chunk_i = num_fingerprints - chunk_i

        # i-th fingerprint to compute
        fingerprints_i = fingerprints[chunk_i, :][numpy.newaxis, :]
        fingerprints_i_rep = numpy.repeat(fingerprints_i, num_chunk_i, axis=0)

        # Fingerprints pair to correlate with the i-th
        fingerprints_from_i = fingerprints[chunk_i:, :]

        fingerprints_mean = (fingerprints_i_rep + fingerprints_from_i) / 2

        fingerprints_mean_row = numpy.mean(fingerprints_mean, axis=1)[:, numpy.newaxis]
        fingerprints_mean_rep = numpy.repeat(fingerprints_mean_row, num_lower_dim, axis=1)

        # Compute eta square
        eta2_numerator = sum_coeff(fingerprints_i_rep,
                                   fingerprints_from_i,
                                   fingerprints_mean)

        eta2_denominator = sum_coeff(fingerprints_i_rep,
                                     fingerprints_from_i,
                                     fingerprints_mean_rep)

        eta2_coef[i, chunk_i:] = 1 - eta2_numerator / eta2_denominator

    return eta2_coef