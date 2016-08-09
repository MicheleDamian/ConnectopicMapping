"""Class and methods to implement Haak's pipeline."""
from __future__ import print_function
import multiprocessing
import os
import GPy
import numpy
from scipy.sparse import csgraph
from scipy.optimize import minimize
from sklearn import manifold

cimport numpy
cimport cython

__author__ = "Michele Damian"
__email__ = "michele.damian@gmail.com"
__version__ = "0.3.0br1"


class Haak:
    """Extract grey matter connectopies from resting state fMRI data.

    Parameters
    ----------
    num_lower_dim : str or int, (options: 'haak', 'full'), (default: 'haak')
        Number of dimensions of the lower dimensional space used
        to compute the fingerprints.

        * If `num_lower_dim` is a natural number, that specifies the
          dimensions.
        * If `num_lower_dim` == 'haak', the length of the time-series - 1
          is used [1]_.
        * If `num_lower_dim` == 'full', all dimensions are used and the
          resulting fingerprints are the correlation of a voxel in the
          ROI with every other voxel.

    num_processes : str or int, (options: 'cpu'), (default: 'cpu')
        Number of processes to create to compute the fingerprints.

        * If `num_processes` is a natural number, that specifies among how
          many processes to split the execution.
        * If `num_processes` == 'cpu', the method balance the execution among
          all available CPUs.

    manifold_learning : str, (options: 'tSNE', 'spectral', 'isomap'), (default: 'spectral')
        Manifold learning algorithm to use to find a lower dimensional
        representation of the voxels correlation map. The options are
        tSNE, spectral embedding or isomap. Refer to sklearn.manifold
        for further documentation.

    manifold_components : int, (default: 1)
        Number of components of the lower dimensional manifold.

    out_path : str, (default: '.')
        Path of the folder where to save the intermediate results. If it
        doesn't exist the method tries to create one. By default the
        results are saved in the same folder where the script is located.
        They are not recomputed if they are are already present;
        unless `out_path` == None, then no intermediate result is saved
        and everything is recomputed.

    verbose : int, (default: 0)
        Verbosity level.

    Notes
    -----
    This implementation differs from [1]_ in the way it computes the
    fingerprints from the voxels. [1]_ separates the in-ROI and
    out-ROI voxels, reduces out-ROI dimensionality by singular value
    decomposition and computes the fingerprints as the correlations
    between in-ROI voxels and the reduced out-ROI voxels.
    This implementation keeps the in-ROI and out-ROI voxels together,
    it reduces their dimensionality by singular value decomposition and
    use the right eigenvectors of the in-ROI voxels as fingerprints.

    In case spectral embedding is used the pipeline creates a graph with
    `num_fingerprints` vertexes and edges' weights described by the
    `eta2_coef` matrix. Then, from the set of all the edges, it keeps
    the smallest subset with the smallest weights such that each
    pair of vertex is still connected (through any number of edges).
    Spectral embedding learns a manifold of this graph.

    References
    ----------
    .. [1] Haak et al., "Connectopic mapping with resting-state fMRI", 2016
    .. [2] Cohen et al., "Defining functional areas in individual human brains using resting functional connectivity MRI", 2008

    """

    def __init__(self, num_lower_dim='haak', num_processes='cpu',
                 manifold_learning='spectral', manifold_components=1,
                 out_path='.', verbose=0):

        assert(manifold_learning in ['tSNE', 'spectral', 'isomap'])
        assert(num_lower_dim in ['haak', ] or type(num_lower_dim) == int)

        self._manifold_learning = manifold_learning
        self._manifold_components = manifold_components
        self._out_path = out_path
        self._verbose = verbose
        self._num_lower_dim = num_lower_dim

        # Get number of CPUs available
        if num_processes == 'cpu':
            self._num_processes = multiprocessing.cpu_count()
        else:
            self._num_processes = num_processes

        #
        # Define filenames of intermediate results
        #
        if self._out_path is not None:

            # Create folder and sub-folders if they don't exist
            os.makedirs(out_path, exist_ok=True)

            # Instantiate filename variables
            self._fname_fingerprints = out_path + os.sep + \
                'fingerprints.npy'
            self._fname_eta2_coef = out_path + os.sep + \
                'eta2_coef.npy'
            self._fname_embedding = out_path + os.sep + \
                'embedding_{0}.npy'.format(manifold_learning)
            self._fname_connectopic_map = out_path + os.sep + \
                'connectopic_map_{0}.npy'.format(manifold_learning)
            self._fname_connectopic_var = out_path + os.sep + \
                'connectopic_var_{0}.npy'.format(manifold_learning)

    def fit_transform(self, data, roi_mask):
        """Run the pipeline to extract connectopies from time-series.

        Parameters
        ----------
        data : numpy.ndarray, shape (num_timepoints, num_voxels)
            The voxels' time-series from the resting state fMRI acquired.
            The first N voxels (with N equal to the number of voxels set
            to True in `roi_mask`) along the `num_voxels` axes must
            belong to the region of interest where the mapping is computed.

        roi_mask : numpy.ndarray, shape (x_dim, y_dim, z_dim)
            Boolean 3-dimensional numpy array with the same dimensions of a
            single frame of the nifti image provided as input. Voxels
            marked as True are part of the same region of interest (ROI).

        Returns
        -------
        eta2_coef : numpy.ndarray, shape (num_fingerprints, num_fingerprints)
            Symmetric matrix containing the eta square coefficients
            between pairs of voxels.

        embedding : numpy.ndarray, shape (num_fingerprints, manifold_components)
            Lower dimensional manifold.

        connectopic_map : numpy.ndarray, (n_predictions, n_dim)
            Posterior mean from the Gaussian process for each ROI voxel.

        connectopic_var : numpy.ndarray, (n_predictions, )
            Variance from the Gaussian process for each ROI voxel.

        """

        #
        # Compute ROI voxels fingerprints
        #

        if self._verbose:
            print("Computing fingerprints...", end="", flush=True)

        if self._out_path is not None and os.path.isfile(self._fname_fingerprints):
            fingerprints = numpy.load(self._fname_fingerprints)
        else:

            fingerprints = self._compute_fingerprints(data, roi_mask)

            if self._out_path is not None:
                numpy.save(self._fname_fingerprints, fingerprints)

        if self._verbose:
            print("\rComputing fingerprints... Done!", flush=True)

        #
        # Compute similarities between pair of voxels' fingerprints as
        # eta square coefficients
        #

        if self._verbose:
            print("Computing similarity maps...", end="", flush=True)

        if self._out_path is not None and os.path.isfile(self._fname_eta2_coef):
            eta2_coef = numpy.load(self._fname_eta2_coef)
        else:

            eta2_coef = self._distribute_similarity_map(fingerprints)

            if self._out_path is not None:
                numpy.save(self._fname_eta2_coef, eta2_coef)

        if self._verbose:
            print("\rComputing similarity maps... Done!", flush=True)

        #
        # Learn the manifold for this data and reproject the similarity
        # graph on the two most significant connectopies
        #

        if self._verbose:
            print("Learning embedding...", end="", flush=True)

        if self._out_path is not None and os.path.isfile(self._fname_embedding):
            embedding = numpy.load(self._fname_embedding)
        else:

            embedding = self._learn_embedding(eta2_coef)

            if self._out_path is not None:
                numpy.save(self._fname_embedding, embedding)

        if self._verbose:
            print("\rLearning embedding... Done!", flush=True)

        #
        # Estimate a smooth function generating the voxels.
        #

        if self._out_path is not None and os.path.isfile(self._fname_connectopic_map):
            connectopic_map = numpy.load(self._fname_connectopic_map)
            connectopic_var = numpy.load(self._fname_connectopic_var)
        else:

            # Run gaussian process on embedding to compute spatial statistics
            connectopic_map, connectopic_var = self._spatial_statistic(embedding, roi_mask)

            if self._out_path is not None:
                numpy.save(self._fname_connectopic_map, connectopic_map)
                numpy.save(self._fname_connectopic_var, connectopic_var)

        if self._verbose:
            print("Spatial statistics... Done!", flush=True)

        return eta2_coef, embedding, connectopic_map, connectopic_var

    def _spatial_statistic(self, data, roi_mask):
        """Approximate each connectopic map using a spatial model.

        The input is smoothed by a zero-mean Gaussian process where the
        kernel is given by a Marten covariance function added to Gaussian
        noise. This method estimate the parameters and return the values
        predicted at the locations in x_predictions.

        Parameters
        ----------
        data : numpy.ndarray, shape (num_timepoints, num_voxels)
            The voxels' timeseries from the resting state fMRI acquired.
            The first N voxels (with N equal to the number of voxels set
            to True in roi_mask) along the num_voxels axes belong to the
            region of interest where the mapping should be computed.

        roi_mask : numpy.ndarray, shape (x_dim, y_dim, z_dim)
            Boolean 3-dimensional numpy array with the same dimensions of a
            single frame of the nifti image provided as input. Voxels
            marked as True are part of the same region of interest (ROI).

        Returns
        -------
        y_means : numpy.ndarray, (n_predictions, n_dim)
            A posterior mean from the Gaussian process for each dimension
            given as input.

        y_vars : numpy.ndarray, (n_predictions, )
            Variance at each x_predictions point.

        """

        coord_x, coord_y, coord_z = numpy.where(roi_mask)
        coords = numpy.array([[x, y, z] for x, y, z in zip(coord_x, coord_y, coord_z)])

        # Matern kernel with v=5/2, the GPRegression model includes a
        # Gaussian noise by default.
        kernMat = GPy.kern.Matern52(input_dim=3, variance=1., lengthscale=1.)
        kernRBF = GPy.kern.RBF(input_dim=3, variance=1, lengthscale=1)

        # Estimate the model
        model = GPy.models.GPRegression(coords, data, kernMat+kernRBF)
        model.optimize(messages=True)

        y_means, y_vars = model.predict(coords)

        return y_means, y_vars

    def _compute_fingerprints(self, data, roi_mask):
        """Computes the voxels' fingerprints using the left eigenvectors
        and the eigenvalues from singular value decomposition.

        Each row of the output matrix represents the correlation between
        a voxel inside the ROI and the num_lower_dim axis in the lower
        dimensional space.

        Parameters
        ----------
        data : numpy.ndarray, shape (num_timepoints, num_voxels)
            A real matrix representing the voxels' time-series with one
            voxel per each column. The first num_fingerprints columns
            are the voxels on which to compute the fingerprints.

        Returns
        -------
        fingerprints : numpy.ndarray, shape (num_fingerprints, num_lower_dim)
            The correlation between the voxels and the lower dimensional
            embedding.
        """

        num_fingerprints = numpy.sum(roi_mask, axis=None)

        data_u, data_s, data_v = numpy.linalg.svd(data, full_matrices=False)

        # Keep a just num_lower_dim dimensions
        if self._num_lower_dim == 'haak':
            num_lower_dim = data.shape[0] - 1
        elif self._num_lower_dim == 'full':
            num_lower_dim = data_s.shape[0]

        data_s = data_s[:num_lower_dim]
        data_v = data_v[:num_lower_dim, :num_fingerprints]

        fingerprints = numpy.dot(numpy.diag(data_s), data_v)

        return fingerprints.T

    def _distribute_similarity_map(self, fingerprints):
        """Create several processes to run the method compute_similarity_map
        in parallel and balance the load among them.

        Parameters
        ----------
        fingerprints : numpy.ndarray, shape (num_fingerprints, num_lower_dim)
            The correlation between the voxels and the lower dimensional
            embedding.

        Results
        -------
        eta2_coef : numpy.ndarray, shape (num_fingerprints, num_fingerprints)
            The symmetric matrix containing the eta square coefficients
            between pairs of voxels.

        """

        num_fingerprints = fingerprints.shape[0]

        # Distribute load equally among all CPUs
        pool = multiprocessing.Pool(self._num_processes)

        # Approximation of the optimal fraction of the dataset to
        # allocate to each CPU
        processes_idx = numpy.arange(self._num_processes + 1)
        fingerprints_ratio = processes_idx * 2 / (self._num_processes * (self._num_processes + 1))

        ratio_loss = lambda ratios: sum([(ratios[i-1]*sum(ratios[i-1:]) - ratios[i]*sum(ratios[i:]))**2 for i in range(1, len(ratios))]) + (sum(ratios) - 1)**2

        res_optimize = minimize(ratio_loss, fingerprints_ratio[1:], method='nelder-mead', options={'xtol': 1e-8})

        if all(res_optimize.x > 0) and all(res_optimize.x < 1):
            fingerprints_ratio[1:] = res_optimize.x

        fingerprints_idx = (numpy.cumsum(fingerprints_ratio) * num_fingerprints).astype(int)
        fingerprints_idx[-1] = num_fingerprints

        pool_idx = [numpy.arange(fingerprints_idx[i], fingerprints_idx[i+1])
                    for i in range(len(fingerprints_idx)-1)]

        starmap_input = [(fingerprints, idx) for idx in pool_idx]

        # Run compute_similarity_map in parallel
        eta2_chunks = pool.starmap(self._compute_similarity_map, starmap_input)

        # Merge together results from all processes
        eta2_coef = numpy.zeros((num_fingerprints, num_fingerprints))

        for i_cpu in range(self._num_processes):
            for i_chunk in range(len(pool_idx[i_cpu])):

                i_eta = pool_idx[i_cpu][i_chunk]

                eta2_row = eta2_chunks[i_cpu][i_chunk, i_eta:]

                if i_cpu == self._num_processes-1 and i_eta >= num_fingerprints - 10:
                    print(i_cpu, i_eta, eta2_row)

                eta2_coef[i_eta, i_eta:] = eta2_row
                eta2_coef[i_eta:, i_eta] = eta2_row

        return eta2_coef

    @cython.boundscheck(False)
    @cython.cdivision(True)
    def _compute_similarity_map(self, numpy.ndarray[numpy.float32_t, ndim=2] fingerprints, numpy.ndarray[long] idx_chunk):
        """Compute the eta square coefficients from the voxels' fingerprints.

        The result is a symmetric square matrix where each element is a
        real value in the range 0.0 to 1.0 with 0 indicating the pair of
        voxels is completely dissimilar and 1 equal.

        Parameters
        ----------
        fingerprints : numpy.ndarray, shape (num_fingerprints, num_lower_dim)
            The correlations between the voxels and the lower dimensional
            embedding.

        idx_chunk : numpy.ndarray, shape (K, )
            Indexes of the num_fingerprints voxels to consider when computing
            the eta square coefficients. This parameter can be used to split
            the computation among several CPUs.

        Returns
        -------
        eta2_coef : numpy.ndarray, shape (K, num_fingerprints)
            The symmetric matrix containing the eta square coefficients between
            pairs of voxels where the voxel at row i has index idx_chunk[i] in
            fingerprints.

        """

        cdef:
            numpy.float64_t eta2_numerator, eta2_denominator
            numpy.float64_t fingerprints_ij_mean, fingerprints_ijk_mean
            numpy.float64_t fingerprints_i, fingerprints_j
            int num_fingerprints = fingerprints.shape[0]
            int num_lower_dim = fingerprints.shape[1]
            int i, j, k, i_chunk
            int progress, progress_old
            int num_chunk = idx_chunk.shape[0]
            numpy.ndarray[numpy.float32_t, ndim=2] eta2_coef = numpy.zeros((num_chunk, num_fingerprints), dtype=numpy.float32)

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

            if self._verbose and progress > progress_old:
                print("\rComputing similarity maps... {0}".format(progress), end="", flush=True)
                progress_old = progress

        return eta2_coef

    def _learn_embedding(self, eta2_coef):
        """ Learn a lower dimensional manifold from the space described by the
        eta square coefficients.

        Parameters
        ----------
        eta2_coef : numpy.ndarray, shape (num_fingerprints, num_fingerprints)
            The symmetric matrix containing the eta square coefficients
            between pairs of voxels.

        Returns
        -------
        embedding : numpy.ndarray, shape (num_fingerprints, manifold_components)
            The lower dimensional manifold.

        """

        # Transform correlations into euclidean distances
        distances = 2 * (1 - eta2_coef)

        if self._manifold_learning == 'tSNE':

            manifold_class = manifold.TSNE(n_components=self._manifold_components,
                                           perplexity=30.0,
                                           early_exaggeration=4.0,
                                           learning_rate=200.0,
                                           n_iter=1000,
                                           n_iter_without_progress=30,
                                           min_grad_norm=1e-07,
                                           metric='precomputed',
                                           init='random',
                                           verbose=self._verbose,
                                           random_state=0,
                                           method='exact')

        elif self._manifold_learning == 'isomap':

            manifold_class = manifold.Isomap(n_neighbors=5,
                                             n_components=self._manifold_components,
                                             eigen_solver='dense',
                                             neighbors_algorithm='brute')

        elif self._manifold_learning == 'spectral':

            # Binary search a similarity threshold, that is the minimum value
            # of the L2-norm between pair of voxels' similarities required
            # such that all the voxels in the ROI are connected.

            similarity_values = numpy.sort(distances, axis=None)
            high_index = eta2_coef.shape[0]**2 - 1
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

            # Disconnect distant voxels
            eta2_coef[~adjacency] = 0
            distances = eta2_coef

            manifold_class = manifold.SpectralEmbedding(n_components=self._manifold_components,
                                                        affinity='precomputed',
                                                        eigen_solver=None)

        return manifold_class.fit_transform(distances)
