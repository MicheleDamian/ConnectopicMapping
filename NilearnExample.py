import nilearn
import numpy
from nilearn import image
from nilearn import plotting
from scipy import stats
from matplotlib import pyplot


#
# Basic Nilearn example to display slice
#

FMRI_PATH = '/Users/michele/Documents/Workspaces/UpWork/Morgan_Hough/fmri/fmri_fluency/fmri.nii.gz'

#
# Load and smooth data
#

# 1) If FWHM == None remove infinites
# fmri_4D = image.smooth_img(NIFTI_PATH, None);
# 2) If FWHM == ndarray use those FWHM along each axis
# fmri_4D = image.smooth_img(NIFTI_PATH, [5 5 7]);
# 3) If FWHM == 'fast' use kernel [0.2, 0.1, 0.2]
# fmri_4D = image.smooth_img(NIFTI_PATH, 'fast');
# 4) If FWHM == scalar use that FWHM
fmri_4D = image.smooth_img(FMRI_PATH, 5);

#
# Data info
#

# Data
nifti_data = fmri_4D.get_data()
print('Data shape: %s' % str(nifti_data.shape))

# Affine matrix
nifti_affine = fmri_4D.get_affine()
print('Affine: \n%s' % nifti_affine)

# Header
nifti_header = fmri_4D.get_header()
print('Header: \n%s' % nifti_header)

# TR
TR = nifti_header['pixdim'][4]

print('TR:', TR)

#
# Slice image
#

# Get first TR of a 4D image
nifti_TR0 = image.index_img(fmri_4D, 0)

# Mean of a 4D image along the time dimension
nifti_mean = image.mean_img(fmri_4D)

# Plot 3D image at coords (0, 0, 0)
cut_coords = (0, 0, 0)
plotting.plot_anat(nifti_TR0, cut_coords=cut_coords, title='Slice TR=0')
plotting.plot_anat(nifti_mean, cut_coords=cut_coords, title='Mean image')
plotting.show()

# Plot middle voxel timeserie
voxel_coords = ()
timeseries = nifti_data[42,52,31,:]
timeseries_filtered = nilearn.signal.clean(timeseries,
                                           detrend=False,
                                           standardize=False,
                                           high_pass=0.1,
                                           t_r=TR)
pyplot.plot(numpy.arange(0, len(timeseries)), timeseries, 'r-')
pyplot.plot(numpy.arange(0, len(timeseries_filtered)), timeseries_filtered,
            'b-')
pyplot.legend(("Original", "Highpass Filtered"))
pyplot.grid()
pyplot.show()


# ## Get t stats
# _, p_values = stats.ttest_ind(nifti_data[..., haxby_labels == b'face'],
#                               nifti_data[..., haxby_labels == b'house'],
#                               axis=-1)
#
# # Use a log scale for p-values
# log_p_values = -np.log10(p_values)
# log_p_values[np.isnan(log_p_values)] = 0.
# log_p_values[log_p_values > 10.] = 10.
# from nilearn.plotting import plot_stat_map
# plot_stat_map(new_img_like(fmri_img, log_p_values),
#               mean_img, title="p-values", cut_coords=cut_coords)

print(nifti_data.shape)