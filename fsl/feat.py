"""
FMRI data processing carried out using FEAT (FMRI Expert Analysis Tool).
Time-series statistical analysis was carried out using FILM with local
autocorrelation correction [Woolrich 2001].

References
[Woolrich 2001] M.W. Woolrich, B.D. Ripley, J.M. Brady and S.M. Smith.
Temporal Autocorrelation in Univariate Linear Modelling of FMRI Data.
NeuroImage 14:6(1370-1386) 2001.
"""

# Author: Michele Damian (michele.damian@gmail.com)

import os
import numpy
import scipy
import scipy.stats
import nilearn
from matplotlib import pyplot


def stats(nifti_image, glm, prewhitening=True):
    """ Perform statistical analysis on a Nifti 4D image. The data must
        have already been preprocessed; i.e. motion correction,
        spatially/temporally smoothing, registration to high-resolution
        images, etc.. must have already been performed.


    Parameters
    ==========
    nifti_image: nilearn.image
        4D nifti image on which to compute the statistics.

    glm: collections.namedtuple
        General linear model that explains the time-series in the
        nifti_image.

    prewhitening: bool [True]
        Specifies if to use FILM prewhitening to make the statistics
        valid and maximally efficient.


    Returns
    =======
    stats: collections.namedtuple
        A tuple, where each entry represents a contrast in the GLM model
        and the corresponding nifti_image with the per-voxel statistics.


    Note
    ====
    So far just the custom 3-columns shape GLM is supported.

    """

    pass
    # TODO put this methods somewhere else


class ConvolutionKernel:
    """ This class is a factory for creating a kernel to convolve
    a stimulus by the method feat.Stimulus.convolve() as represented.
    The supported functions are "gamma" and "double_gamma". For the
    parameters accepted by each the functions refer to the
    documentation of the class methods _create_*().
    """

    supported_kernels = ["double_gamma", "gamma"]

    def __init__(self, kernel_type, n_points,
                 sampling_frequency, **kwargs):
        """
        Constructor for the specified kernel.

        Parameters
        ----------
        kernel_type : str
            Type of kernel to convolve with the stimulus with. Accepted
            kernels are "HRF" and "gamma".

        n_points : int
            Number of points at which the function is sampled. By
            default each point corresponds to a second. See
            sampling_frequency to change this behavior.

        sampling_frequency : float (default: 1.0)
            Number of points to use for each second. For example, with
            sampling_frequency=4 the interval between each point at
            which the pdf is sampled to build the kernel is 1/4th of a
            second.

        **kargs :
            Parameters of the function used to create the kernel.
        """

        if kernel_type not in ConvolutionKernel.supported_kernels:
            raise ValueError("kernel must be one of %s" %
                             ConvolutionKernel.supported_kernels)

        self.kernel_type = kernel_type
        self.sampling_frequency = sampling_frequency

        x = numpy.linspace(0, n_points-1, n_points) / sampling_frequency

        if kernel_type == "double_gamma":
            self.kernel = \
                ConvolutionKernel._create_double_gamma(x, **kwargs)
        elif kernel_type == "gamma":
            self.kernel = ConvolutionKernel._create_gamma(x, **kwargs)

    @classmethod
    def _create_double_gamma(cls, x, sigma1=2.449, sigma2=4, lag1=6,
                             lag2=16, ratio=6):
        """ Create a double-gamma kernel.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_points, )
            Values at which to compute the pdf.

        sigma1 : float (default: 2.449)
            Half-width of the first gamma smoothing of the input
            waveform.

        sigma2 : float (default: 4)
            Half-width of the second gamma smoothing of the input
            waveform.

        lag1 : float (default: 6)
            Mean lag of the first gamma smoothing of the input waveform.

        lag2 : float (default: 16)
            Mean lag of the second gamma smoothing of the input
            waveform.

        ratio : float (default: 6)
            HRF = PDF(gamma1) - PDF(gamma2) / ratio;

        Returns
        -------
        kernel : numpy.ndarray, shape (n_points, )
            A HRF kernel.
        """

        kernel = ConvolutionKernel._create_gamma(x, sigma1, lag1) - \
            ConvolutionKernel._create_gamma(x, sigma2, lag2) / ratio

        return kernel

    @classmethod
    def _create_gamma(cls, x, sigma=3, lag=6):
        """ Create a gamma function kernel.

        Parameters
        ----------
        x : numpy.ndarray, shape (n_points, )
            Values at which to compute the pdf.

        sigma : float (default: 3)
            Half-width of the gamma smoothing of the input waveform.

        lag : float (default : 6)
            Mean lag of the gamma smoothing of the input waveform.

        Returns
        -------
        kernel : numpy.ndarray, shape (n_points, )
            A gamma kernel.
        """

        a = lag ** 2 / sigma ** 2
        b = sigma ** 2 / lag
        kernel = scipy.stats.gamma.pdf(x, a, scale=b)

        return kernel


class Stimulus:
    """ This is the input stimulus waveform presented to the subject
    during the experiment. Each instance of the class Stimulus is a
    time-series representing a defined event.

    Example: a language-related experiment where the subject is
    presented two tasks (e.g. word-generation and word-shadowing) as
    input will be represented programmatically by two instances of
    the class Stimulus.

    Parameters
    ----------
    shape : str, ('custom', 'custom3')
        Shape of the waveform representing the stimulus.

        'custom': loads from the text file pointed by filename a
        time-series as a list of numbers, separated by spaces or
        newlines, with one number for each volume (after subtracting
        the number of deleted images). These numbers can either all
        be 0s and 1s, or can take a range of values. The former case
        would be appropriate if the same stimulus was applied at
        varying time points; the latter would be appropriate,
        for example, if recorded subject responses are to be inserted
        as an effect to be modelled. Note that it may or may not be
        appropriate to convolve this particular waveform with an HRF
        - in the case of single-event, it is.

        'custom3': loads from the text file pointed by filename a
        time series as a each triplet describes a short period of
        time and the value of the model during that time. The first
        number in each triplet is the onset (in seconds) of the
        period, the second number is the duration (in seconds) of
        the period, and the third number is the value of the input
        during that period. The same comments as above apply, about
        whether these numbers are 0s and 1s, or vary continuously.
        The start of the first non-deleted volume corresponds to t=0.

    sampling_frequency : float (default: 1.0)
        Number of points to use for each second. For example,
        with sampling_frequency=4 each point at which the pdf is sampled
        to build the kernel corresponds to 1/4th of a second.

    filename : str
        Full path to the text file. Use if shape='custom' or 'custom3'.

    Attributes
    ----------
    waveform : numpy.ndarray, shape(n_events*sampling_frequency, )
        Value of the input waveform for each point. Each point lasts
        for 1s / sampling_frequency.
    """

    def __init__(self, shape, sampling_frequency=1.0, **kwargs):

        # Check input
        if shape not in ['custom', 'custom3']:
            raise ValueError("This shape is not defined")

        elif shape == 'custom' and kwargs.get('filename') is None:
            raise ValueError("filename attribute must be defined for "
                             "shape='custom'")

        elif shape == 'custom3' and kwargs.get('filename') is None:
            raise ValueError("filename attribute must be defined for "
                             "shape='custom3'")

        elif shape in ['custom', 'custom3'] and not \
                os.path.exists(kwargs['filename']):
            raise ValueError('%s not found' % kwargs['filename'])

        elif sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be positive")

        self.sampling_frequency = sampling_frequency

        if shape == 'custom':
            self.waveform = Stimulus._create_custom(kwargs['filename'],
                                                    sampling_frequency)
        elif shape == 'custom3':
            self.waveform = Stimulus._create_custom3(kwargs['filename'],
                                                     sampling_frequency)

    @classmethod
    def _create_custom(cls, filename, sampling_frequency):

        with open(filename, 'rt') as file:
            tokens = numpy.array([float(token) for token in
                                  [line.split() for line in file]])

        if not tokens:
            raise ValueError("%s is empty")

        waveform = numpy.repeat(tokens, sampling_frequency)

        return waveform

    @classmethod
    def _create_custom3(cls, filename, sampling_frequency):

        with open(filename, 'rt') as file:
            tokens = [float(token) for line in file
                      for token in line.split()]

        if not tokens:
            raise ValueError("%s is empty", filename)

        tokens = numpy.asarray(tokens)

        onset = [float(token) for token in tokens[0::3]]
        period = [float(token) for token in tokens[1::3]]
        value = [float(token) for token in tokens[2::3]]

        n_points = numpy.ceil((onset[-1]+period[-1]) *
                              sampling_frequency)
        waveform = numpy.zeros(n_points)

        for i in range(0, len(onset)):
            start = numpy.round(onset[i] * sampling_frequency)
            stop = numpy.round((onset[i] + period[i]) *
                               sampling_frequency)
            waveform[start:stop] = value[i]

        return waveform

    def convolve(self, kernel):
        """ Convolve the stimulus waveform represented by this class
        with the kernel specified. As result this class will
        represent the convolution and the length of its waveform will
        be equal to the length of the original stimulus and the
        length of the kernel.

        Parameters
        ----------
        kernel : feat.ConvolutionKernel
            A kernel to convolve this stimulus with. Stimulus and
            kernel should have the same sampling frequency (i.e. at an
            equal increment in the index of stimulus.waveform and
            convolution_kernel.kernel should correspond the same time
            lapse).

        """

        self.waveform = numpy.convolve(self.waveform, kernel.kernel)

    def phase(self, phase):
        """ Phase the stimulus waveform. Positive values shifts the
        time-series earlier in time.

        Parameters
        ----------
        phase : int
            The phase in seconds.
        """

        if phase >= 0:
            self.waveform = self.waveform[phase:]
        else:
            phase = -phase
            self.waveform = numpy.insert(self.waveform, 0, [0]*phase)

    def highpass_temporal_filter(self, sigma, demean=True):
        """ Filter the stimulus with a high pass gaussian-weighted
        least-squares straight line fitting temporal filter. The
        standard deviation is represented by sigma.

        Parameters
        ----------
        sigma : float
            Sigma of the gaussian function that filters the stimulus
            in seconds.

        demean : bool (default: True)
            True if the resulting stimulus must have mean equal to 0.
        """

        self.waveform = Utils.high_pass_temporal_filter(
                        self.waveform, sigma, self.sampling_frequency)

        if demean:
            self.waveform -= numpy.mean(self.waveform)

    def lowpass_temporal_filter(self, sigma=2.8):
        """ Filter the stimulus with a low pass gaussian temporal
        filter which standard deviation is represented by sigma.

        Parameters
        ----------
        sigma : float (default: 2.8)
            Sigma of the gaussian function that filters the stimulus.
        """

        _, kernel = Utils.gaussian_kernel(sigma,
                                          20,
                                          self.sampling_frequency)

        kernel_half_width = numpy.int(len(kernel) / 2)

        len_waveform = len(self.waveform)

        temp_waveform = numpy.zeros(len_waveform)

        sum_w = numpy.sum(kernel)

        for idx in range(0, len_waveform):

            idx_waveform = numpy.arange(max(0,
                                            idx-kernel_half_width),
                                        min(len_waveform,
                                            idx+kernel_half_width+1),
                                        dtype=numpy.int)

            x = self.waveform[idx_waveform]

            # Signal padding
            if idx < kernel_half_width:
                x = numpy.lib.pad(x, (kernel_half_width - idx, 0),
                                  mode='constant',
                                  constant_values=x[0])

            if idx > len_waveform - kernel_half_width - 1:
                x = numpy.lib.pad(x,
                                  (0, idx + kernel_half_width + 1 -
                                      len_waveform),
                                  mode='constant',
                                  constant_values=x[-1])

            temp_waveform[idx] = numpy.sum(kernel * x)

        self.waveform = temp_waveform / sum_w

    def temporal_derivative(self):
        """ Returns the temporal derivative of this model.
        Subtracting the result of this method from the original
        waveform delays the waveform by 1 temporal unit (i.e. 1 /
        sampling_frequency seconds)

        Returns
        -------
        temp_deriv : numpy.ndarray
            The temporal derivative of self.waveform.
        """

        x = self.waveform
        x_before = numpy.append(x[0], x[:-1])
        x_after = numpy.append(x[1:], x[-1])

        return 0.5 * (x_after - x_before)


class NiftiImage:
    """

    """

    def __init__(self, filename, spatial_smoothing=5.0):
        """

        Parameters
        ----------
        filename : str
            Full path of the NIFTI image to load.

        spatial_smoothing : float (default: 5.0)
            Spatial smoothing is carried out on each volume of the
            FMRI data set separately. This is intended to reduce
            noise without reducing valid activation; this is
            successful as long as the underlying activation area is
            larger than the extent of the smoothing. Thus if you are
            looking for very small activation areas then you should
            maybe reduce smoothing from the default of 5mm, and if
            you are looking for larger areas, you can increase it,
            maybe to 10 or even 15mm. To turn off spatial smoothing
            simply set this parameter spatial_smoothing=None.
        """

        self.nifti = nilearn.image.smooth_img(filename,
                                              spatial_smoothing)

    def highpass_temporal_filter(self, sigma, demean=False):

        nifti_data = self.nifti.get_data()

        for i in range(0, nifti_data.shape(0)):
            for j in range(0, nifti_data.shape(1)):
                for k in range(0, nifti_data.shape(2)):
                    nifti_data(i, j, k, :) =
                    Utils.high_pass_temporal_filter(nifti_data(i, j, k, :),
                                                    sigma,
                                                    )

        pass


class Utils:

    @classmethod
    def high_pass_temporal_filter(cls,
                                  signal,
                                  sigma,
                                  sampling_frequency):
        """ Filter the signal with a high pass gaussian-weighted
        least-squares straight line fitting temporal filter. The
        standard deviation is represented by sigma in seconds.

        Parameters
        ----------
        signal : numpy.ndarray
            Signal to be filtered.

        sigma : float
            Sigma of the gaussian function that filters the stimulus
            in seconds.

        sampling_frequency : float
            Frequency at which to quantize the gaussian function.
        """

        t_kernel, kernel = Utils.gaussian_kernel(sigma,
                                                     3,
                                                     sampling_frequency)

        kernel_half_width = numpy.int(len(kernel) / 2)

        len_signal = len(signal)

        sum_w = numpy.sum(kernel)
        sum_w_dt = numpy.sum(kernel * t_kernel)
        sum_w_dt2 = numpy.sum(kernel * (t_kernel**2))

        numerator = numpy.zeros(len_signal)
        denominator = sum_w_dt2 * sum_w - sum_w_dt**2

        if denominator == 0:
            return signal

        for idx in range(0, len_signal):

            idx_signal = numpy.arange(max(0,
                                          idx-kernel_half_width),
                                      min(len_signal,
                                          idx+kernel_half_width+1),
                                      dtype=numpy.int)

            x = signal[idx_signal]

            # Signal padding
            if idx < kernel_half_width:
                x = numpy.lib.pad(x,
                                  (kernel_half_width - idx, 0),
                                  mode='constant',
                                  constant_values=x[0])

            if idx > len_signal - kernel_half_width - 1:
                x = numpy.lib.pad(x,
                                  (0, idx + kernel_half_width + 1 -
                                   len_signal),
                                  mode='constant',
                                  constant_values=x[-1])

            sum_w_x = numpy.sum(kernel * x)
            sum_w_x_dt = numpy.sum(kernel * x * t_kernel)

            numerator[idx] = sum_w_x * sum_w_dt2 - \
                             sum_w_x_dt * sum_w_dt

        result = numerator / denominator

        return signal + result[0] - result

    @classmethod
    def gaussian_kernel(cls,
                        sigma,
                        sigma_multiplier,
                        sampling_frequency):
        """ Helper method to build a Gaussian kernel with the
        same sampling frequency of the stimulus.

        Parameters
        ----------
        sigma : float
            Standard deviation of the gaussian function.

        sigma_multiplier : float
            Number of standard deviations from the mean to include
            in the output kernel.

        sampling_frequency : float
            Frequency at which to quantize the gaussian function.

        Returns
        -------
        t : numpy.ndarray
            Time in seconds correspondent to each index of the
            kernel. The length of the kernel is always odd and
            the median index correspond to time=0.

        kernel : numpy.ndarray
            Quantized gaussian function.
        """

        # floor(sampling_frequency) is the same behavior of the stimulus
        # loader -- _create_custom and _create_custom3

        # Kernel half size
        # Trunc kernel max value at 3 * std
        kernel_n_points = numpy.round(sigma *
                                      sigma_multiplier *
                                      sampling_frequency)

        kernel_max_val = kernel_n_points / sampling_frequency

        t = numpy.linspace(-kernel_max_val,
                           kernel_max_val,
                           2*kernel_n_points + 1)

        kernel = numpy.exp(-(t**2) / (2 * (sigma**2)))

        return t, kernel


""" Run FEAT script
"""
if __name__ == "__main__":

    pass
