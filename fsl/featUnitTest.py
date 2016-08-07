import unittest
import fsl
import fsl.feat
import numpy
from matplotlib import pyplot
from scipy import signal

class ConvolutionKernelTestCase(unittest.TestCase):

    # Test gamma PDF
    def test_gamma(self):

        # 100 samples spaced by 1 second
        ck = fsl.feat.ConvolutionKernel('gamma', 100, 1)
        self.assertEquals(len(ck.kernel), 100)
        self.assertAlmostEquals(ck.kernel[0], 0)
        self.assertAlmostEquals(ck.kernel[99], 0)

        # 100 samples spaced by 0.05 seconds
        ck = fsl.feat.ConvolutionKernel('gamma', 100, 20,
                                        sigma=1.4, lag=4)
        self.assertEqual(len(ck.kernel), 100)
        self.assertAlmostEqual(ck.kernel[0], 0, places=3)
        self.assertAlmostEqual(ck.kernel[20], 0.0063, places=3)
        self.assertAlmostEqual(ck.kernel[60], 0.2765, places=3)

        # 100 samples spaced by 0.1 seconds
        ck = fsl.feat.ConvolutionKernel('gamma', 100, 10,
                                        sigma=2, lag=6)
        self.assertEqual(len(ck.kernel), 100)
        self.assertAlmostEqual(ck.kernel[0], 0, places=3)
        self.assertAlmostEqual(ck.kernel[30], 0.0695, places=3)
        self.assertAlmostEqual(ck.kernel[80], 0.0983, places=3)

    # Test double gamma PDF
    def test_double_gamma(self):

        # 200 samples spaced by 0.2 seconds
        ck = fsl.feat.ConvolutionKernel('double_gamma', 200, 5)
        self.assertEquals(len(ck.kernel), 200)
        self.assertAlmostEquals(ck.kernel[0], 0, places=3)
        self.assertAlmostEquals(ck.kernel[30], 0.1605, places=3)
        self.assertAlmostEquals(ck.kernel[50], 0.0320, places=3)
        self.assertAlmostEquals(ck.kernel[130], -0.0011, places=3)
        self.assertAlmostEquals(ck.kernel[199], 0, places=3)


class StimulusTestCase(unittest.TestCase):

    # Test class constructor
    def setUp(self):
        self.stimulus = fsl.feat.Stimulus(shape='custom3',
                                          sampling_frequency=5,
                                          filename='/Users/michele/Documents/Workspaces/UpWork/Morgan_Hough/fmri/fmri_fluency/word_generation.txt')

    # Test time-series
    def test_custom3(self):
        self.assertGreater(len(self.stimulus.waveform), 0)

    # Test convolution with gamma PDF
    def test_gamma_convolve(self):
        ck = fsl.feat.ConvolutionKernel('gamma', 100, 5)
        len_stimulus_before = len(self.stimulus.waveform)
        self.stimulus.convolve(ck)
        self.assertEquals(len(self.stimulus.waveform),
                          len_stimulus_before + len(ck.kernel) - 1)
        self.assertAlmostEquals(self.stimulus.waveform[95], 0,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[297], 0.2104,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[311], 0.0684,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[1145], 0,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[1250], 0,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[1817], 0.7942,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[1892], 0.1537,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[1976], 0,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[2053], 0.1343,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[2083], 0.5057,
                                places=3)

    # Test convolution with double gamma PDF
    def test_double_gamma_convolve(self):
        ck = fsl.feat.ConvolutionKernel('double_gamma', 150, 5)
        len_stimulus_before = len(self.stimulus.waveform)
        self.stimulus.convolve(ck)
        self.assertEquals(len(self.stimulus.waveform),
                          len_stimulus_before + len(ck.kernel) - 1)
        self.assertAlmostEquals(self.stimulus.waveform[1], 0,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[215], 0,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[248], 0.0004,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[364], 0.8431,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[664], 0.0399,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[666], 0.0152,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[693], -0.0714,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[720], -0.0200,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[944], -0.0563,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[1395], 0.5739,
                                places=3)

    # Test high pass filter
    def test_highpass_temporal_filter_sigma_1(self):
        ck = fsl.feat.ConvolutionKernel('double_gamma', 150, 5)
        self.stimulus.convolve(ck)

        len_stimulus = len(self.stimulus.waveform)

        self.stimulus.highpass_temporal_filter(1.0, False)

        len_stimulus_filtered = len(self.stimulus.waveform)

        self.assertEquals(len_stimulus, len_stimulus_filtered)
        self.assertAlmostEquals(self.stimulus.waveform[103], 0.0,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[432], -0.0503,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[589], 0.0009,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[1443], -0.0151,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[1495], 0.0010,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[1497], 0.0010,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[1562], -0.0018,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[1816], -0.0803,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[1954], 0.0007,
                                places=3)
        self.assertAlmostEquals(self.stimulus.waveform[2007], 0.0,
                                places=3)

    # Test high pass filter
    def test_highpass_temporal_filter_sigma_10(self):
        ck = fsl.feat.ConvolutionKernel('double_gamma', 150, 5)
        self.stimulus.convolve(ck)

        len_stimulus = len(self.stimulus.waveform)

        pyplot.figure(1)
        t = numpy.arange(0, len(self.stimulus.waveform))
        pyplot.plot(t, self.stimulus.waveform, 'k-')

        self.stimulus.highpass_temporal_filter(10.0)

        pyplot.plot(t, self.stimulus.waveform, 'r-')

        # Nilearn filter
        b, a = signal.cheby2(5, 40, 0.05, btype='highpass')
        output = signal.filtfilt(b, a, self.stimulus.waveform)
        pyplot.plot(t, output, 'b-')
        pyplot.legend(("x", "HP(x)", "B(x)"))
        pyplot.grid(True)
        pyplot.show()

        len_stimulus_filtered = len(self.stimulus.waveform)

        self.assertEquals(len_stimulus, len_stimulus_filtered)
        self.assertAlmostEquals(self.stimulus.waveform[58], -0.0017,
                                places=2)
        self.assertAlmostEquals(self.stimulus.waveform[202], -0.0655,
                                places=2)
        self.assertAlmostEquals(self.stimulus.waveform[270], 0.6450,
                                places=2)
        self.assertAlmostEquals(self.stimulus.waveform[365], 0.4891,
                                places=2)
        self.assertAlmostEquals(self.stimulus.waveform[491], 0.0747,
                                places=2)
        self.assertAlmostEquals(self.stimulus.waveform[1006], -0.1096,
                                places=2)
        self.assertAlmostEquals(self.stimulus.waveform[1039], 0.1376,
                                places=2)
        self.assertAlmostEquals(self.stimulus.waveform[1541], -0.1006,
                                places=2)
        self.assertAlmostEquals(self.stimulus.waveform[1627], -0.1338,
                                places=2)

    # Test low pass filter
    def test_lowpass_temporal_filter(self):
        ck = fsl.feat.ConvolutionKernel('double_gamma', 150, 5)
        self.stimulus.convolve(ck)

        len_stimulus = len(self.stimulus.waveform)

        self.stimulus.lowpass_temporal_filter()

        len_stimulus_filtered = len(self.stimulus.waveform)

        self.assertEquals(len_stimulus, len_stimulus_filtered)

    # Test low pass filter
    def test_temporal_derivative(self):
        ck = fsl.feat.ConvolutionKernel('double_gamma', 150, 5)
        self.stimulus.convolve(ck)

        len_stimulus = len(self.stimulus.waveform)

        derivative_stimulus = self.stimulus.temporal_derivative()

        len_derivative_stimulus = len(derivative_stimulus)

        self.assertEquals(len_stimulus, len_derivative_stimulus)


convolutionKernelSuite = unittest.TestLoader().loadTestsFromTestCase(
    ConvolutionKernelTestCase)
stimulusSuite = unittest.TestLoader().loadTestsFromTestCase(
    StimulusTestCase)
unittest.TextTestRunner(verbosity=2).run(convolutionKernelSuite)
unittest.TextTestRunner(verbosity=2).run(stimulusSuite)

