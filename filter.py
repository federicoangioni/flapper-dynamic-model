from scipy import signal
import numpy as np

class LP2Pfilter:
    def __init__(self, sample_freq, cutoff_freq):
        self.sample_freq = sample_freq
        self.cutoff_freq = cutoff_freq
        self._design_filter()

        self.z = np.zeros(max(len(self.a), len(self.b)) - 1)

    def _design_filter(self):
        nyquist = 0.5 * self.sample_freq
        normal_cutoff = self.cutoff_freq / nyquist

        # Design 2nd order Butterworth low-pass filter
        self.b, self.a = signal.butter(2, normal_cutoff, btype='low', analog=False)
    
    def apply(self, sample):

        output, self.z = signal.lfilter(self.b, self.a, [sample], zi=self.z)
        return output[0]