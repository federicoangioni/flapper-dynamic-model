import numpy as np

class LP2Pfilter:
    def __init__(self, sample_freq, cutoff_freq):

        fr = sample_freq/cutoff_freq
        ohm = np.tan(np.pi/fr)
        c = 1.0+2.0*np.cos(np.pi / 4.0)*ohm + ohm*ohm
        self.b0 = ohm*ohm/c
        self.b1 = 2.0*self.b0
        self.b2 = self.b0
        self.a1 = 2.0 *(ohm*ohm -1.0)/c
        self.a2 = (1.0 - 2.0*np.cos(np.pi / 4.0)*ohm + ohm*ohm)/c
        self.delay_element_1 = 0.0
        self.delay_element_2 = 0.0

    def apply(self, sample):

        delay_element_0 = sample - self.delay_element_1 * self.a1 - self.delay_element_2 * self.a2

        if not np.isfinite(delay_element_0):
            delay_element_0 = sample
        
        output = delay_element_0 * self.b0 + self.delay_element_1 * self.b1 + self.delay_element_2 * self.b2

        self.delay_element_2 = self.delay_element_1
        self.delay_element_1 = delay_element_0

        return output
    