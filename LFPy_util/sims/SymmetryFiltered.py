from scipy.signal import butter, filtfilt
from LFPy_util.sims.Symmetry import Symmetry

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

class SymmetryFiltered(Symmetry):
    """docstring for SymmetryFiltered"""
    def __init__(self):
        Symmetry.__init__(self)
        self.name = "symf"

        self.process_param['low_cut'] = 0.8
        self.process_param['high_cut'] = 6.7
        
    def process_data(self):
        run_param = self.run_param
        data = self.data
        process_param = self.process_param
        # Applying band pass filter.
        data['frec_sample'] = 1.0/data['dt'] # kHz
        b, a = butter_bandpass(
            process_param['low_cut']*1000, 
            process_param['high_cut']*1000, 
            data['frec_sample']*1000
            )
        data['LFP'] = filtfilt(b, a, data['LFP'], axis=1)

        Symmetry.process_data(self)
