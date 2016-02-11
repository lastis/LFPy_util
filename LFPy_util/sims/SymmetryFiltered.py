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
        super(Symmetry, self).__init__()
        self.name = "symf"

        self.run_param['low_cut'] = 0.8
        self.run_param['high_cut'] = 6.7
        
    def process_data(self):
        run_param = self.run_param
        data = self.data
        # Applying band pass filter.
        data['frec_sample'] = 1.0/run_param['dt'] # kHz
        b, a = butter_bandpass(run_param['low_cut'], run_param['high_cut'])
        LFP = filtfilt(b, a, data['LFP'], axis=1)
        data['LFP'] = LFP
