from scipy.signal import butter, filtfilt, lfilter
from LFPy_util.sims.Symmetry import Symmetry

class SymmetryFiltered(Symmetry):
    """docstring for SymmetryFiltered"""
    def __init__(self):
        Symmetry.__init__(self)
        self.name = "symf"

        self.process_param['freq_low'] = 0.3 # kHz
        self.process_param['freq_high'] = 6.7 # kHz
        self.process_param['order'] = 2
        
    def process_data(self):
        run_param = self.run_param
        data = self.data
        process_param = self.process_param
        # Applying band pass filter.
        data['freq_sample'] = 1.0/data['dt']*1000 # kHz

        nyq = 0.5 * data['freq_sample']
        low = process_param['freq_low'] / nyq * 1000
        high = process_param['freq_high'] / nyq * 1000
        b, a = butter(process_param['order'], [low, high], btype='band')

        # data['LFP'] = filtfilt(b, a, data['LFP'], axis=1)
        data['LFP'] = lfilter(b, a, data['LFP'], axis=1)

        # Run the rest of the processing.
        Symmetry.process_data(self)
