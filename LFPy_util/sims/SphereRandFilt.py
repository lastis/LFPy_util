from scipy.signal import butter, filtfilt
from LFPy_util.sims.SphereRand import SphereRand

class SphereRandFilt(SphereRand):
    """docstring for SphereRandFilt"""
    def __init__(self):
        SphereRand.__init__(self)
        self.name = "spheref"

        self.process_param['freq_low'] = 0.3 # kHz
        self.process_param['freq_high'] = 6.7 # kHz
        self.process_param['order'] = 2
        
    def process_data(self):
        run_param = self.run_param
        data = self.data
        process_param = self.process_param
        # Applying band pass filter.
        data['freq_sample'] = 1.0/data['dt'] # kHz

        nyq = 0.5 * data['freq_sample']
        low = process_param['freq_low'] / nyq 
        high = process_param['freq_high'] / nyq
        b, a = butter(process_param['order'], [low, high], btype='band')

        data['LFP'] = filtfilt(b, a, data['LFP'], axis=1)

        # Run the rest of the processing.
        SphereRand.process_data(self)
