from scipy.signal import butter, filtfilt, lfilter
from LFPy_util.sims.SpikeWidthDef import SpikeWidthDef

class SpikeWidthDefFilt(SpikeWidthDef):
    def __init__(self):
        SpikeWidthDef.__init__(self)
        self.name = 'widthdeffilt'
        self.name_save_load = 'widthdef'

        self.simulate = True
        self.process_param['freq_low'] = 0.3 # kHz
        self.process_param['freq_high'] = 6.7 # kHz
        self.process_param['order'] = 2
        self.process_param['filter'] = 'filtfilt'

    def simulate(self, cell):
        if self.simulate == False:
            return
        SpikeWidthDef.simulate(self, cell)
        
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

        if self.process_param['filter'] == 'filtfilt':
            data['LFP'] = filtfilt(b, a, data['LFP'], axis=1)
        elif self.process_param['filter'] == 'lfilter':
            data['LFP'] = lfilter(b, a, data['LFP'], axis=1)
        else:
            raise ValueError("process_param['filter'] is not a valid string.")

        # Run the rest of the processing.
        SpikeWidthDef.process_data(self)
