from Simulation import Simulation
import os
import LFPy
import LFPy_util
import LFPy_util.data_extraction as de
import LFPy_util.plot as lplot
import numpy as np
import matplotlib.pyplot as plt

class SphereElectrodes(Simulation):
    """docstring for Grid"""

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.ID = "sphere"

        self.debug = False
        self.run_param['N'] = 1000
        self.run_param['R'] = 100
        self.run_param['sigma'] = 0.3
        self.run_param['pre_dur'] = 16.7*0.5
        self.run_param['post_dur'] = 16.7*0.5
        self.run_param['threshold'] = 3
        self.run_param['amp_option'] = 'neg'

        self.width_def = 'type_I'
        self.bins = 9

    def __str__(self):
        return "Sphere Electrodes"

    def simulate(self,cell):
        data = self.data
        run_param = self.run_param
        # Calculate random numbers in a sphere.
        l = np.random.uniform(0,1,run_param['N'])
        u = np.random.uniform(-1,1,run_param['N'])
        phi = np.random.uniform(-1,1,run_param['N'])
        x = run_param['R']*np.power(l,1/3.0)*np.sqrt(1-u*u)*np.cos(phi),
        y = run_param['R']*np.power(l,1/3.0)*np.sqrt(1-u*u)*np.sin(phi),
        z = run_param['R']*np.power(l,1/3.0)*u

        cell.simulate(rec_imem=True)

        # Record the LFP of the electrodes. 
        electrode = LFPy.RecExtElectrode(cell,x=x,y=y,z=z,sigma=run_param['sigma'])
        electrode.calc_lfp()

        data['LFP'] = electrode.LFP
        data['elec_x'] = x
        data['elec_y'] = y
        data['elec_z'] = z
        data['t_vec'] = cell.tvec
        data['dt'] = cell.timeres_NEURON

    def process_data(self):
        data = self.data
        run_param = self.run_param
        t_vec = np.array(data['t_vec'])
        v_vec = np.array(data['LFP'])
        x = np.array(data['elec_x'])
        y = np.array(data['elec_y'])
        z = np.array(data['elec_z'])
        r = np.sqrt(x*x + y*y + z*z).flatten()
        spikes = []
        for row in xrange(v_vec.shape[0]):
            signal = v_vec[row]
            spike, spikes_t_vec, I = de.extract_spikes(
                    t_vec, 
                    signal, 
                    pre_dur=run_param['pre_dur'], 
                    post_dur=run_param['post_dur'],
                    threshold=run_param['threshold'],
                    amp_option=run_param['amp_option'],
                    )
            # Assume there is only one spike.
            spikes.append(spike[0])
        spikes = np.array(spikes)

        widths, widths_trace = de.find_wave_width_type_I(spikes, dt=data['dt'])
        bins = np.linspace(0,run_param['R'],self.bins,endpoint=True)
        inds = np.digitize(r,bins)
        widths_mean = np.empty(self.bins)
        widths_std = np.empty(self.bins)
        for bin_1 in xrange(len(bins)):
            widths_at_r = []
            for i, bin_2 in enumerate(inds):
                if bin_1 != bin_2: continue
                widths_at_r.append(widths[i])
            if len(widths_at_r) == 0:
                widths_mean[bin_1] = 0
                widths_std[bin_1] = 0
            widths_at_r = np.array(widths_at_r)
            widths_mean[bin_1] = np.mean(widths_at_r)
            widths_std[bin_1] = np.sqrt(np.var(widths_at_r))
            
        data['widths_mean'] = widths_mean
        data['widths_std'] = widths_std
        data['bins'] = bins
        data['elec_r'] = r
        data['spikes'] = spikes
        data['spikes_t_vec'] = spikes_t_vec

    def plot(self,dir_plot):
        data = self.data
        # Set global matplotlib parameters.
        LFPy_util.plot.set_rc_param()
        # Create the directory if it does not exist.
        if not os.path.exists(dir_plot):
            os.makedirs(dir_plot)

        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        # Disable spines.
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # Disable ticks.
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # Plot
        plt.plot(data['bins'], data['widths_mean'])
        # Save plt.
        fname = "sphere_spike_width"
        path = os.path.join(dir_plot,fname)
        plt.savefig(path,format='pdf',bbox_inches='tight')
        plt.close()












