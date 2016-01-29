import os
import numpy as np
import LFPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import LFPy_util
import LFPy_util.data_extraction as de
import LFPy_util.plot as lplot
from LFPy_util.sims.Simulation import Simulation


class SphereElectrodes(Simulation):
    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.name = "sphere"

        self.debug = False
        self.run_param['N'] = 500
        self.run_param['R'] = 50
        self.run_param['sigma'] = 0.3

        self.amp_option = 'both'
        self.pre_dur = 16.7 * 0.5
        self.post_dur = 16.7 * 0.5
        self.threshold = 3
        self.elec_to_plot = []
        self.bins = 11
        self.amp_threshold = 0  # uV

    def __str__(self):
        return "Sphere Electrodes"

    def simulate(self, cell):
        data = self.data
        run_param = self.run_param
        # Calculate random numbers in a sphere.
        l = np.random.uniform(0, 1, run_param['N'])
        u = np.random.uniform(-1, 1, run_param['N'])
        phi = np.random.uniform(0, 2 * np.pi, run_param['N'])
        x = run_param['R'] * np.power(l, 1 /
                                      3.0) * np.sqrt(1 - u * u) * np.cos(phi),
        y = run_param['R'] * np.power(l, 1 /
                                      3.0) * np.sqrt(1 - u * u) * np.sin(phi),
        z = run_param['R'] * np.power(l, 1 / 3.0) * u

        cell.simulate(rec_imem=True)

        # Record the LFP of the electrodes.
        electrode = LFPy.RecExtElectrode(cell,
                                         x=x,
                                         y=y,
                                         z=z,
                                         sigma=run_param['sigma'])
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
        # Calculate the radial distance to the electrodes.
        r = np.sqrt(x * x + y * y + z * z).flatten()
        r_all = r
        # Find the first spike of each electrode.
        spikes = []
        deleted_elem = []
        for row in xrange(v_vec.shape[0]):
            signal = v_vec[row]
            # If the amplitude is less than what can be "recorded", skip it.
            if np.fabs(signal).max() * 1000 < self.amp_threshold:
                deleted_elem.append(row)
                continue
            spike, spikes_t_vec, I = de.extract_spikes(
                t_vec,
                signal,
                pre_dur=self.pre_dur,
                post_dur=self.post_dur,
                threshold=self.threshold,
                amp_option=self.amp_option, )
            # Assume there is only one spike.
            spikes.append(spike[0])
        spikes = np.array(spikes)
        r = np.delete(r, deleted_elem)

        # Find widths of the spikes, trace can be used for plotting.
        widths_I, widths_I_trace = de.find_wave_width_type_I(spikes,
                                                             dt=data['dt'])
        widths_II, widths_II_trace = de.find_wave_width_type_II(
            spikes,
            dt=data['dt'],
            amp_option=self.amp_option)
        amps_I = de.find_amplitude_type_I(spikes, amp_option=self.amp_option)
        amps_II = de.find_amplitude_type_II(spikes)
        # Put widths_I in bins decided by the radial distance. 
        # Then calculate std and mean.
        bins = np.linspace(0, run_param['R'], self.bins, endpoint=True)
        inds = np.digitize(r, bins)
        widths_I_mean = np.empty(self.bins)
        widths_I_std = np.empty(self.bins)
        widths_II_mean = np.empty(self.bins)
        widths_II_std = np.empty(self.bins)
        amps_I_mean = np.empty(self.bins)
        amps_I_std = np.empty(self.bins)
        amps_II_mean = np.empty(self.bins)
        amps_II_std = np.empty(self.bins)
        for bin_1 in xrange(len(bins)):
            widths_I_at_r = []
            widths_II_at_r = []
            amps_I_at_r = []
            amps_II_at_r = []
            for i, bin_2 in enumerate(inds):
                if bin_1 != bin_2: continue
                widths_I_at_r.append(widths_I[i])
                widths_II_at_r.append(widths_II[i])
                amps_I_at_r.append(amps_I[i])
                amps_II_at_r.append(amps_II[i])
            if len(widths_I_at_r) == 0:
                widths_I_mean[bin_1] = np.nan
                widths_I_std[bin_1] = np.nan
                widths_II_mean[bin_1] = np.nan
                widths_II_std[bin_1] = np.nan
                amps_I_mean[bin_1] = np.nan
                amps_I_std[bin_1] = np.nan
                amps_II_mean[bin_1] = np.nan
                amps_II_std[bin_1] = np.nan
            else:
                widths_I_at_r = np.array(widths_I_at_r)
                widths_I_mean[bin_1] = np.mean(widths_I_at_r)
                widths_I_std[bin_1] = np.sqrt(np.var(widths_I_at_r))
                widths_II_at_r = np.array(widths_II_at_r)
                widths_II_mean[bin_1] = np.mean(widths_II_at_r)
                widths_II_std[bin_1] = np.sqrt(np.var(widths_II_at_r))
                amps_I_mean[bin_1] = np.mean(amps_I_at_r)
                amps_I_std[bin_1] = np.sqrt(np.var(amps_I_at_r))
                amps_II_mean[bin_1] = np.mean(amps_II_at_r)
                amps_II_std[bin_1] = np.sqrt(np.var(amps_II_at_r))

        data['elec_r_all'] = r_all

        data['amps_I_mean'] = amps_I_mean * 1000
        data['amps_I_std'] = amps_I_std * 1000
        data['amps_I'] = amps_I * 1000

        data['amps_II_mean'] = amps_II_mean * 1000
        data['amps_II_std'] = amps_II_std * 1000
        data['amps_II'] = amps_II * 1000

        data['widths_I_mean'] = widths_I_mean
        data['widths_I_std'] = widths_I_std
        data['widths_I'] = widths_I
        data['widths_I_trace'] = widths_I_trace * 1000

        data['widths_II_mean'] = widths_II_mean
        data['widths_II_std'] = widths_II_std
        data['widths_II'] = widths_II
        data['widths_II_trace'] = widths_II_trace * 1000

        data['bins'] = bins
        data['elec_r'] = r
        data['spikes'] = spikes * 1000
        data['spikes_t_vec'] = spikes_t_vec

    def plot(self, dir_plot):
        data = self.data
        run_param = self.run_param
        format = 'pdf'
        # Set global matplotlib parameters.
        LFPy_util.plot.set_rc_param()
        # Create the directory if it does not exist.
        if not os.path.exists(dir_plot):
            os.makedirs(dir_plot)

        # 3D plot.
        fname = "sphere_elec_pos"
        c = lplot.get_short_color_array(5)[2]
        fig = plt.figure(figsize=lplot.size_common)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data['elec_x'], data['elec_y'], data['elec_z'], c=c)
        ax.set_xlim(-run_param['R'], run_param['R'])
        ax.set_ylim(-run_param['R'], run_param['R'])
        ax.set_zlim(-run_param['R'], run_param['R'])
        # Save plt.
        path = os.path.join(dir_plot, fname + "." + format)
        plt.savefig(path, format=format, bbox_inches='tight')
        plt.close()

        # New plot.
        fname = 'sphere_spike_amps_I'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['bins'],
                 data['amps_I_mean'],
                 color=lplot.color_array_long[0],
                 marker='o',
                 markersize=5)
        ax.fill_between(data['bins'],
                        data['amps_I_mean'] - data['amps_I_std'],
                        data['amps_I_mean'] + data['amps_I_std'],
                        color=lplot.color_array_long[0],
                        alpha=0.2)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Distance from Soma")
        # Save plt.
        path = os.path.join(dir_plot, fname + "." + format)
        plt.savefig(path, format=format, bbox_inches='tight')
        plt.close()

        # New plot.
        fname = 'sphere_spike_amps_II'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['bins'],
                 data['amps_II_mean'],
                 color=lplot.color_array_long[0],
                 marker='o',
                 markersize=5)
        ax.fill_between(data['bins'],
                        data['amps_II_mean'] - data['amps_II_std'],
                        data['amps_II_mean'] + data['amps_II_std'],
                        color=lplot.color_array_long[0],
                        alpha=0.2)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Distance from Soma")
        # Save plt.
        path = os.path.join(dir_plot, fname + "." + format)
        plt.savefig(path, format=format, bbox_inches='tight')
        plt.close()

        # New plot.
        fname = 'sphere_spike_width_I'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['bins'],
                 data['widths_I_mean'],
                 color=lplot.color_array_long[0],
                 marker='o',
                 markersize=5)
        ax.fill_between(data['bins'],
                        data['widths_I_mean'] - data['widths_I_std'],
                        data['widths_I_mean'] + data['widths_I_std'],
                        color=lplot.color_array_long[0],
                        alpha=0.2)
        ax.set_ylabel("Width Type I")
        ax.set_xlabel("Distance from Soma")
        # Save plt.
        path = os.path.join(dir_plot, fname + "." + format)
        plt.savefig(path, format=format, bbox_inches='tight')
        plt.close()

        # New plot.
        fname = 'sphere_spike_width_II'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['bins'],
                 data['widths_II_mean'],
                 color=lplot.color_array_long[0],
                 marker='o',
                 markersize=5)
        ax.fill_between(data['bins'],
                        data['widths_II_mean'] - data['widths_II_std'],
                        data['widths_II_mean'] + data['widths_II_std'],
                        color=lplot.color_array_long[0],
                        alpha=0.2)
        ax.set_ylabel("Width Type II")
        ax.set_xlabel("Distance from Soma")
        # Save plt.
        path = os.path.join(dir_plot, fname + "." + format)
        plt.savefig(path, format=format, bbox_inches='tight')
        plt.close()

        # New plot.
        for i in self.elec_to_plot:
            fname = 'sphere_elec_{}'.format(i)
            print "plotting            :", fname
            c = lplot.get_short_color_array(2 + 1)
            plt.figure(figsize=lplot.size_common)
            ax = plt.gca()
            lplot.nice_axes(ax)
            # Plot
            plt.plot(data['spikes_t_vec'], data['spikes'][i], color=c[0])

            # Trace I
            trace_idx = np.where(~np.isnan(data['widths_I_trace'][i]))[0]
            trace_idx = [trace_idx[0], trace_idx[-1]]
            plt.plot(data['spikes_t_vec'][trace_idx],
                     data['widths_I_trace'][i][trace_idx],
                     color=c[1],
                     marker="|")
            # Trace II
            trace_idx = np.where(~np.isnan(data['widths_II_trace'][i]))[0]
            trace_idx = [trace_idx[0], trace_idx[-1]]
            plt.plot(data['spikes_t_vec'],
                     data['widths_II_trace'][i],
                     color=c[1])
            # Save plt.
            path = os.path.join(dir_plot, fname + "." + format)
            plt.savefig(path, format=format, bbox_inches='tight')
            plt.close()

        # Correlation scatter plot of spike widths.
        fname = 'sphere_correlation'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.scatter(data['widths_II'],
                    data['widths_I'],
                    color=lplot.color_array_long[0],
                    marker='x')
        ax.set_xlabel("Spike Width Type II")
        ax.set_ylabel("Spike Width Type I")
        # ax.set_aspect('equal')
        # Save plt.
        path = os.path.join(dir_plot, fname + "." + format)
        plt.savefig(path, format=format, bbox_inches='tight')
        plt.close()

        # Correlation scatter plot of spike widths.
        fname = 'sphere_r_hist'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.hist(data['elec_r_all'],
                 self.bins - 1,
                 range=(0, data['elec_r_all'].max()),
                 facecolor=lplot.color_array_long[0],
                 alpha=0.5)
        # Save plt.
        path = os.path.join(dir_plot, fname + "." + format)
        plt.savefig(path, format=format, bbox_inches='tight')
        plt.close()
