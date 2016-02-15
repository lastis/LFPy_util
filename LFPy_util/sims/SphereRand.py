import os
import numpy as np
import LFPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import LFPy_util
import LFPy_util.data_extraction as de
import LFPy_util.plot as lplot
import warnings
from LFPy_util.sims.Simulation import Simulation


class SphereRand(Simulation):
    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.name = "sphere"

        self.debug = False
        self.run_param['N'] = 500
        self.run_param['R'] = 50
        self.run_param['sigma'] = 0.3
        self.run_param['ext_method'] = 'som_as_point'

        self.process_param['amp_option'] = 'both'
        self.process_param['pre_dur'] = 16.7 * 0.5
        self.process_param['post_dur'] = 16.7 * 0.5
        self.process_param['threshold'] = 3
        self.process_param['elec_to_plot'] = []
        self.process_param['bins'] = 11
        self.process_param['amp_threshold'] = 0  # uV
        self.process_param['spike_to_measure'] = 0 

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
        electrode.method = run_param['ext_method']
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
        process_param = self.process_param
        t_vec = np.array(data['t_vec'])
        v_vec = np.array(data['LFP'])*1000
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
            if np.fabs(signal).max() * 1000 < self.process_param['amp_threshold']:
                deleted_elem.append(row)
                continue
            spike, spikes_t_vec, I = de.extract_spikes(
                t_vec,
                signal,
                pre_dur=self.process_param['pre_dur'],
                post_dur=self.process_param['post_dur'],
                threshold=self.process_param['threshold'],
                amp_option=self.process_param['amp_option'], )
            # Only use one spike from each electrode.
            spikes.append(spike[process_param['spike_to_measure']])
        spikes = np.array(spikes)
        r = np.delete(r, deleted_elem)

        # Find widths of the spikes, trace can be used for plotting.
        widths_I, widths_I_trace = de.find_wave_width_type_I(spikes,
                                                             dt=data['dt'])
        widths_II, widths_II_trace = de.find_wave_width_type_II(
            spikes,
            dt=data['dt'],
            amp_option=self.process_param['amp_option'])
        amps_I = de.find_amplitude_type_I(spikes, amp_option=self.process_param['amp_option'])
        amps_II = de.find_amplitude_type_II(spikes)
        # Put widths_I in bins decided by the radial distance. 
        # Then calculate std and mean.
        bins = np.linspace(0, run_param['R'], self.process_param['bins'], endpoint=True)
        # Find the indices of the bins to which each value in r array belongs.
        inds = np.digitize(r, bins)
        # Widths and amps binned by distance.
        widths_I_at_r = [widths_I[inds == i] for i in range(len(bins))]
        widths_II_at_r = [widths_II[inds == i] for i in range(len(bins))]
        amps_I_at_r = [amps_I[inds == i] for i in range(len(bins))]
        amps_II_at_r = [amps_II[inds == i] for i in range(len(bins))]

        with warnings.catch_warnings():
            # Ignore warnings where mean or var has zero sized array as input.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            widths_I_mean = [widths_I[inds == i].mean() for i in range(len(bins))]
            widths_I_std = [np.sqrt(widths_I[inds == i].var()) for i in range(len(bins))]
            widths_II_mean = [widths_II[inds == i].mean() for i in range(len(bins))]
            widths_II_std = [np.sqrt(widths_II[inds == i].var()) for i in range(len(bins))]
            amps_I_mean = [amps_I[inds == i].mean() for i in range(len(bins))]
            amps_I_std = [np.sqrt(amps_I[inds == i].var()) for i in range(len(bins))]
            amps_II_mean = [amps_II[inds == i].mean() for i in range(len(bins))]
            amps_II_std = [np.sqrt(amps_II[inds == i].var()) for i in range(len(bins))]

            widths_I_mean = np.array(widths_I_mean)
            widths_II_mean = np.array(widths_II_mean)
            amps_I_mean = np.array(amps_I_mean)
            amps_II_mean = np.array(amps_II_mean)
            widths_I_std = np.array(widths_I_std)
            widths_II_std = np.array(widths_II_std)
            amps_I_std = np.array(amps_I_std)
            amps_II_std = np.array(amps_II_std)

        data['elec_r_all'] = r_all

        data['amps_I_mean'] = amps_I_mean
        data['amps_I_std'] = amps_I_std
        data['amps_I'] = amps_I
        data['amps_I_at_r'] = amps_I_at_r

        data['amps_II_mean'] = amps_II_mean
        data['amps_II_std'] = amps_II_std
        data['amps_II'] = amps_II
        data['amps_II_at_r'] = amps_II_at_r

        data['widths_I_mean'] = widths_I_mean
        data['widths_I_std'] = widths_I_std
        data['widths_I'] = widths_I
        data['widths_I_trace'] = widths_I_trace 
        data['widths_I_at_r'] = widths_I_at_r 

        data['widths_II_mean'] = widths_II_mean
        data['widths_II_std'] = widths_II_std
        data['widths_II'] = widths_II
        data['widths_II_trace'] = widths_II_trace 
        data['widths_II_at_r'] = widths_II_at_r

        data['bins'] = bins
        data['elec_r'] = r
        data['spikes'] = spikes
        data['spikes_t_vec'] = spikes_t_vec

    def plot(self, dir_plot):
        data = self.data
        run_param = self.run_param
        # Set global matplotlib parameters.
        LFPy_util.plot.set_rc_param()

        # Plot 3d points {{{1 #
        # 3D plot.
        fname = self.name + "_elec_pos"
        c = lplot.get_short_color_array(5)[2]
        fig = plt.figure(figsize=lplot.size_common)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data['elec_x'], data['elec_y'], data['elec_z'], c=c)
        ax.set_xlim(-run_param['R'], run_param['R'])
        ax.set_ylim(-run_param['R'], run_param['R'])
        ax.set_zlim(-run_param['R'], run_param['R'])
        # Save plt.
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
        # Plot spike amps I {{{1 #
        # New plot.
        fname = self.name + '_spike_amps_I'
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
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
        # Plot spike amps II {{{1 #
        # New plot.
        fname = self.name + '_spike_amps_II'
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
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
        # Plot Spike Width I {{{1 #
        # New plot.
        fname = self.name + '_spike_width_I'
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
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        #  }}} #
        # Plot Spike Width I {{{1 #
        # New plot.
        fname = self.name + '_spike_width_II'
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
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        #  }}} #
        # Plot Single Electrodes {{{1 #
        # Title string that can be formatted.
        title_str = r"Distance from Soma = \SI{{{}}}{{\micro\metre}}"
        for i in self.plot_param['elec_to_plot']:
            title_str_1 = title_str.format(round(data['elec_r'][[i]],2))
            fname = self.name + '_elec_{}'.format(i)
            print "plotting            :", fname
            c = lplot.get_short_color_array(2 + 1)
            plt.figure(figsize=lplot.size_common)
            ax = plt.gca()
            lplot.nice_axes(ax)
            # Plot
            plt.plot(data['spikes_t_vec'], data['spikes'][i], color=c[0])

            # Trace I
            plt.plot(data['spikes_t_vec'],
                     data['widths_I_trace'][i],
                     color=c[1],
                     )
            # Trace II
            plt.plot(data['spikes_t_vec'],
                     data['widths_II_trace'][i],
                     color=c[1])
            plt.title(title_str_1)
            # Save plt.
            lplot.save_plt(plt, fname, dir_plot)
            plt.close()
        # }}} #
        # Plot Correlation {{{1 #
        # Correlation scatter plot of spike widths.
        fname = self.name + '_correlation'
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
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # }}} #
        # Plot Electrode Histo {{{1 #
        fname = self.name + '_r_hist'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.hist(data['elec_r_all'],
                 self.process_param['bins'] - 1,
                 range=(0, data['elec_r_all'].max()),
                 facecolor=lplot.color_array_long[0],
                 alpha=0.5)
        # Save plt.
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # }}} #
