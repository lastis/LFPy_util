import os
import numpy as np
import LFPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import LFPy_util
import LFPy_util.data_extraction as de
import LFPy_util.plot as lplot
from LFPy_util.sims.Simulation import Simulation


class Symmetry(Simulation):
    """
    Symmetry simulation
    """

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.set_name("sym")

        self.verbose = False
        self.run_param['n'] = 9
        self.run_param['n_phi'] = 8
        self.run_param['theta'] = [10, 50, 90, 130, 170]
        self.run_param['R'] = 50
        self.run_param['sigma'] = 0.3
        self.run_param['R_0'] = 10
        self.run_param['ext_method'] = 'som_as_point'

        self.process_param['amp_option'] = 'both'
        self.process_param['pre_dur'] = 16.7 * 0.5
        self.process_param['post_dur'] = 16.7 * 0.5
        self.process_param['threshold'] = 4
        # Index of the spike to measure from.
        self.process_param['spike_to_measure'] = 0
        self.plot_param['plot_detailed'] = False
        # kHz.
        self.plot_param['freq_end'] = None

    def simulate(self, cell):
        # pylint: disable=invalid-name,no-member
        data = self.data
        run_param = self.run_param
        elec_x = np.empty([
            len(run_param['theta']),
            run_param['n_phi'],
            run_param['n'],
        ])
        elec_y = np.empty(elec_x.shape)
        elec_z = np.empty(elec_x.shape)
        for i, theta in enumerate(run_param['theta']):
            theta = theta * np.pi / 180
            for j in xrange(run_param['n_phi']):
                phi = float(j) / run_param['n_phi'] * np.pi * 2
                y1 = run_param['R'] * np.cos(theta)
                x1 = run_param['R'] * np.sin(theta) * np.sin(phi)
                z1 = run_param['R'] * np.sin(theta) * np.cos(phi)
                y0 = run_param['R_0'] * np.cos(theta)
                x0 = run_param['R_0'] * np.sin(theta) * np.sin(phi)
                z0 = run_param['R_0'] * np.sin(theta) * np.cos(phi)
                elec_x[i, j] = np.linspace(x0, x1, run_param['n'])
                elec_y[i, j] = np.linspace(y0, y1, run_param['n'])
                elec_z[i, j] = np.linspace(z0, z1, run_param['n'])
        x = elec_x.flatten()
        y = elec_y.flatten()
        z = elec_z.flatten()
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
        data['soma_v'] = cell.somav
        data['dt'] = cell.timeres_NEURON
        data['poly_morph'] \
                = de.get_polygons_no_axon(cell,['x','y'])
        data['poly_morph_axon'] \
                = de.get_polygons_axon(cell,['x','y'])

    def process_data(self):
        data = self.data
        run_param = self.run_param
        process_param = self.process_param

        # Get the signal from the soma potential.
        # signal = data['LFP'][0]
        signal = data['soma_v']
        spike, spikes_t_vec, I = de.extract_spikes(
            data['t_vec'],
            signal,
            pre_dur=process_param['pre_dur'],
            post_dur=process_param['post_dur'],
            threshold=process_param['threshold'],
            amp_option=process_param['amp_option'], 
            )
        # Gather all spikes from the same indices as where the spike appears
        # in the first electrode.
        spike_index = process_param['spike_to_measure']
        if spike.shape[0] < spike_index:
            raise ValueError("Found fewer spikes than process_param['spike_to_measure']")
        spikes = data['LFP'][:, I[spike_index, 0]:I[spike_index, 1]]

        amps_I = de.find_amplitude_type_I(spikes, amp_option=process_param['amp_option'])
        amps_II = de.find_amplitude_type_II(spikes)
        widths_I, widths_I_trace = de.find_wave_width_type_I(spikes,
                                                             dt=data['dt'])
        widths_II, widths_II_trace = de.find_wave_width_type_II(
            spikes,
            dt=data['dt'],
            amp_option=process_param['amp_option'])

        t = len(run_param['theta'])
        p = run_param['n_phi']
        n = run_param['n']

        amps_I = np.reshape(amps_I, (t, p, n))
        amps_II = np.reshape(amps_II, (t, p, n))
        widths_I = np.reshape(widths_I, (t, p, n))
        widths_II = np.reshape(widths_II, (t, p, n))

        # Becomes [t x n] matrix.
        amps_I_mean = np.mean(amps_I, 1)
        amps_I_std = np.std(amps_I, 1)
        amps_II_mean = np.mean(amps_II, 1)
        amps_II_std = np.std(amps_II, 1)
        widths_I_mean = np.mean(widths_I, 1)
        widths_I_std = np.std(widths_I, 1)
        widths_II_mean = np.mean(widths_II, 1)
        widths_II_std = np.std(widths_II, 1)

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

        data['spikes'] = spikes * 1000
        data['spikes_t_vec'] = spikes_t_vec
        data['r_vec'] = np.linspace(run_param['R_0'], run_param['R'],
                                    run_param['n'])

    def plot(self, dir_plot):
        """
        Plotting stats about the spikes.
        """
        # pylint: disable=too-many-locals
        data = self.data
        run_param = self.run_param

        # String to put before output to the terminal.
        str_start = self.name
        str_start += " "*(20 - len(self.name)) + ":"

        # Set global matplotlib parameters.
        LFPy_util.plot.set_rc_param()

        # 3D plot {{{1 #
        fname = self.name + "_elec_pos"
        c = lplot.get_short_color_array(5)[2]
        fig = plt.figure(figsize=lplot.size_common)
        ax = fig.add_subplot(111, projection='3d')
        cnt = 0
        for i, theta in enumerate(run_param['theta']):
            for j in xrange(run_param['n_phi']):
                for k in xrange(run_param['n']):
                    ax.scatter(data['elec_x'][cnt],
                               data['elec_y'][cnt],
                               data['elec_z'][cnt],
                               c=c)
                    cnt += 1
        ax.set_xlim(-run_param['R'], run_param['R'])
        ax.set_ylim(-run_param['R'], run_param['R'])
        ax.set_zlim(-run_param['R'], run_param['R'])
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
        # Plot spike amps I {{{1 #
        fname = self.name + '_spike_amps_I'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        c = lplot.get_short_color_array(data['amps_I_mean'].shape[0] + 1)
        for t in xrange(data['amps_I_mean'].shape[0]):
            label = r'$\theta = {}\degree$'.format(run_param['theta'][t])
            plt.plot(data['r_vec'],
                     data['amps_I_mean'][t],
                     color=c[t],
                     marker='o',
                     markersize=5,
                     label=label)
            ax.fill_between(data['r_vec'],
                            data['amps_I_mean'][t] - data['amps_I_std'][t],
                            data['amps_I_mean'][t] + data['amps_I_std'][t],
                            color=c[t],
                            alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        # Position the legen on the right side of the plot.
        ax.legend(handles,
                  labels,
                  loc='upper left',
                  bbox_to_anchor=(1.0, 1.04), )
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Distance from Soma")
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # }}} # 
        # Plot spike amps II {{{1 #
        fname = self.name + '_spike_amps_II'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        c = lplot.get_short_color_array(data['amps_II_mean'].shape[0] + 1)
        for t in xrange(data['amps_II_mean'].shape[0]):
            label = r'$\theta = {}\degree$'.format(run_param['theta'][t])
            plt.plot(data['r_vec'],
                     data['amps_II_mean'][t],
                     color=c[t],
                     marker='o',
                     markersize=5,
                     label=label)
            ax.fill_between(data['r_vec'],
                            data['amps_II_mean'][t] - data['amps_II_std'][t],
                            data['amps_II_mean'][t] + data['amps_II_std'][t],
                            color=c[t],
                            alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        # Position the legen on the right side of the plot.
        ax.legend(handles,
                  labels,
                  loc='upper left',
                  bbox_to_anchor=(1.0, 1.04), )
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Distance from Soma")
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
        # Plot spike width I {{{1 #
        fname = self.name + '_spike_width_I'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        c = lplot.get_short_color_array(data['widths_I_mean'].shape[0] + 1)
        for t in xrange(data['widths_I_mean'].shape[0]):
            label = r'$\theta = {}\degree$'.format(run_param['theta'][t])
            plt.plot(data['r_vec'],
                     data['widths_I_mean'][t],
                     color=c[t],
                     marker='o',
                     markersize=5,
                     label=label)
            ax.fill_between(data['r_vec'],
                            data['widths_I_mean'][t] - data['widths_I_std'][t],
                            data['widths_I_mean'][t] + data['widths_I_std'][t],
                            color=c[t],
                            alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        # Position the legen on the right side of the plot.
        ax.legend(handles,
                  labels,
                  loc='upper left',
                  bbox_to_anchor=(1.0, 1.04), )
        ax.set_ylabel("Spike Width")
        ax.set_xlabel("Distance from Soma")
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
        # Plot spike widths II {{{1 #
        fname = self.name + '_spike_width_II'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        c = lplot.get_short_color_array(data['widths_II_mean'].shape[0] + 1)
        for t in xrange(data['widths_II_mean'].shape[0]):
            label = r'$\theta = {}\degree$'.format(run_param['theta'][t])
            plt.plot(data['r_vec'],
                     data['widths_II_mean'][t],
                     color=c[t],
                     marker='o',
                     markersize=5,
                     label=label)
            ax.fill_between(
                data['r_vec'],
                data['widths_II_mean'][t] - data['widths_II_std'][t],
                data['widths_II_mean'][t] + data['widths_II_std'][t],
                color=c[t],
                alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        # Position the legen on the right side of the plot.
        ax.legend(handles,
                  labels,
                  loc='upper left',
                  bbox_to_anchor=(1.0, 1.04), )
        ax.set_ylabel("Spike Width")
        ax.set_xlabel("Distance from Soma")
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
        # Plot morphology {{{1 #
        LFPy_util.plot.morphology(data['poly_morph'],
                                  data['poly_morph_axon'],
                                  elec_x=data['elec_x'],
                                  elec_y=data['elec_y'],
                                  fig_size=lplot.size_common,
                                  fname=self.name + "_morph_elec",
                                  plot_save_dir=dir_plot,
                                  show=False)
        # 1}}} #
        # Get the spike to plot.
        elec_index = run_param['n']/2
        # title_str = r"Distance from Soma = \SI{{{}}}{{\micro\metre}}"
        # title_str = title_str.format(round(data['r_vec'][elec_index]),2)
        # Plot middle electrode spike {{{1 #
        fname = self.name + '_middle_elec_spike'
        print "plotting            :", fname
        c = lplot.get_short_color_array(2 + 1)
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['spikes_t_vec'],
                 data['spikes'][elec_index],
                 color=c[0])
        # Trace I
        plt.plot(data['spikes_t_vec'],
                 data['widths_I_trace'][elec_index],
                 color=c[1],
                 )
        # Trace II
        plt.plot(data['spikes_t_vec'],
                 data['widths_II_trace'][elec_index],
                 color=c[1])
        # plt.title(title_str)
        # Save plt.
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
        # Plot middle electrode spike freq {{{1 #
        fname = self.name + '_middle_elec_spike_fourier'
        freq, amp, phase = de.find_freq_and_fft(
            data['dt'],
            data['spikes'][elec_index],
            )
        # Remove the first coefficient as we don't care about the baseline.
        freq = np.delete(freq, 0)
        amp = np.delete(amp, 0)
        # Delete frequencies above the option.
        if self.plot_param['freq_end'] is not None:
            idx = min(
                range(len(freq)), 
                key=lambda i: abs(freq[i] - self.plot_param['freq_end'])
                )
            freq = freq[0:idx]
            amp = amp[0:idx]
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        plt.plot(freq, amp, color=c[0])
        # plt.title(title_str)
        ax.set_ylabel(r'Amplitude \textbf[$\mathbf{mV}$\textbf]')
        ax.set_xlabel(r'Frequency \textbf[$\mathbf{kHz}$\textbf]')
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
        # Plot middle electrode signal {{{1 #
        fname = self.name + '_middle_elec'
        print "plotting            :", fname
        c = lplot.get_short_color_array(2 + 1)
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['t_vec'],
                 data['LFP'][elec_index],
                 color=c[0])
        # plt.title(title_str)
        # Save plt.
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
        # Plot middle electrode signal freq {{{1 #
        fname = self.name + '_middle_elec_fourier'
        freq, amp, phase = de.find_freq_and_fft(
            data['dt'],
            data['LFP'][elec_index],
            )
        # Remove the first coefficient as we don't care about the baseline.
        freq = np.delete(freq, 0)
        amp = np.delete(amp, 0)
        if self.plot_param['freq_end'] is not None:
            idx = min(
                range(len(freq)), 
                key=lambda i: abs(freq[i] - self.plot_param['freq_end'])
                )
            freq = freq[0:idx]
            amp = amp[0:idx]
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        plt.plot(freq, amp, color=c[0])
        ax.set_ylabel(r'Amplitude \textbf[$\mathbf{mV}$\textbf]')
        ax.set_xlabel(r'Frequency \textbf[$\mathbf{kHz}$\textbf]')
        # plt.title(title_str)
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #

        if self.plot_param['plot_detailed']:
            # Create the directory if it does not exist.
            sub_dir = os.path.join(dir_plot, self.name + "_detailed")
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            t = len(run_param['theta'])
            # p = run_param['n_phi']
            p = 1
            n = run_param['n']
            # This title string should be formatted.
            title_str = r"Distance from Soma = \SI{{{}}}{{\micro\metre}}"

            cnt = 0
            for i in xrange(t):
                for j in xrange(p):
                    for k in xrange(n):
                        title_str_1 = title_str.format(data['r_vec'][k])
                        # Plot all elec spikes {{{1 #
                        fname = self.name + '_elec_t_{}_p_{}_n_{}'.format(
                            run_param['theta'][i], j * 360 / p, k)
                        print "plotting            :", fname
                        c = lplot.get_short_color_array(2 + 1)
                        plt.figure(figsize=lplot.size_common)
                        ax = plt.gca()
                        lplot.nice_axes(ax)
                        # Plot
                        plt.plot(data['spikes_t_vec'],
                                 data['spikes'][cnt],
                                 color=c[0])

                        # Trace I
                        plt.plot(data['spikes_t_vec'],
                                 data['widths_I_trace'][cnt],
                                 color=c[1],
                                 )
                        # Trace II
                        plt.plot(data['spikes_t_vec'],
                                 data['widths_II_trace'][cnt],
                                 color=c[1])
                        # plt.title(title_str_1)
                        # Save plt.
                        lplot.save_plt(plt, fname, sub_dir)
                        plt.close()
                        # 1}}} #
                        # Plot all elec spikes freq {{{1 #
                        # Fourier plot.
                        fname = self.name + '_freq_elec_t_{}_p_{}_n_{}'.format(
                            run_param['theta'][i], j * 360 / p, k)
                        freq, amp, phase = de.find_freq_and_fft(
                            data['dt'],
                            data['spikes'][cnt],
                            )
                        # Remove the first coefficient as we don't care about the baseline.
                        freq = np.delete(freq, 0)
                        amp = np.delete(amp, 0)
                        if self.plot_param['freq_end'] is not None:
                            idx = min(
                                range(len(freq)), 
                                key=lambda i: abs(freq[i] - self.plot_param['freq_end'])
                                )
                            freq = freq[0:idx]
                            amp = amp[0:idx]
                        print "plotting            :", fname
                        plt.figure(figsize=lplot.size_common)
                        ax = plt.gca()
                        lplot.nice_axes(ax)
                        plt.plot(freq, amp, color=c[0])
                        ax.set_ylabel(r'Amplitude \textbf[$\mathbf{mV}$\textbf]')
                        ax.set_xlabel(r'Frequency \textbf[$\mathbf{kHz}$\textbf]')
                        # plt.title(title_str_1)
                        lplot.save_plt(plt, fname, sub_dir)
                        plt.close()
                        # 1}}} #
                        cnt += 1
