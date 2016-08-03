import os
import numpy as np
import LFPy
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import LFPy_util
import LFPy_util.data_extraction as de
import LFPy_util.plot as lplot
import LFPy_util.colormaps as lcmaps
from LFPy_util.sims.Simulation import Simulation


class DiscXY(Simulation):
    """
    Disc simulation
    """

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.set_name("disc")

        self.verbose = False
        self.run_param['n'] = 9
        self.run_param['n_phi'] = 8
        self.run_param['R'] = 50
        self.run_param['sigma'] = 0.3
        self.run_param['R_0'] = 10
        self.run_param['ext_method'] = 'som_as_point'

        self.process_param['amp_option'] = 'both'
        self.process_param['pre_dur'] = 16.7 * 0.5
        self.process_param['post_dur'] = 16.7 * 0.5
        self.process_param['threshold'] = 4
        self.process_param['width_II_thresh'] = 0.5
        self.process_param['width_III_thresh'] = 0.5
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
            run_param['n_phi'],
            run_param['n'],
        ])
        elec_y = np.empty(elec_x.shape)
        elec_z = np.empty(elec_x.shape)

        theta = 0.5 * np.pi
        for i in xrange(run_param['n_phi']):
            phi = float(i) / run_param['n_phi'] * np.pi * 2
            y1 = run_param['R'] * np.cos(theta)
            x1 = run_param['R'] * np.sin(theta) * np.sin(phi)
            z1 = run_param['R'] * np.sin(theta) * np.cos(phi)
            y0 = run_param['R_0'] * np.cos(theta)
            x0 = run_param['R_0'] * np.sin(theta) * np.sin(phi)
            z0 = run_param['R_0'] * np.sin(theta) * np.cos(phi)
            elec_x[i] = np.linspace(x0, x1, run_param['n'])
            elec_y[i] = np.linspace(y0, y1, run_param['n'])
            elec_z[i] = np.linspace(z0, z1, run_param['n'])

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
        data['poly_morph_xz'] \
                = de.get_polygons_no_axon(cell,['x','z'])
        data['poly_morph_axon_xz'] \
                = de.get_polygons_axon(cell,['x','z'])

    def process_data(self):
        data = self.data
        run_param = self.run_param
        process_param = self.process_param

        # Get the signal from the soma potential.
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
            threshold=process_param['width_II_thresh'],
            dt=data['dt'],
            amp_option=process_param['amp_option'])

        widths_III, widths_III_trace = de.find_wave_width_type_III(
            spikes,
            threshold=process_param['width_III_thresh'],
            dt=data['dt'],
            amp_option=process_param['amp_option'])

        p = run_param['n_phi']
        n = run_param['n']

        amps_I = np.reshape(amps_I, (p, n))
        amps_II = np.reshape(amps_II, (p, n))
        widths_I = np.reshape(widths_I, (p, n))
        widths_II = np.reshape(widths_II, (p, n))
        widths_III = np.reshape(widths_III, (p, n))

        # Becomes vector with length n.
        amps_I_mean = np.mean(amps_I, 0)
        amps_I_std = np.std(amps_I, 0)
        amps_II_mean = np.mean(amps_II, 0)
        amps_II_std = np.std(amps_II, 0)
        widths_I_mean = np.mean(widths_I, 0)
        widths_I_std = np.std(widths_I, 0)
        widths_II_mean = np.mean(widths_II, 0)
        widths_II_std = np.std(widths_II, 0)
        widths_III_mean = np.mean(widths_III, 0)
        widths_III_std = np.std(widths_III, 0)

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

        data['widths_III_mean'] = widths_III_mean
        data['widths_III_std'] = widths_III_std
        data['widths_III'] = widths_III
        data['widths_III_trace'] = widths_III_trace * 1000

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

        # Plot spike amps I {{{1 #
        fname = self.name + '_spike_amps_I'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['r_vec'],
                 data['amps_I_mean'],
                 color=lcmaps.get_color(0),
                 marker='o',
                 markersize=5,
                 )
        ax.fill_between(data['r_vec'],
                        data['amps_I_mean'] - data['amps_I_std'],
                        data['amps_I_mean'] + data['amps_I_std'],
                        color=lcmaps.get_color(0),
                        alpha=0.2)
        ax.set_ylabel(r"Amplitude \textbf{[\si{\milli\volt}]}")
        ax.set_xlabel(r"Distance from Soma \textbf{[\si{\micro\metre}]}")
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # }}} # 
        # {{{ Plot spike amps I log
        fname = self.name + '_spike_amps_I_log'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['r_vec'],
                 data['amps_I_mean'],
                 color=lcmaps.get_color(0),
                 marker='o',
                 markersize=5,
                 )
        ax.fill_between(data['r_vec'],
                        data['amps_I_mean'] - data['amps_I_std'],
                        data['amps_I_mean'] + data['amps_I_std'],
                        color=lcmaps.get_color(0),
                        alpha=0.2)
        # Ugly way to put in some graphs for power laws.
        # Left side.
        x0 = data['r_vec'][0]
        x1 = data['r_vec'][1]
        y0 = data['amps_I_mean'][0]
        for p in [1.0, 2.0, 3.0]:
            y1 = np.power( 1.0 / data['r_vec'][1], p) * \
                    data['amps_I_mean'][ 0] / \
                    np.power(1.0 / data['r_vec'][0], p)
            ax.plot([x0, x1], [y0, y1], color='black')
            ax.annotate(
                    '-'+str(int(p)), 
                    xy=(x1,y1), 
                    xytext=(x1*1.01, y1*0.95),
                    )
        # Right side.
        x0 = data['r_vec'][-3]
        x1 = data['r_vec'][-1]
        y1 = data['amps_I_mean'][-1]
        for p in [1.0, 2.0, 3.0]:
            y0 = np.power(1.0 / data['r_vec'][-3], p) * \
                    data['amps_I_mean'][-1] / \
                    np.power(1.0 / data['r_vec'][-1], p) 
            ax.plot([x0, x1], [y0, y1], color='black')
            ax.annotate(
                    '-'+str(int(p)), 
                    xy=(x0,y0), 
                    xytext=(x0*0.99, y0*0.95),
                    horizontalalignment='right',
                    )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([data['r_vec'].min(), data['r_vec'].max()])
        ticker = mpl.ticker.MaxNLocator(nbins=7)
        ax.xaxis.set_major_locator(ticker)
        ax.xaxis.get_major_formatter().labelOnlyBase = False

        # ticker = mpl.ticker.MaxNLocator(nbins=7)
        # ax.yaxis.set_major_locator(ticker)
        # ax.yaxis.get_major_formatter().labelOnlyBase = False

        # Set a label formatter to use normal numbers.
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

        ax.set_ylabel(r"Amplitude \textbf{[\si{\milli\volt}]}")
        ax.set_xlabel(r"Distance from Soma \textbf{[\si{\micro\metre}]}")
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # }}} 
        # Plot spike amps II {{{1 #
        fname = self.name + '_spike_amps_II'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['r_vec'],
                 data['amps_II_mean'],
                 color=lcmaps.get_color(0),
                 marker='o',
                 markersize=5,
                 )
        ax.fill_between(data['r_vec'],
                        data['amps_II_mean'] - data['amps_II_std'],
                        data['amps_II_mean'] + data['amps_II_std'],
                        color=lcmaps.get_color(0),
                        alpha=0.2)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Distance from Soma")
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
        # {{{ Plot spike amps II log
        fname = self.name + '_spike_amps_II_log'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['r_vec'],
                 data['amps_II_mean'],
                 color=lcmaps.get_color(0),
                 marker='o',
                 markersize=5,
                 )
        ax.fill_between(data['r_vec'],
                        data['amps_II_mean'] - data['amps_II_std'],
                        data['amps_II_mean'] + data['amps_II_std'],
                        color=lcmaps.get_color(0),
                        alpha=0.2)

        # Ugly way to put in some graphs for power laws.
        # Left side.
        x0 = data['r_vec'][0]
        x1 = data['r_vec'][1]
        y0 = data['amps_II_mean'][0]
        for p in [1.0, 2.0, 3.0]:
            y1 = np.power( 1.0 / data['r_vec'][1], p) * \
                    data['amps_II_mean'][ 0] / \
                    np.power(1.0 / data['r_vec'][0], p)
            ax.plot([x0, x1], [y0, y1], color='black')
            ax.annotate(
                    '-'+str(int(p)), 
                    xy=(x1,y1), 
                    xytext=(x1*1.01, y1*0.95),
                    )
        # Right side.
        x0 = data['r_vec'][-3]
        x1 = data['r_vec'][-1]
        y1 = data['amps_II_mean'][-1]
        for p in [1.0, 2.0, 3.0]:
            y0 = np.power(1.0 / data['r_vec'][-3], p) * \
                    data['amps_II_mean'][-1] / \
                    np.power(1.0 / data['r_vec'][-1], p) 
            ax.plot([x0, x1], [y0, y1], color='black')
            ax.annotate(
                    '-'+str(int(p)), 
                    xy=(x0,y0), 
                    xytext=(x0*0.99, y0*0.95),
                    horizontalalignment='right',
                    )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([data['r_vec'].min(), data['r_vec'].max()])
        ticker = mpl.ticker.MaxNLocator(nbins=7)
        ax.xaxis.set_major_locator(ticker)
        ax.xaxis.get_major_formatter().labelOnlyBase = False

        # ticker = mpl.ticker.MaxNLocator(nbins=7)
        # ax.yaxis.set_major_locator(ticker)
        # ax.yaxis.get_major_formatter().labelOnlyBase = False

        # Set a label formatter to use normal numbers.
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

        ax.set_ylabel(r"Amplitude \textbf{[\si{\milli\volt}]}")
        ax.set_xlabel(r"Distance from Soma \textbf{[\si{\micro\metre}]}")
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # }}} 
        # Plot spike width I {{{1 #
        fname = self.name + '_spike_width_I'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['r_vec'],
                 data['widths_I_mean'],
                 color=lcmaps.get_color(0),
                 marker='o',
                 markersize=5,
                 )
        ax.fill_between(data['r_vec'],
                        data['widths_I_mean'] - data['widths_I_std'],
                        data['widths_I_mean'] + data['widths_I_std'],
                        color=lcmaps.get_color(0),
                        alpha=0.2)
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
        plt.plot(data['r_vec'],
                 data['widths_II_mean'],
                 color=lcmaps.get_color(0),
                 marker='o',
                 markersize=5,
                 )
        ax.fill_between(
            data['r_vec'],
            data['widths_II_mean'] - data['widths_II_std'],
            data['widths_II_mean'] + data['widths_II_std'],
            color=lcmaps.get_color(0),
            alpha=0.2)
        ax.set_ylabel("Spike Width")
        ax.set_xlabel("Distance from Soma")
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
        # Plot spike widths III {{{1 #
        fname = self.name + '_spike_width_III'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['r_vec'],
                 data['widths_III_mean'],
                 color=lcmaps.get_color(0),
                 marker='o',
                 markersize=5,
                 )
        ax.fill_between(
            data['r_vec'],
            data['widths_III_mean'] - data['widths_III_std'],
            data['widths_III_mean'] + data['widths_III_std'],
            color=lcmaps.get_color(0),
            alpha=0.2)
        ax.set_ylabel("Spike Width")
        ax.set_xlabel("Distance from Soma")
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
        # Plot morphology xz {{{ #
        LFPy_util.plot.morphology(data['poly_morph_xz'],
                                  data['poly_morph_axon_xz'],
                                  elec_x=data['elec_x'],
                                  elec_y=data['elec_z'],
                                  fig_size=lplot.size_square,
                                  fname=self.name + "_morph_elec_xz",
                                  plot_save_dir=dir_plot,
                                  x_label='x',
                                  y_label='z',
                                  show=False)
        # }}} #
        # Spike to plot.
        elec_index = run_param['n']/2
        # title_str = r"Distance from Soma = \SI{{{}}}{{\micro\metre}}"
        # title_str = title_str.format(round(data['r_vec'][elec_index]),2)
        c = lcmaps.get_short_color_array(2 + 1)
        # Plot middle electrode spike {{{1 #
        fname = self.name + '_middle_elec_spike'
        print "plotting            :", fname
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
        plt.plot(freq, amp, color=lcmaps.get_color(0))
        # plt.title(title_str)
        ax.set_ylabel(r'Amplitude \textbf[$\mathbf{mV}$\textbf]')
        ax.set_xlabel(r'Frequency \textbf[$\mathbf{kHz}$\textbf]')
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
        # Plot middle electrode signal {{{1 #
        fname = self.name + '_middle_elec'
        print "plotting            :", fname
        c = lcmaps.get_short_color_array(2 + 1)
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['t_vec'],
                 data['LFP'][elec_index],
                 color=lcmaps.get_color(0))
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
        plt.plot(freq, amp, color=lcmaps.get_color(0))
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
            # t = len(run_param['theta'])
            # p = run_param['n_phi']
            p = 1
            n = run_param['n']
            # This title string should be formatted.
            title_str = r"Distance from Soma = \SI{{{}}}{{\micro\metre}}"

            cnt = 0
            for j in xrange(p):
                for k in xrange(n):
                    title_str_1 = title_str.format(data['r_vec'][k])
                    # Plot all elec spikes {{{1 #
                    fname = self.name + '_elec_p_{}_n_{}'.format(j * 360 / p, k)
                    print "plotting            :", fname
                    c = lcmaps.get_short_color_array(2 + 1)
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
                    # Trace III
                    plt.plot(data['spikes_t_vec'],
                             data['widths_III_trace'][cnt],
                             color=c[1])
                    # plt.title(title_str_1)
                    # Save plt.
                    lplot.save_plt(plt, fname, sub_dir)
                    plt.close()
                    # 1}}} #
                    # Plot all elec spikes freq {{{1 #
                    # Fourier plot.
                    fname = self.name + '_freq_elec_p_{}_n_{}'.format(j * 360 / p, k)
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
