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
import quantities as pq
from LFPy_util.sims.Simulation import Simulation


class SpikeWidthDef(Simulation):
    """
    SpikeWidthDef simulation
    """

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.set_name("widthdef")

        self.verbose = False
        self.run_param['sigma'] = 0.3
        self.run_param['ext_method'] = 'som_as_point'
        self.run_param['r'] = 50
        # self.run_param['random_elec'] = True
        self.run_param['N'] = 1
        self.run_param['seed'] = 1234

        self.process_param['amp_option'] = 'both'
        self.process_param['pre_dur'] = 16.7 * 0.5
        self.process_param['post_dur'] = 16.7 * 0.5
        self.process_param['threshold'] = 4
        # Index of the spike to measure from.
        self.process_param['spike_to_measure'] = 0

        self.plot_param['freq_end'] = 3*pq.kHz
        self.plot_param['use_tex'] = True

    def simulate(self, cell):
        # pylint: disable=invalid-name,no-member
        data = self.data
        run_param = self.run_param
        cell.simulate(rec_imem=True)

        np.random.seed(run_param['seed'])
        angle = np.random.uniform(0, 2*np.pi, size=run_param['N'])
        z = np.random.uniform(-1, 1, size=run_param['N']) 
        x = np.sqrt(1-z*z)*np.cos(angle) * run_param['r']
        y = np.sqrt(1-z*z)*np.sin(angle) * run_param['r']
        z = z * run_param['r']

        # Record the LFP of the electrodes.
        electrode = LFPy.RecExtElectrode(cell,
                                         x=x,
                                         y=y,
                                         z=z,
                                         sigma=run_param['sigma'])
        electrode.method = run_param['ext_method']
        electrode.calc_lfp()

        data['LFP'] = electrode.LFP*1000
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
        data['poly_morph_xz'] \
                = de.get_polygons_no_axon(cell,['x','z'])
        data['poly_morph_axon_xz'] \
                = de.get_polygons_axon(cell,['x','z'])

    def process_data(self):
        data = self.data
        run_param = self.run_param
        process_param = self.process_param

        signal = data['soma_v']
        spikes_soma, spikes_t_vec, I = de.extract_spikes(
            data['t_vec'],
            signal,
            pre_dur=process_param['pre_dur'],
            post_dur=process_param['post_dur'],
            threshold=process_param['threshold'],
            amp_option=process_param['amp_option'], 
            )

        spike_index = process_param['spike_to_measure']
        if spikes_soma.shape[0] < spike_index:
            raise ValueError("Found fewer spikes than process_param['spike_to_measure']")
        # Gather all spikes from the same indices as where the spike appears
        # in the membrane potential.
        spikes = data['LFP'][:, I[spike_index, 0]:I[spike_index, 1]]

        widths_I, widths_I_trace = de.find_wave_width_type_I(
            spikes,
            dt=data['dt'],
            )

        widths_II, widths_II_trace = de.find_wave_width_type_II(
            spikes,
            dt=data['dt'],
            amp_option=process_param['amp_option'],
            )

        widths_I_soma, widths_I_trace_soma = de.find_wave_width_type_I(
            spikes_soma[spike_index],
            dt=data['dt'],
            )

        widths_II_soma, widths_II_trace_soma = de.find_wave_width_type_II(
            spikes_soma[spike_index],
            dt=data['dt'],
            amp_option=process_param['amp_option'],
            )

        freq, amp, phase = \
            LFPy_util.data_extraction.find_freq_and_fft(data['dt'], spikes)
        # Remove the first coefficient as we don't care about the baseline.
        freq = np.delete(freq, 0)
        amp = np.delete(amp, 0, axis=1)

        data['freq'] = freq * pq.kHz
        data['amp'] = amp
        data['phase'] = phase

        data['spikes'] = spikes
        data['spikes_t_vec'] = spikes_t_vec
        data['spike_soma'] = spikes_soma[spike_index]
        data['width_I'] = widths_I
        data['width_I_trace'] = widths_I_trace
        data['width_II'] = widths_II
        data['width_II_trace'] = widths_II_trace
        data['width_I_soma'] = widths_I_soma
        data['width_I_trace_soma'] = widths_I_trace_soma[0]
        data['width_II_soma'] = widths_II_soma
        data['width_II_trace_soma'] = widths_II_trace_soma[0]

        self.info['elec_x'] = data['elec_x']
        self.info['elec_y'] = data['elec_y']
        self.info['elec_z'] = data['elec_z']

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

        # {{{ Plot Ext. Elec. Overlay
        fname = self.name + '_ext_all'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        for i in xrange(run_param['N']):
            # Plot
            plt.plot(data['spikes_t_vec'],
                     data['spikes'][i]/np.abs(data['spikes'][i].min()),
                     color=lcmaps.get_color(0),
                     alpha=0.3,
                     )
            ax.set_xlabel(r"Time \textbf{[\si{\milli\second}]}")
        # Save plt.
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # }}} 
        # {{{ Plot Ext. Elec. Fourier
        freq = data['freq']
        amps = data['amp']
        if self.plot_param['freq_end'] is not None:
            idx = min(
                range(len(freq)), 
                key=lambda i: abs(freq[i] - self.plot_param['freq_end'])
                )
            freq = freq[0:idx]
            amps = amps[:,:idx]
        for i in xrange(run_param['N']):
            amp = amps[i]
            fname = self.name + '_ext'+str(i+1) + "_fourier"
            print "plotting            :", fname
            plt.figure(figsize=lplot.size_common)
            ax = plt.gca()
            lplot.nice_axes(ax)
            # Plot
            plt.plot(freq, amp, color=lcmaps.get_color(0))
            ax.set_ylabel(r'Amplitude \textbf{[\si{\milli\volt}]}')
            ax.set_xlabel(r'Frequency \textbf{[\si{\kilo\hertz}]}')
            # Save plt.
            lplot.save_plt(plt, fname, dir_plot)
            plt.close()
        # }}} 
        # {{{ Plot soma voltage.
        fname = self.name + '_soma_mem'
        print "plotting            :", fname
        fig = plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['spikes_t_vec'],
                 data['spike_soma'],
                 color=lcmaps.get_color(0),
                 )
        plt.plot(data['spikes_t_vec'],
                 data['width_I_trace_soma'],
                 color=lcmaps.get_color(0.5),
                 )
        plt.plot(data['spikes_t_vec'],
                 data['width_II_trace_soma'],
                 color=lcmaps.get_color(0.5),
                 )

        # Plot annotations type I.
        # Get linesegments from the trace.
        lines_x, lines_y = lplot._get_line_segments(
                data['spikes_t_vec'], 
                data['width_II_trace_soma'],
                )
        width = round(data['width_II_soma'], 3)
        plt.hold('on')
        text = r"\SI{" + str(width) + "}{\milli\second}"
        ax.annotate(text,
                    xy=(lines_x[0, 0] + 0.5*width, lines_y[0, 0]),
                    xycoords='data',
                    xytext=(20, -20),
                    textcoords='offset points',
                    va="center",
                    ha="left",
                    bbox=dict(boxstyle="round4",
                              fc="w"),
                    arrowprops=dict(arrowstyle="-|>",
                                    connectionstyle="arc3,rad=-0.2",
                                    fc="w"), )

        # Plot annotations type II.
        # Get linesegments from the trace.
        lines_x, lines_y = lplot._get_line_segments(
                data['spikes_t_vec'], 
                data['width_I_trace_soma'],
                )
        width = round(data['width_I_soma'], 3)
        plt.hold('on')
        text = r"\SI{" + str(width) + r"}{\milli\second}"
        ax.annotate(text,
                    xy=(lines_x[0, 0] + 0.5*width, lines_y[0, 0]),
                    xycoords='data',
                    xytext=(20, -20),
                    textcoords='offset points',
                    va="center",
                    ha="left",
                    bbox=dict(boxstyle="round4",
                              fc="w"),
                    arrowprops=dict(arrowstyle="-|>",
                                    connectionstyle="arc3,rad=-0.2",
                                    fc="w"), )
        ax.set_ylabel(r"Membrane Potential \textbf{[\si{\milli\volt}]}")
        ax.set_xlabel(r"Time \textbf{[\si{\milli\second}]}")
        # Save plt.
        lplot.save_plt(plt, fname, dir_plot)
        # Change size and save again.
        fig.set_size_inches(lplot.size_square)
        lplot.save_plt(plt, fname+'_small', dir_plot)
        plt.close()
        # }}} 
        # {{{ Plot Ext. Elec.
        for i in xrange(run_param['N']):
            fname = self.name + '_ext'+str(i+1)
            print "plotting            :", fname
            plt.figure(figsize=lplot.size_common)
            ax = plt.gca()
            lplot.nice_axes(ax)
            # Plot
            plt.plot(data['spikes_t_vec'],
                     data['spikes'][i],
                     color=lcmaps.get_color(0),
                     )
            plt.plot(data['spikes_t_vec'],
                     data['width_I_trace'][i],
                     color=lcmaps.get_color(0.5),
                     )
            plt.plot(data['spikes_t_vec'],
                     data['width_II_trace'][i],
                     color=lcmaps.get_color(0.5),
                     )
            ax.set_ylabel(r"Potential \textbf{[\si{\micro\volt}]}")
            ax.set_xlabel(r"Time \textbf{[\si{\milli\second}]}")
            # Save plt.
            lplot.save_plt(plt, fname, dir_plot)
            plt.close()

            fname = self.name + '_ext'+str(i+1)+'_full'
            print "plotting            :", fname
            plt.figure(figsize=lplot.size_common)
            ax = plt.gca()
            lplot.nice_axes(ax)
            # Plot
            plt.plot(data['t_vec'],
                     data['LFP'][i],
                     color=lcmaps.get_color(0),
                     )
            ax.set_ylabel(r"Potential \textbf{[\si{\micro\volt}]}")
            ax.set_xlabel(r"Time \textbf{[\si{\milli\second}]}")
            # Save plt.
            lplot.save_plt(plt, fname, dir_plot)
            plt.close()
        # }}} 
        # {{{ Morphology
        LFPy_util.plot.morphology(data['poly_morph'],
                                  data['poly_morph_axon'],
                                  elec_x=data['elec_x'],
                                  elec_y=data['elec_y'],
                                  fig_size=lplot.size_square,
                                  fname=self.name + "_morph_elec_xy",
                                  plot_save_dir=dir_plot,
                                  show=False,
                                  numbering=True,
                                  )
        # }}} 
        # {{{ Morphology xz
        LFPy_util.plot.morphology(data['poly_morph_xz'],
                                  data['poly_morph_axon_xz'],
                                  elec_x=data['elec_x'],
                                  elec_y=data['elec_z'],
                                  x_label='x',
                                  y_label='z',
                                  fig_size=lplot.size_square,
                                  fname=self.name + "_morph_elec_xz",
                                  plot_save_dir=dir_plot,
                                  show=False,
                                  numbering=True,
                                  )
        # }}} 
