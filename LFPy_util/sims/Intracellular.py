from Simulation import Simulation
import LFPy
import LFPy_util
import LFPy_util.plot as lplot
import LFPy_util.colormaps as lcmaps
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from scipy.stats.mstats import zscore


class Intracellular(Simulation):
    """docstring for Grid"""

    def __init__(self):
        Simulation.__init__(self)
        self.set_name("soma")

        self.debug = False

        # Used by the custom plot function.
        self.show = False

        self.process_param['padding_factor'] = 1

        self.plot_param['freq_end'] = 3*pq.kHz

        # Plot names.
        self.fname_intra_plot = 'intra_soma_mem'
        self.fname_intra_plot_zscore = 'intra_soma_mem_zscore'
        self.fname_intra_plot_fourier = 'intra_soma_mem_fourier'
        self.fname_intra_plot_i_mem_v_mem = 'intra_i_mem_v_mem'

    def simulate(self, cell):
        data = self.data
        run_param = self.run_param

        v_vec_list, i_vec_list, t_vec, rec_pos = \
                LFPy_util.data_extraction.rec_along_longest_branch()

        cell.simulate(rec_vmem=True,
                      rec_imem=True,
                      rec_istim=True,
                      rec_isyn=True)

        self.data['soma_v'] = cell.somav
        self.data['soma_v_z'] = zscore(cell.somav)
        self.data['soma_t'] = cell.tvec
        self.data['dt'] = cell.timeres_NEURON
        self.data['v_vec_list'] = v_vec_list
        self.data['i_vec_list'] = i_vec_list
        self.data['t_vec'] = t_vec.as_numpy()
        self.data['rec_pos'] = rec_pos
        self.data['poly_morph'] = cell.get_idx_polygons(('x', 'y'))

        # The first value of t_vec is 0 even though it should not be.
        self.data['t_vec'] = np.delete(self.data['t_vec'], 0)
        self.data['v_vec_list'] = np.delete(self.data['v_vec_list'], 0, 1)
        self.data['i_vec_list'] = np.delete(self.data['i_vec_list'], 0, 1)
        self.data['rec_x'] = self.data['rec_pos'][:, 0]
        self.data['rec_y'] = self.data['rec_pos'][:, 1]

    def process_data(self):
        # Gather data for the fourier specter.
        data = self.data
        soma_v = data['soma_v']
        length = soma_v.shape[-1]*self.process_param['padding_factor']
        freq, amp, phase = \
            LFPy_util.data_extraction.find_freq_and_fft(data['dt'], soma_v, length)
        # Remove the first coefficient as we don't care about the baseline.
        freq = np.delete(freq, 0)
        amp = np.delete(amp, 0)
        self.data['freq'] = freq * pq.kHz
        self.data['amp'] = amp
        self.data['phase'] = phase

    def plot(self, dir_plot):
        data = self.data
        run_param = self.run_param
        # Set global matplotlib parameters.
        LFPy_util.plot.set_rc_param()

        # Plot zscore of membrane potential {{{ #
        fname = self.name + '_mem_zscore'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        plt.plot(data['soma_t'], data['soma_v_z'], color=lcmaps.get_color(0))
        ax.set_ylabel(r'Membrane Potential Z-score')
        ax.set_xlabel(r'Time \textbf{[\si{\milli\second}]}')
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # }}} 

        LFPy_util.plot.soma(data['soma_t'],
                            data['soma_v'],
                            self.fname_intra_plot,
                            plot_save_dir=dir_plot,
                            show=self.show)

        # Plot along the longest branch.
        LFPy_util.plot.scattered_i_mem_v_mem(data['v_vec_list'],
                                             data['i_vec_list'],
                                             data['t_vec'],
                                             data['rec_x'],
                                             data['rec_y'],
                                             data['poly_morph'],
                                             self.fname_intra_plot_i_mem_v_mem,
                                             plot_save_dir=dir_plot,
                                             show=self.show, )

        # Plot fourier analysis {{{1 #
        fname = self.name + '_fourier'
        # Delete frequencies above the option.
        freq = data['freq']
        amp = data['amp']
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
        ax.set_ylabel(r'Amplitude \textbf{[\si{\milli\volt}]}')
        ax.set_xlabel(r'Frequency \textbf{[\si{\kilo\hertz}]}')
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # 1}}} #
