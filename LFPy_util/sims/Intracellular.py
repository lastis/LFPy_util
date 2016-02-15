from Simulation import Simulation
import LFPy
import LFPy_util
import numpy as np
from scipy.stats.mstats import zscore


class Intracellular(Simulation):
    """docstring for Grid"""

    def __init__(self):
        Simulation.__init__(self)
        self.name = 'soma'

        self.debug = False

        # Used by the custom plot function.
        self.show = False

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

        # Gather data for the fourier specter.
        soma_t = self.data['soma_t']
        soma_v = self.data['soma_v']
        freqs, amps, phase = \
            LFPy_util.data_extraction.find_freq_and_fft(soma_t,soma_v)
        # Remove the first coefficient as we don't care about the baseline.
        freqs = np.delete(freqs, 0)
        amps = np.delete(amps, 0)
        self.data['freqs'] = freqs
        self.data['amps'] = amps
        self.data['phase'] = phase

    def process_data(self):
        pass

    def plot(self, dir_plot):
        data = self.data
        run_param = self.run_param

        LFPy_util.plot.soma(data['soma_t'],
                            data['soma_v_z'],
                            self.fname_intra_plot_zscore,
                            plot_save_dir=dir_plot,
                            show=self.show)

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

        LFPy_util.plot.fourierSpecter(data['freqs'],
                                      data['amps'],
                                      fname=self.fname_intra_plot_fourier,
                                      plot_save_dir=dir_plot,
                                      f_end=3,
                                      show=self.show)
