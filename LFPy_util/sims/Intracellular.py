from Simulation import Simulation
import LFPy
import LFPy_util
import numpy as np

class Intracellular(Simulation):
    """docstring for Grid"""

    def __init__(self):
        super(Intracellular,self).__init__()
        # Used by the super save and load function.
        self.fname_run_param = 'soma_run_param'
        self.fname_results = 'soma_results'

        self.debug = False

        # Used by the custom plot function.
        self.show = False

        # Plot names.
        self.fname_intra_plot             = 'intra_soma_mem'
        self.fname_intra_plot_fourier     = 'intra_soma_mem_fourier'
        self.fname_intra_plot_i_mem_v_mem = 'intra_i_mem_v_mem'

    def __str__(self):
        return "Intracellular"

    def simulate(self):
        results = self.results
        run_param = self.run_param

        v_vec_list, i_vec_list, t_vec, rec_pos = \
                LFPy_util.data_extraction.rec_along_longest_branch()

        self.cell.simulate(rec_vmem=True,rec_imem=True,rec_istim=True,rec_isyn=True)

        self.results['soma_v']    = self.cell.somav
        self.results['soma_t']    = self.cell.tvec
        self.results['dt']        = self.cell.timeres_NEURON
        self.results['v_vec_list']    = v_vec_list
        self.results['i_vec_list']    = i_vec_list
        self.results['t_vec']         = t_vec.as_numpy()
        self.results['rec_pos']       = rec_pos
        self.results['poly_morph']    = self.cell.get_idx_polygons(('x','y'))

        # The first value of t_vec is 0 even though it should not be. 
        self.results['t_vec'] = np.delete(self.results['t_vec'],0)
        self.results['v_vec_list'] = np.delete(self.results['v_vec_list'],0,1)
        self.results['i_vec_list'] = np.delete(self.results['i_vec_list'],0,1)
        self.results['rec_x'] = self.results['rec_pos'][:,0]
        self.results['rec_y'] = self.results['rec_pos'][:,1]

        # Gather data for the fourier specter.
        soma_t = self.results['soma_t']
        soma_v = self.results['soma_v']
        freqs, amps, phase = \
            LFPy_util.data_extraction.findFreqAndFft(soma_t,soma_v)
        # Remove the first coefficient as we don't care about the baseline.
        freqs = np.delete(freqs,0)
        amps = np.delete(amps,0)
        self.results['freqs']    = freqs
        self.results['amps']     = amps
        self.results['phase']    = phase

    def plot(self):
        results = self.results
        run_param = self.run_param

        LFPy_util.plot.soma(
            results['soma_t'],
            results['soma_v'],
            self.fname_intra_plot, 
            plot_save_dir=self.dir_plot,
            show=self.show
        )

        # Plot along the longest branch.
        LFPy_util.plot.scattered_i_mem_v_mem(
            results['v_vec_list'], 
            results['i_vec_list'],
            results['t_vec'],
            results['rec_x'], 
            results['rec_y'],
            results['poly_morph'],
            self.fname_intra_plot_i_mem_v_mem, 
            plot_save_dir=self.dir_plot,
            show=self.show,
        )

        LFPy_util.plot.fourierSpecter(
                results['freqs'], 
                results['amps'],
                fname=self.fname_intra_plot_fourier,
                plot_save_dir=self.dir_plot,
                f_end=3,
                show=self.show
        )
