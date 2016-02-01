"""
Grid simulation.
"""
import numpy as np
import matplotlib.pyplot as plt
import LFPy
import LFPy_util
import LFPy_util.data_extraction as de
from LFPy_util.sims.Simulation import Simulation


class Grid(Simulation):
    """docstring for Grid"""

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.name = 'grid_x_y'

        # Used by the custom simulate and plot function.
        self.run_param['n_elec_x'] = 11
        self.run_param['n_elec_y'] = 11
        self.run_param['x_lim'] = [-100, 100]
        self.run_param['y_lim'] = [-100, 100]
        self.run_param['amp_option'] = 'both'

    def __str__(self):
        return "Grid x y"

    def simulate(self, cell):
        run_param = self.run_param
        cell = cell
        # Calculate the electrode linspaces for the two planes.
        n_elec = run_param['n_elec_x']
        pos_min = run_param['x_lim'][0]
        pos_max = run_param['x_lim'][1]
        lin_x = np.linspace(pos_min, pos_max, n_elec)

        n_elec = run_param['n_elec_y']
        pos_min = run_param['y_lim'][0]
        pos_max = run_param['y_lim'][1]
        lin_y = np.linspace(pos_min, pos_max, n_elec)

        # Simulate and store the currents.
        cell.simulate(rec_vmem=True,
                      rec_imem=True,
                      rec_istim=True,
                      rec_isyn=True)

        electrode_dict = \
            LFPy_util.electrodes.grid_electrodes(lin_x, lin_y, [0])

        electrode_dict['sigma'] = 0.3
        electrode = LFPy.RecExtElectrode(cell, **electrode_dict)
        electrode.calc_lfp()

        self.data['electrode_dict'] = electrode_dict
        self.data['LFP'] = electrode.LFP
        self.data['lin_x'] = lin_x
        self.data['lin_y'] = lin_y
        self.data['dt'] = cell.timeres_NEURON
        self.data['t_vec'] = cell.tvec

        self.data['poly_morph'] = de.get_polygons_no_axon(cell, ['x', 'y'])
        self.data['poly_morph_axon'] = de.get_polygons_axon(cell, ['x', 'y'])

    def process_data(self):
        pass

    def plot(self, dir_plot):
        data = self.data
        run_param = self.run_param

        fname = 'grid_x_y_elec_morph'
        LFPy_util.plot.morphology(data['poly_morph'],
                                  data['poly_morph_axon'],
                                  elec_x=data['electrode_dict']['x'],
                                  elec_y=data['electrode_dict']['y'],
                                  fig_size='square',
                                  fname=fname,
                                  plot_save_dir=dir_plot,
                                  show=False, )

        fname = 'grid_x_y_signals_2d'
        LFPy_util.plot.signals2D(data['LFP'],
                                 data['lin_x'],
                                 data['lin_y'],
                                 poly_morph=data['poly_morph'],
                                 normalization=False,
                                 amp_option=run_param['amp_option'],
                                 fname=fname,
                                 show=False,
                                 plot_save_dir=dir_plot)
        fname = 'grid_y_x_signals_2d_normalized'
        LFPy_util.plot.signals2D(data['LFP'],
                                 data['lin_x'],
                                 data['lin_y'],
                                 poly_morph=data['poly_morph'],
                                 normalization=True,
                                 amp_option=run_param['amp_option'],
                                 fname=fname,
                                 show=False,
                                 plot_save_dir=dir_plot)

