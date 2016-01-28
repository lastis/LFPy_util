"""
Grid simulation.
"""
import numpy as np
import LFPy
import LFPy_util
import LFPy_util.data_extraction as de
from LFPy_util.sims.Simulation import Simulation


class Grid(Simulation):
    """docstring for Grid"""

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.name = 'grid_{}_{}'.format('x', 'y')

        # Used by the custom simulate and plot function.
        self.run_param['plane'] = ['x', 'y']
        self.run_param['n_elec_x'] = 11
        self.run_param['n_elec_y'] = 11
        self.run_param['x_lim'] = [-100, 100]
        self.run_param['y_lim'] = [-100, 100]
        self.run_param['amp_option'] = 'neg'

        # Used by the custom plot function.
        self.show = False
        self.fname_grid_plot_elec_morph          \
                = 'grid_{}_{}_elec_morph'.format('x','y')
        self.fname_grid_plot_signals_2d          \
                = 'grid_{}_{}_signals_2d'.format('x','y')
        self.fname_grid_plot_signals_2d_norm     \
                = 'grid_{}_{}_signals_2d_normalized'.format('x','y')
        self.fname_grid_plot_gradient_2d         \
                = 'grid_{}_{}_gradient_2d'.format('x','y')
        self.fname_grid_plot_gradient_2d_anim    \
                = 'grid_{}_{}_gradient_2d_anim'.format('x','y')

    def __str__(self):
        return "Grid {} {}".format(*self.run_param['plane'])

    def set_plane(self, axis1, axis2):
        self.run_param['plane'] = [axis1, axis2]
        self.ID = 'grid_{}_{}'.format(axis1, axis2)
        self.fname_grid_plot_elec_morph          \
                = 'grid_{}_{}_elec_morph'.format(axis1,axis2)
        self.fname_grid_plot_signals_2d          \
                = 'grid_{}_{}_signals_2d'.format(axis1,axis2)
        self.fname_grid_plot_signals_2d_norm     \
                = 'grid_{}_{}_signals_2d_normalized'.format(axis1,axis2)
        self.fname_grid_plot_gradient_2d         \
                = 'grid_{}_{}_gradient_2d'.format(axis1,axis2)
        self.fname_grid_plot_gradient_2d_anim    \
                = 'grid_{}_{}_gradient_2d_anim'.format(axis1,axis2)

    def simulate(self, cell):
        run_param = self.run_param
        cell = cell
        # Calculate the electrode linspaces for the two planes.
        cnt = 0
        n_elec = run_param['n_elec_x']
        pos_min = run_param['x_lim'][0]
        pos_max = run_param['x_lim'][1]
        plane = run_param['plane']
        x = [0]
        if plane[0] == 'x' or plane[1] == 'x':
            x = np.linspace(pos_min, pos_max, n_elec)
            cnt += 1
        if cnt == 1:
            n_elec = run_param['n_elec_y']
            pos_min = run_param['y_lim'][0]
            pos_max = run_param['y_lim'][1]
        y = [0]
        if plane[0] == 'y' or plane[1] == 'y':
            y = np.linspace(pos_min, pos_max, n_elec)
            cnt += 1
        if cnt == 1:
            n_elec = run_param['n_elec_y']
            pos_min = run_param['y_lim'][0]
            pos_max = run_param['y_lim'][1]
        z = [0]
        if plane[0] == 'z' or plane[1] == 'z':
            z = np.linspace(pos_min, pos_max, n_elec)
            cnt += 1
        if cnt != 2:
            raise ValueError('Plane description not accepted.')

        # Simulate and store the currents.
        cell.simulate(rec_vmem=True,
                      rec_imem=True,
                      rec_istim=True,
                      rec_isyn=True)

        electrode_dict = LFPy_util.electrodes.gridElectrodes(x, y, z)

        electrode_dict['sigma'] = 0.3
        electrode = LFPy.RecExtElectrode(cell, **electrode_dict)
        electrode.calc_lfp()

        self.data['electrode_dict'] = electrode_dict
        self.data['LFP'] = electrode.LFP
        self.data['lin_x'] = x
        self.data['lin_y'] = y
        self.data['lin_z'] = z
        self.data['dt'] = cell.timeres_NEURON
        self.data['t_vec'] = cell.tvec

        self.data['poly_morph'] \
                = de.get_polygons_no_axon(cell, self.run_param['plane'])
        self.data['poly_morph_axon'] \
                = de.get_polygons_axon(cell, self.run_param['plane'])

    def process_data(self):
        pass

    def plot(self, dir_plot):
        data = self.data
        run_param = self.run_param

        LFPy_util.plot.morphology(data['poly_morph'],
                                  data['poly_morph_axon'],
                                  elec_x=data['electrode_dict']['x'],
                                  elec_y=data['electrode_dict']['y'],
                                  fig_size='square',
                                  fname=self.fname_grid_plot_elec_morph,
                                  plot_save_dir=dir_plot,
                                  show=self.show, )

        LFPy_util.plot.signals2D(data['LFP'],
                                 data['lin_x'],
                                 data['lin_y'],
                                 poly_morph=data['poly_morph'],
                                 normalization=False,
                                 amp_option=run_param['amp_option'],
                                 fname=self.fname_grid_plot_signals_2d,
                                 show=self.show,
                                 plot_save_dir=dir_plot)
        LFPy_util.plot.signals2D(data['LFP'],
                                 data['lin_x'],
                                 data['lin_y'],
                                 poly_morph=data['poly_morph'],
                                 normalization=True,
                                 amp_option=run_param['amp_option'],
                                 fname=self.fname_grid_plot_signals_2d_norm,
                                 show=self.show,
                                 plot_save_dir=dir_plot)
