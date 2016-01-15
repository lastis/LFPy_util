from Simulation import Simulation
import LFPy
import LFPy_util
import numpy as np


class Grid(Simulation):
    """docstring for Grid"""

    def __init__(self):
        super(Grid,self).__init__()
        # Used by the super save and load function.
        self.fname_run_param = 'grid_{}_{}_run_param'.format('x','y')
        self.fname_results = 'grid_{}_{}_results'.format('x','y')
        
        # Used by the custom simulate and plot function.
        self.run_param['plane'] = ['x','y']
        self.run_param['n_elec_x'] = 11
        self.run_param['n_elec_y'] = 11
        self.run_param['x_lim'] = [-100,100]
        self.run_param['y_lim'] = [-100,100]
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

    def _set_plane(self,axis1,axis2):
        self.run_param['plane'] = [axis1,axis2]
        self.fname_run_param = 'grid_{}_{}_run_param'.format(axis1,axis2)
        self.fname_results = 'grid_{}_{}_results'.format(axis1,axis2)
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

    def simulate(self):
        run_param = self.run_param
        cell = self.cell
        # Calculate the electrode linspaces for the two planes.
        cnt = 0
        n_elec = run_param['n_elec_x']
        pos_min = run_param['x_lim'][0]
        pos_max = run_param['x_lim'][1]
        plane = run_param['plane']
        x = [0]
        if plane[0] == 'x' or plane[1] == 'x':
            x = np.linspace(pos_min,pos_max,n_elec)
            cnt += 1
        if cnt == 1:
            n_elec = run_param['n_elec_y']
            pos_min = run_param['y_lim'][0]
            pos_max = run_param['y_lim'][1]
        y = [0]
        if plane[0] == 'y' or plane[1] == 'y':
            y = np.linspace(pos_min,pos_max,n_elec)
            cnt += 1
        if cnt == 1:
            n_elec = run_param['n_elec_y']
            pos_min = run_param['y_lim'][0]
            pos_max = run_param['y_lim'][1]
        z = [0]
        if plane[0] == 'z' or plane[1] == 'z':
            z = np.linspace(pos_min,pos_max,n_elec)
            cnt += 1
        if cnt != 2: 
            raise ValueError('Plane description not accepted.')

        # Simulate and store the currents. 
        cell.simulate(rec_vmem=True,rec_imem=True,rec_istim=True,rec_isyn=True)

        electrode_dict = LFPy_util.electrodes.gridElectrodes(x,y,z)

        electrode_dict['sigma'] = 0.3
        electrode = LFPy.RecExtElectrode(cell, **electrode_dict)
        electrode.calc_lfp()

        self.results['electrode_dict'] = electrode_dict
        self.results['LFP']    = electrode.LFP
        self.results['lin_x']     = x
        self.results['lin_y']     = y
        self.results['lin_z']     = z
        self.results['dt']           = cell.timeres_NEURON
        self.results['poly_morph']   = cell.get_idx_polygons(plane)
        self.results['t_vec']         = cell.tvec

    def plot(self):
        results = self.results
        run_param = self.run_param

        LFPy_util.plot.morphology(
            results['poly_morph'],
            elec_x = results['electrode_dict']['x'],
            elec_y = results['electrode_dict']['y'],
            fig_size='square',
            fname=self.fname_grid_plot_elec_morph,
            plot_save_dir=self.dir_plot,
            show=self.show,
        )

        LFPy_util.plot.signals2D(
                results['LFP'], 
                results['lin_x'], 
                results['lin_y'], 
                poly_morph = results['poly_morph'],
                normalization=False, 
                amp_option=run_param['amp_option'],
                fname=self.fname_grid_plot_signals_2d,
                show=self.show, 
                plot_save_dir=self.dir_plot
        )
        LFPy_util.plot.signals2D(
                results['LFP'], 
                results['lin_x'], 
                results['lin_y'], 
                poly_morph = results['poly_morph'],
                normalization=True, 
                amp_option=run_param['amp_option'],
                fname=self.fname_grid_plot_signals_2d_norm,
                show=self.show, 
                plot_save_dir=self.dir_plot
        )

