"""
Grid simulation.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import LFPy
import LFPy_util
import LFPy_util.data_extraction as de
import LFPy_util.plot as lp
from LFPy_util.sims.Simulation import Simulation


class GridDense(Simulation):
    """docstring for Grid"""

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.set_name("grid_dense_xy")

        # Used by the custom simulate and plot function.
        self.run_param['elec_dx'] = 2
        self.run_param['elec_dy'] = 2
        self.run_param['x_lim'] = [-50, 50]
        self.run_param['y_lim'] = [-50, 50]
        self.run_param['ext_method'] = 'som_as_point'

        self.process_param['amp_option'] = 'both'
        self.process_param['pre_dur'] = 16.7 * 0.5
        self.process_param['post_dur'] = 16.7 * 0.5
        self.process_param['threshold'] = 4
        # Index of the spike to measure from.
        self.process_param['spike_to_measure'] = 0

        self.plot_param['use_tex'] = True

    def simulate(self, cell):
        run_param = self.run_param
        cell = cell
        # Calculate the electrode linspaces for the two planes.
        elec_dx = run_param['elec_dx']
        pos_min = run_param['x_lim'][0]
        pos_max = run_param['x_lim'][1]
        n_elec_x = abs(pos_max - pos_min)/elec_dx + 1
        lin_x = np.linspace(pos_min, pos_max, n_elec_x, endpoint=True)

        elec_dy = run_param['elec_dy']
        pos_min = run_param['y_lim'][0]
        pos_max = run_param['y_lim'][1]
        n_elec_y = abs(pos_max - pos_min)/elec_dy + 1
        lin_y = np.linspace(pos_min, pos_max, n_elec_y, endpoint=True)

        # Simulate and store the currents.
        cell.simulate(rec_imem=True)

        electrode_dict = \
            LFPy_util.electrodes.grid_electrodes(lin_x, lin_y, [0])

        electrode_dict['sigma'] = 0.3
        electrode = LFPy.RecExtElectrode(cell, **electrode_dict)
        electrode.method = run_param['ext_method']
        electrode.calc_lfp()

        self.data['electrode_dict'] = electrode_dict
        self.data['elec_x'] = electrode_dict['x']
        self.data['elec_y'] = electrode_dict['y']
        self.data['n_elec_x'] = n_elec_x
        self.data['n_elec_y'] = n_elec_y
        self.data['LFP'] = electrode.LFP
        self.data['lin_x'] = lin_x
        self.data['lin_y'] = lin_y
        self.data['dt'] = cell.timeres_NEURON
        self.data['t_vec'] = cell.tvec
        self.data['soma_v'] = cell.somav

        self.data['poly_morph'] = de.get_polygons_no_axon(cell, ['x', 'y'])
        self.data['poly_morph_axon'] = de.get_polygons_axon(cell, ['x', 'y'])

    def process_data(self):
        data = self.data
        run_param = self.run_param
        process_param = self.process_param

        LFP_amp = LFPy_util.data_extraction.maxabs(data['LFP'],axis=1)
        LFP_amp = LFP_amp.reshape([data['n_elec_x'],data['n_elec_y']])

        grid_x, grid_y = np.meshgrid(data['lin_x'],data['lin_y'])

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

        widths_I = widths_I.reshape([data['n_elec_x'],data['n_elec_y']])
        widths_II = widths_II.reshape([data['n_elec_x'],data['n_elec_y']])

        data['LFP_amp'] = LFP_amp*1000
        data['grid_x'] = grid_x
        data['grid_y'] = grid_y

        data['amps_I'] = amps_I * 1000

        data['amps_II'] = amps_II * 1000

        data['widths_I'] = widths_I
        data['widths_I_trace'] = widths_I_trace * 1000

        data['widths_II'] = widths_II
        data['widths_II_trace'] = widths_II_trace * 1000

        data['spikes'] = spikes * 1000
        data['spikes_t_vec'] = spikes_t_vec

    def plot(self, dir_plot):
        data = self.data
        run_param = self.run_param

        format = 'pdf'
        # Set global matplotlib parameters.
        LFPy_util.plot.set_rc_param()
        # Create the directory if it does not exist.
        if not os.path.exists(dir_plot):
            os.makedirs(dir_plot)

        # Plot Grid Dense Amp {{{1 # 
        # New plot.
        fname = 'grid_dense_gradient'
        print "plotting            :", fname
        plt.figure(figsize=lp.size_square)
        ax = plt.gca()
        
        LFP_amp = data['LFP_amp']
        # LFP_min, LFP_max = -np.abs(LFP_amp).max(), np.abs(LFP_amp).max()
        LFP_amp = np.abs(LFP_amp)
        LFP_min, LFP_max = np.amin(LFP_amp), np.amax(LFP_amp)
        grid_x = data['grid_x']
        grid_y = data['grid_y']
        pcol = plt.pcolormesh(grid_x, 
                              grid_y, 
                              LFP_amp, 
                              cmap=LFPy_util.colormaps.viridis,
                              norm=colors.LogNorm(vmin=LFP_min, vmax=LFP_max),
                              )
        pcol.set_edgecolor('face')

        ticks = np.logspace(np.log10(LFP_min),np.log10(LFP_max),5,endpoint=True)
        formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False)
        cbar = plt.colorbar(ticks=ticks, format=formatter)
        cbar.solids.set_rasterized(True)
        cbar.update_ticks()

        # Plot morphology.
        zips = []
        for a, b in data['poly_morph']:
            zips.append(zip(a, b))
        polycol = mpl.collections.PolyCollection(zips,
                                                 edgecolors='none',
                                                 facecolor='black',
                                                 alpha = 0.2)
        ax.add_collection(polycol, )
        zips = []
        for a, b in data['poly_morph_axon']:
            zips.append(zip(a, b))
        polycol = mpl.collections.PolyCollection(zips,
                                                 edgecolors='none',
                                                 facecolor='white',
                                                 alpha = 0.2)
        ax.add_collection(polycol, )

        ax.set_ylabel("y")
        ax.set_xlabel("x")
        ax.autoscale(tight=True)
        ax.set_aspect('equal')
        # plt.axis('tight')
        plt.axis([grid_x.min(),grid_x.max(),grid_y.min(),grid_y.max()])
        # Save plt.
        path = os.path.join(dir_plot, fname + "." + format)
        plt.savefig(path, format=format, bbox_inches='tight')
        plt.close()
        #  1}}} # 
        # Plot Grid Dense Width {{{1 # 
        # New plot.
        fname = 'grid_dense_width'
        print "plotting            :", fname
        plt.figure(figsize=lp.size_square)
        ax = plt.gca()
        
        LFP_amp = data['widths_I']
        # LFP_min, LFP_max = -np.abs(LFP_amp).max(), np.abs(LFP_amp).max()
        LFP_amp = np.abs(LFP_amp)
        LFP_min, LFP_max = np.amin(LFP_amp), np.amax(LFP_amp)
        grid_x = data['grid_x']
        grid_y = data['grid_y']
        pcol = plt.pcolormesh(grid_x, 
                              grid_y, 
                              LFP_amp, 
                              cmap=LFPy_util.colormaps.viridis,
                              )
        pcol.set_edgecolor('face')

        cbar = plt.colorbar()
        # ticks = np.logspace(np.log10(LFP_min),np.log10(LFP_max),5,endpoint=True)
        # formatter = mpl.ticker.LogFormatter(10, labelOnlyBase=False)
        # cbar = plt.colorbar(ticks=ticks, format=formatter)
        # cbar.solids.set_rasterized(True)
        # cbar.update_ticks()

        # Plot morphology.
        zips = []
        for a, b in data['poly_morph']:
            zips.append(zip(a, b))
        polycol = mpl.collections.PolyCollection(zips,
                                                 edgecolors='none',
                                                 facecolor='black',
                                                 alpha = 0.2)
        ax.add_collection(polycol, )
        zips = []
        for a, b in data['poly_morph_axon']:
            zips.append(zip(a, b))
        polycol = mpl.collections.PolyCollection(zips,
                                                 edgecolors='none',
                                                 facecolor='white',
                                                 alpha = 0.2)
        ax.add_collection(polycol, )

        ax.set_ylabel("y")
        ax.set_xlabel("x")
        ax.autoscale(tight=True)
        ax.set_aspect('equal')
        # plt.axis('tight')
        plt.axis([grid_x.min(),grid_x.max(),grid_y.min(),grid_y.max()])
        # Save plt.
        path = os.path.join(dir_plot, fname + "." + format)
        plt.savefig(path, format=format, bbox_inches='tight')
        plt.close()
        #  1}}} # 

        LFPy_util.plot.morphology(data['poly_morph'],
                                  data['poly_morph_axon'],
                                  elec_x=data['elec_x'],
                                  elec_y=data['elec_y'],
                                  fig_size=lp.size_common,
                                  fname="grid_dense_morph_elec",
                                  plot_save_dir=dir_plot,
                                  show=False)

