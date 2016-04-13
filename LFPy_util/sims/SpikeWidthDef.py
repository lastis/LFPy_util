import os
import numpy as np
import LFPy
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import LFPy_util
import LFPy_util.data_extraction as de
import LFPy_util.plot as lplot
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
        self.run_param['r'] = 30
        self.run_param['seed'] = 1234

        self.process_param['amp_option'] = 'both'
        self.process_param['pre_dur'] = 16.7 * 0.5
        self.process_param['post_dur'] = 16.7 * 0.5
        self.process_param['threshold'] = 4
        # Index of the spike to measure from.
        self.process_param['spike_to_measure'] = 0

    def simulate(self, cell):
        # pylint: disable=invalid-name,no-member
        data = self.data
        run_param = self.run_param
        cell.simulate(rec_imem=True)

        angle = np.random.uniform(0, 2*np.pi)
        z = np.random.uniform(-1, 1)
        x = np.sqrt(1-z*z)*np.cos(angle)
        y = np.sqrt(1-z*z)*np.sin(angle)

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
        data['poly_morph_xz'] \
                = de.get_polygons_no_axon(cell,['x','z'])
        data['poly_morph_axon_xz'] \
                = de.get_polygons_axon(cell,['x','z'])

    def process_data(self):
        data = self.data
        run_param = self.run_param
        process_param = self.process_param

        # Get the signal from the soma potential.
        # signal = data['LFP'][0]
        signal = data['soma_v']
        spikes, spikes_t_vec, I = de.extract_spikes(
            data['t_vec'],
            signal,
            pre_dur=process_param['pre_dur'],
            post_dur=process_param['post_dur'],
            threshold=process_param['threshold'],
            amp_option=process_param['amp_option'], 
            )
        data['spike'] = spikes[run_param['spike_index']]
        data['spike_t_vec'] = spikes_t_vec

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

        # {{{ Plot 1
        fname = self.name + '_extracellular'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        plt.plot(data['spike_t_vec'],
                 data['spike'],
                 color=lcmaps.get_color(0),
                 marker='o',
                 markersize=5)
        # Save plt.
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # }}} 

