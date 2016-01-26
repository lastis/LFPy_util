from Simulation import Simulation
import LFPy
import LFPy_util
import LFPy_util.data_extraction as de
import numpy as np
import os

class DiscElectrodes(Simulation):
    """docstring for Grid"""

    def __init__(self):
        Simulation.__init__(self)

        # Used by the custom simulate and plot function.
        self.run_param['r'] = 100
        self.run_param['n'] = 9
        self.run_param['n_theta'] = 8
        self.run_param['r_0'] = 20
        self.run_param['threshold'] = 3
        self.run_param['pre_dur'] = 16.7*0.5
        self.run_param['post_dur'] = 16.7*0.5
        self.run_param['amp_option'] = 'neg'
        self.run_param['plane'] = ['x','z']
        self.wave_def = 'half_amp'
        self.debug = False
        self.plot_detailed = False

        # Used by the custom plot function.
        self.show = False

        self._set_names()

    def __str__(self):
        return "DiskElectrodes {} {}".format(*self.run_param['plane'])

    def set_plane(self,axis1,axis2):
        self.run_param['plane'] =[axis1,axis2]
        self._set_names()

    def _set_names(self):
        self.ID = 'disc_{}_{}'.format(*self.run_param['plane'])
        self.fname_disc_plot                     \
                = 'disc_{}_{}_gradient'.format(*self.run_param['plane'])
        self.fname_disc_plot_spike_amp_all       \
                = 'disc_{}_{}_spike_amp_all'.format(*self.run_param['plane'])
        self.fname_disc_plot_spike_amp_all_log   \
                = 'disc_{}_{}_spike_amp_all_log'.format(*self.run_param['plane'])
        self.fname_disc_plot_spike_amp_std       \
                = 'disc_{}_{}_spike_amp_std'.format(*self.run_param['plane'])
        self.fname_disc_plot_spike_amp_std_log   \
                = 'disc_{}_{}_spike_amp_std_log'.format(*self.run_param['plane'])
        self.fname_disc_plot_spike_width_all     \
                = 'disc_{}_{}_spike_width_all'.format(*self.run_param['plane'])
        self.fname_disc_plot_spike_width_all_log \
                = 'disc_{}_{}_spike_width_all_log'.format(*self.run_param['plane'])
        self.fname_disc_plot_spike_width_std     \
                = 'disc_{}_{}_spike_width_std'.format(*self.run_param['plane'])
        self.fname_disc_plot_spike_width_std_log \
                = 'disc_{}_{}_spike_width_std_log'.format(*self.run_param['plane'])
        self.fname_disc_plot_elec_signal         \
                = 'disc_{}_{}_elec_signal'.format(*self.run_param['plane'])
        self.fname_disc_plot_elec_signal_3       \
                = 'disc_{}_{}_elec_signal_3'.format(*self.run_param['plane'])
        self.fname_disc_plot_elec_morph          \
                = 'disc_{}_{}_elec_morph'.format(*self.run_param['plane'])


    def simulate(self,cell):
        run_param = self.run_param
        data = self.data
        cell = cell

        cell.simulate(rec_vmem=True,rec_imem=True,rec_istim=True,rec_isyn=True)

        # Create electrodes.
        electrode_func = None
        x = False
        y = False
        z = False
        plane = run_param['plane']
        cnt = 0
        if plane[0] == 'x' or plane[1] == 'x':
            x = True
            cnt += 1
        if plane[0] == 'y' or plane[1] == 'y':
            y = True
            cnt += 1
        if plane[0] == 'z' or plane[1] == 'z':
            z = True
            cnt += 1
        if cnt != 2: 
            raise ValueError('Plane description not accepted.')
        if x and z:
            # print "xz"
            electrode_func = LFPy_util.electrodes.circularElectrodesXZ
        elif x and y:
            # print "xy"
            electrode_func = LFPy_util.electrodes.circularElectrodesXY
        elif y and z:
            # print "yz"
            electrode_func = LFPy_util.electrodes.circularElectrodesYZ

        electrode_dict = electrode_func(
                r = run_param['r'],
                n = run_param['n'],
                n_theta = run_param['n_theta'],
                r_0 = run_param['r_0'],
                x_0=[0,0,0]
        ) 
        electrode_dict['sigma'] = 0.3
        # Record the LFP of the electrodes. 
        electrode = LFPy.RecExtElectrode(cell, **electrode_dict)
        electrode.calc_lfp()

        data['electrode_dict'] = electrode_dict
        data['LFP']     = electrode.LFP
        data['dt']           = cell.timeres_NEURON
        data['t_vec']        = cell.tvec
        data['poly_morph'] \
                = de.get_polygons_no_axon(cell,self.run_param['plane'])
        data['poly_morph_axon'] \
                = de.get_polygons_axon(cell,self.run_param['plane'])

    def process_data(self):
        data = self.data
        run_param = self.run_param
        data['LFP']  = np.array(data['LFP'])
        r       = run_param['r']
        n       = run_param['n']
        r_0     = run_param['r_0']
        n_theta = run_param['n_theta']
        LFP     = data['LFP']
        dt      = data['dt']

        data['dr'] = (r-r_0)/(n-1)

        electrode_spikes = []
        electrode_t_vec = []
        # For each angle around the center point, there are multiple electrodes.
        for i in xrange(n_theta):
            # Get the signals from the electrodes.
            row_start = i*n
            row_end = row_start + n
            mat = LFP[row_start:row_end,:]

            # Extract the spikes from the closest electrode.
            spikes, t_vec_spike, I = LFPy_util.data_extraction.extract_spikes(
                data['t_vec'],
                mat[0],
                pre_dur=run_param['pre_dur'],
                post_dur=run_param['post_dur'],
                threshold=run_param['threshold'],
                amp_option=run_param['amp_option'],
            )
            if len(spikes) == 0:
                if self.debug:
                    print "Warning (DiscElectrodes): no spikes found!"
                # Store the signal anyway.
                electrode_spikes.append(mat)
                electrode_t_vec.append(data['t_vec'])
            else:
                # Use the same time interval for all electrodes, taken from the 
                # first spike.
                mat = mat[:,I[0,0]:I[0,1]]
                # Store the spike signals.
                electrode_spikes.append(mat)
                electrode_t_vec.append(t_vec_spike)
        data['electrode_t_vec'] = electrode_t_vec
        data['electrode_spikes'] = electrode_spikes
        # Create a vector of the radial positions of the electrodes.
        data['electrode_pos_r'] = np.linspace(r_0,r,n)

        # Spike widths.
        if self.wave_def == 'top_bottom':
            widths, trace = de.find_wave_width_type_1(
                    LFP,
                    dt=dt,
            )
        else:
            widths, trace = de.findWaveWidthsSimple(
                    LFP,
                    dt=dt,
                    amp_option=run_param['amp_option']
            )

        widths = np.reshape(widths,[n_theta,n])
        data['widths'] = widths

        # Spike amplitudes.
        amps = LFPy_util.data_extraction.find_amplitude(
                LFP,
                run_param['amp_option']
        )
        amps = np.reshape(amps,[n_theta,n])
        data['amps'] = amps

    def plot(self,dir_plot):
        run_param = self.run_param
        data = self.data

        # Plot the signals from the electrodes in a circular shape.
        if self.plot_detailed:
            for i in xrange(run_param['n_theta']):
                # Create directory and filename. 
                name =  ('n_theta_%03d' %(i))
                directory = dir_plot+'/'+self.fname_disc_plot_elec_signal
                if len(data['electrode_spikes'][0]) == 0:
                    if self.debug:
                        print "Warning (DiscElectrodes): no spike data found!"
                    continue
                LFPy_util.plot.electrodeSignals(
                        data['electrode_t_vec'][i],
                        data['electrode_spikes'][i],
                        elec_pos=data['electrode_pos_r'],
                        fname=name,
                        show=self.show,
                        plot_save_dir=directory,
                )

                # Also plot the first, middel and last electrode signal by themselves
                # in a third plot.
                mat = data['electrode_spikes'][i]
                n_middle = mat.shape[0]/2
                new_mat = mat[[0,n_middle,-1],:]
                # Create directory and filename. 
                new_name = 'n_theta_%03d' %(i)
                new_directory = dir_plot+'/'+self.fname_disc_plot_elec_signal_3

                # Show the width calculation also.
                if self.wave_def == 'top_bottom':
                    widths, trace = de.find_wave_width_type_1(
                            new_mat,
                            dt=data['dt'],
                    )
                else:
                    widths, trace = de.findWaveWidthsSimple(
                            new_mat,
                            dt=data['dt'],
                            amp_option=run_param['amp_option']
                    )
                LFPy_util.plot.signal_sub_plot_3(
                    data['electrode_t_vec'][i],
                    new_mat,
                    fname=new_name,
                    show=self.show,
                    plot_save_dir=new_directory,
                    width_trace=trace
                )

        LFPy_util.plot.morphology(
            data['poly_morph'],
            data['poly_morph_axon'],
            elec_x = data['electrode_dict'][run_param['plane'][0]],
            elec_y = data['electrode_dict'][run_param['plane'][1]],
            fig_size='square',
            fname=self.fname_disc_plot_elec_morph,
            plot_save_dir=dir_plot,
            show=self.show,
        )

        LFPy_util.plot.spikeAmplitudes(
                amps = data['amps'], 
                dr = data['dr'], 
                r_0 = run_param['r_0'],
                fname = self.fname_disc_plot_spike_amp_all,
                mode = 'all',
                scale = 'linear',
                show_points=True,
                show=self.show,
                plot_save_dir=dir_plot
        )
        LFPy_util.plot.spikeAmplitudes(
                amps = data['amps'], 
                dr = data['dr'], 
                r_0 = run_param['r_0'],
                fname = self.fname_disc_plot_spike_amp_std,
                mode = 'std',
                scale = 'linear',
                show_points=True,
                show=self.show,
                plot_save_dir=dir_plot
        )
        LFPy_util.plot.spikeAmplitudes(
                amps = data['amps'], 
                dr = data['dr'], 
                r_0 = run_param['r_0'],
                fname = self.fname_disc_plot_spike_amp_all_log,
                mode = 'all',
                scale = 'log',
                show_points=True,
                show=self.show,
                plot_save_dir=dir_plot
        )
        LFPy_util.plot.spikeAmplitudes(
                amps = data['amps'], 
                dr = data['dr'], 
                r_0 = run_param['r_0'],
                fname = self.fname_disc_plot_spike_amp_std_log,
                mode = 'std',
                scale = 'log',
                show_points=True,
                show=self.show,
                plot_save_dir=dir_plot
        )

        LFPy_util.plot.spikeWidths(
                widths = data['widths'], 
                dr = data['dr'], 
                r_0 = run_param['r_0'],
                fname = self.fname_disc_plot_spike_width_all,
                mode = 'all',
                scale = 'linear',
                show_points=True,
                show=self.show,
                plot_save_dir=dir_plot
        )
        LFPy_util.plot.spikeWidths(
                widths = data['widths'], 
                dr = data['dr'], 
                r_0 = run_param['r_0'],
                fname = self.fname_disc_plot_spike_width_std,
                mode = 'std',
                scale = 'linear',
                show_points=True,
                show=self.show,
                plot_save_dir=dir_plot
        )
        LFPy_util.plot.spikeWidths(
                widths = data['widths'], 
                dr = data['dr'], 
                r_0 = run_param['r_0'],
                fname = self.fname_disc_plot_spike_width_all_log,
                mode = 'all',
                scale = 'log',
                show_points=True,
                show=self.show,
                plot_save_dir=dir_plot
        )
        LFPy_util.plot.spikeWidths(
                widths = data['widths'], 
                dr = data['dr'], 
                r_0 = run_param['r_0'],
                fname = self.fname_disc_plot_spike_width_std_log,
                mode = 'std',
                scale = 'log',
                show_points=True,
                show=self.show,
                plot_save_dir=dir_plot
        )
