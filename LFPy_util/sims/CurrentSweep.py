"""
Simulation that calculates the current needed to give one spike over
the simulation time. Also applies that electrode when simulation is finised.
"""
import os
import neuron
import numpy as np
import matplotlib.pyplot as plt
import LFPy
import LFPy_util
import LFPy_util.plot as lplot
import LFPy_util.data_extraction as de
import LFPy_util.colormaps as lcmaps
import warnings

from multiprocessing import Process, Manager, Queue
from LFPy_util.sims import Simulation


class CurrentSweep(Simulation):

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.set_name('sweep')

        # Used by the custom simulate and plot function.
        self.run_param['pptype'] = 'IClamp'
        self.run_param['delay'] = 0
        self.run_param['duration'] = 500
        self.run_param['amp_start'] = 0.0
        self.run_param['amp_end'] = 0.5
        self.run_param['sweeps'] = 20
        self.run_param['processes'] = 4
        self.run_param['elec_dist'] = 20 # um
        self.run_param['sigma'] = 0.3
        self.run_param['ext_method'] = 'som_as_point'
        self.run_param['seed'] = 1234
        self.run_param['n_elec'] = 1
        self.process_param['amp_option'] = 'both'
        self.process_param['pre_dur'] = 4
        self.process_param['post_dur'] = 8
        self.process_param['threshold'] = 4
        self.process_param['threshold_abs_soma'] = 0 # mV
        self.process_param['threshold_abs_elec'] = 10 # uV
        self.plot_param['use_tex'] = True
        self.plot_param['formats'] = ['pdf']
        self.verbose = False

    def simulate(self, cell):
        run_param = self.run_param
        data = self.data

        # String to put before output to the terminal.
        str_start = self.name
        str_start += " "*(20 - len(self.name)) + ": "

        amps = np.linspace(run_param['amp_start'], 
                           run_param['amp_end'], 
                           run_param['sweeps'])

        # Create electrode positions.
        np.random.seed(run_param['seed'])
        angle = np.random.uniform(0, 2*np.pi, run_param['n_elec'])
        z = np.random.uniform(-1, 1, run_param['n_elec']) 
        x = np.sqrt(1-z*z)*np.cos(angle) * run_param['elec_dist']
        y = np.sqrt(1-z*z)*np.sin(angle) * run_param['elec_dist']
        z = z * run_param['elec_dist']
        
        input_queue = Queue()
        output_queue = Queue()
        workers = []
        for i in xrange(run_param['processes']):
            worker = Worker(
                    input_queue, 
                    output_queue, 
                    cell, 
                    run_param,
                    x,
                    y, 
                    z)
            worker.start()
            workers.append(worker)
        for i, amp in enumerate(amps.tolist()):
            input_queue.put((i, amp))
        for i in xrange(run_param['processes']):
            input_queue.put(None)

        # Shape: 1d [signal]
        v_vec_soma = [None] * run_param['sweeps']
        # shape: 2d [n_elec x signal]
        v_vec_elec = [None] * run_param['sweeps']

        sweep = 0
        while sweep < run_param['sweeps']:
            output_data = output_queue.get()
            if output_data == None:
                sweep += 1
            else:
                i = output_data[0]
                string_id = output_data[1]
                value = output_data[2]
                if string_id == 'somav':
                    v_vec_soma[i] = value
                elif string_id == 'lfp':
                    v_vec_elec[i] = value

        for worker in workers:
            worker.join()

        v_vec_soma = np.array(v_vec_soma)
        v_vec_elec = np.array(v_vec_elec)

        data['dt'] = cell.timeres_NEURON
        data['v_vec_soma'] = v_vec_soma
        data['v_vec_elec'] = v_vec_elec*1000
        data['t_vec'] = np.arange(v_vec_soma.shape[1])*cell.timeres_NEURON
        data['amps'] = amps
        data['elec_x'] = x
        data['elec_y'] = y
        data['elec_z'] = z

    def process_data(self):
        data = self.data
        run_param = self.run_param
        process_param = self.process_param

        if not data:
            raise ValueError('No data to process.')

        freqs = np.zeros(run_param['sweeps'])
        isi = np.zeros(run_param['sweeps'])
        freqs_elec = np.zeros(run_param['sweeps'])
        isi_elec = np.zeros(run_param['sweeps'])
        
        # Find firing frequency and inter spike interval from soma.
        for i in xrange(run_param['sweeps']):
            spike_indices = de.find_spikes(
                data['t_vec'],
                data['v_vec_soma'][i],
                process_param['threshold'],
                pre_dur=process_param['pre_dur'],
                post_dur=process_param['post_dur'],
                threshold_abs=process_param['threshold_abs_soma']
                )
            isi_tmp = np.diff(spike_indices) * data['dt']
            if isi_tmp.size != 0:
                isi[i] = np.mean(isi_tmp)
                freqs[i] = 1000/isi[i] # Hz
            else:
                isi[i] = 0
                freqs[i] = 0

        spikes_soma = [None]*run_param['sweeps']
        widths_I_soma = [None]*run_param['sweeps']
        widths_I_soma_mean = np.zeros(run_param['sweeps'])
        widths_I_soma_std = np.zeros(run_param['sweeps'])
        widths_II_soma = [None]*run_param['sweeps']
        widths_II_soma_mean = np.zeros(run_param['sweeps'])
        widths_II_soma_std = np.zeros(run_param['sweeps'])

        amps_I_soma = [None]*run_param['sweeps']
        amps_I_soma_mean = np.zeros(run_param['sweeps'])
        amps_I_soma_std = np.zeros(run_param['sweeps'])

        # Soma spikes and widths.
        for i in xrange(run_param['sweeps']):
            with warnings.catch_warnings():
                # Ignore warnings when not finding spikes.
                warnings.simplefilter("ignore", category=RuntimeWarning)

                spikes_soma[i], spikes_t_vec, I = de.extract_spikes(
                    data['t_vec'],
                    data['v_vec_soma'][i],
                    pre_dur=process_param['pre_dur'],
                    post_dur=process_param['post_dur'],
                    threshold=process_param['threshold'],
                    amp_option=process_param['amp_option'], 
                    threshold_abs=process_param['threshold_abs_soma']
                    )
            if spikes_soma[i].shape[0] == 0:
                widths_I_soma[i] = []
                widths_I_soma_mean[i] = np.nan
                widths_II_soma[i] = []
                widths_II_soma_mean[i] = np.nan
                amps_I_soma_mean[i] = np.nan
                amps_I_soma_std[i] = np.nan
                continue

            widths_I_soma[i], trace = de.find_wave_width_type_I(
                spikes_soma[i],
                dt=data['dt'],
                )

            widths_II_soma[i], trace = de.find_wave_width_type_II(
                spikes_soma[i],
                dt=data['dt'],
                amp_option=process_param['amp_option'],
                )

            widths_I_soma_mean[i] = np.mean(widths_I_soma[i])
            widths_II_soma_mean[i] = np.mean(widths_II_soma[i])
            widths_I_soma_std[i] = np.sqrt(np.var(widths_I_soma[i]))
            widths_II_soma_std[i] = np.sqrt(np.var(widths_II_soma[i]))

            amps_I_soma[i] = de.find_amplitude_type_I(
                    spikes_soma[i],
                    process_param['amp_option'])

            amps_I_soma_mean[i] = np.mean(amps_I_soma[i])
            amps_I_soma_std[i] = np.sqrt(np.var(amps_I_soma[i]))


        # There are a variable amount of spikes, so these varibles have
        # one dimension with unknown lenght.
        spikes_elec = [None]*run_param['sweeps']
        widths_I_elec = [None]*run_param['sweeps']
        widths_II_elec = [None]*run_param['sweeps']
        amps_I_elec = [None]*run_param['sweeps']
        for i in xrange(run_param['sweeps']):
            spikes_elec[i] = [None]*run_param['n_elec']
            widths_I_elec[i] = [None]*run_param['n_elec']
            widths_II_elec[i] = [None]*run_param['n_elec']
            amps_I_elec[i] = [None]*run_param['n_elec']

        widths_I_elec_mean  = np.zeros([run_param['sweeps'], run_param['n_elec']])
        widths_I_elec_std  = np.zeros([run_param['sweeps'], run_param['n_elec']])

        widths_II_elec_mean = np.zeros([run_param['sweeps'], run_param['n_elec']])
        widths_II_elec_std = np.zeros([run_param['sweeps'], run_param['n_elec']])
        
        amps_I_elec_mean = np.zeros([run_param['sweeps'], run_param['n_elec']])
        amps_I_elec_std = np.zeros([run_param['sweeps'], run_param['n_elec']])

        # Electrode spikes and widths.
        for i in xrange(run_param['sweeps']):
            for j in xrange(run_param['n_elec']):

                with warnings.catch_warnings():
                    # Ignore warnings when not finding spikes.
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    spikes_elec[i][j], spikes_t_vec, I = de.extract_spikes(
                        data['t_vec'],
                        data['v_vec_elec'][i, j],
                        pre_dur=process_param['pre_dur'],
                        post_dur=process_param['post_dur'],
                        threshold=process_param['threshold'],
                        amp_option=process_param['amp_option'], 
                        threshold_abs=process_param['threshold_abs_elec']
                        )

                if spikes_elec[i][j].shape[0] == 0:
                    widths_I_elec[i][j] = []
                    widths_I_elec_mean[i, j] = np.nan
                    widths_II_elec[i][j] = []
                    widths_II_elec_mean[i, j] = np.nan
                    amps_I_elec_mean[i, j] = np.nan
                    amps_I_elec_std[i, j] = np.nan
                    continue

                widths_I_elec[i][j], trace = de.find_wave_width_type_I(
                    spikes_elec[i][j],
                    dt=data['dt'],
                    )

                widths_II_elec[i][j], trace = de.find_wave_width_type_II(
                    spikes_elec[i][j],
                    dt=data['dt'],
                    amp_option=process_param['amp_option'],
                    )

                widths_I_elec_mean[i, j] = np.mean(widths_I_elec[i][j])
                widths_II_elec_mean[i, j] = np.mean(widths_II_elec[i][j])
                widths_I_elec_std[i, j] = np.sqrt(np.var(widths_I_elec[i][j]))
                widths_II_elec_std[i, j] = np.sqrt(np.var(widths_II_elec[i][j]))

                amps_I_elec[i] = de.find_amplitude_type_I(
                        spikes_elec[i][j],
                        process_param['amp_option'])

                amps_I_elec_mean[i, j] = np.mean(amps_I_elec[i][j])
                amps_I_elec_std[i, j] = np.sqrt(np.var(amps_I_elec[i][j]))

        data['spikes_t_vec'] = spikes_t_vec
        data['spikes_soma'] = spikes_soma
        data['spikes_elec'] = spikes_elec

        data['freqs'] = freqs
        data['isi'] = isi

        data['widths_I_soma'] = widths_I_soma
        data['widths_I_soma_mean'] = widths_I_soma_mean
        data['widths_I_soma_std'] = widths_I_soma_std

        data['widths_II_soma'] = widths_II_soma
        data['widths_II_soma_mean'] = widths_II_soma_mean
        data['widths_II_soma_std'] = widths_II_soma_std

        data['amps_I_soma'] = amps_I_soma
        data['amps_I_soma_mean'] = amps_I_soma_mean
        data['amps_I_soma_std'] = amps_I_soma_std

        data['widths_I_elec'] = widths_I_elec
        data['widths_I_elec_mean'] = widths_I_elec_mean
        data['widths_I_elec_std'] = widths_I_elec_std

        data['widths_II_elec'] = widths_II_elec
        data['widths_II_elec_mean'] = widths_II_elec_mean
        data['widths_II_elec_std'] = widths_II_elec_std

        data['amps_I_elec'] = amps_I_elec
        data['amps_I_elec_mean'] = amps_I_elec_mean
        data['amps_I_elec_std'] = amps_I_elec_std

        self.info['elec_positions'] = zip(data['elec_x'],
                                          data['elec_y'],
                                          data['elec_z'])

    def plot(self, dir_plot):
        data = self.data
        run_param = self.run_param

        LFPy_util.plot.set_rc_param(self.plot_param['use_tex'])
        LFPy_util.plot.plot_format = self.plot_param['formats']

        # {{{ Plot elec spikes overlay
        # Create the directory if it does not exist.
        for elec in xrange(run_param['n_elec']):
            sub_dir = os.path.join(dir_plot, 'elec_{}_all_spikes'.format(elec))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            for i in xrange(run_param['sweeps']):
                fname = self.name + '_elec_all_spikes_sweep_{}'.format(i)
                print "plotting            :", fname
                plt.figure(figsize=lplot.size_common)
                ax = plt.gca()
                lplot.nice_axes(ax)
                for j in xrange(data['spikes_elec'][i][elec].shape[0]):
                    plt.plot(data['spikes_t_vec'],
                             data['spikes_elec'][i][elec][j],
                             color=lcmaps.get_color(0),
                             alpha=0.2,
                             )
                # ax.set_xlabel(r"Stimulus Current \textbf{[\si{\nano\ampere}]}")
                # Save plt.
                lplot.save_plt(plt, fname, sub_dir)
                plt.close()
        # }}} 
        # {{{ Plot soma spikes overlay
        # Create the directory if it does not exist.
        sub_dir = os.path.join(dir_plot, 'soma_all_spikes')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        for i in xrange(run_param['sweeps']):
            fname = self.name + '_soma_all_spikes_sweep_{}'.format(i)
            print "plotting            :", fname
            plt.figure(figsize=lplot.size_common)
            ax = plt.gca()
            lplot.nice_axes(ax)
            for j in xrange(data['spikes_soma'][i].shape[0]):
                plt.plot(data['spikes_t_vec'],
                         data['spikes_soma'][i][j],
                         color=lcmaps.get_color(0),
                         alpha=0.2,
                         )
            # ax.set_xlabel(r"Stimulus Current \textbf{[\si{\nano\ampere}]}")
            # Save plt.
            lplot.save_plt(plt, fname, sub_dir)
            plt.close()
        # }}} 
        # Plotting all sweeps {{{ 
        if run_param['sweeps'] <= 10:
            fname = self.name + '_soma_mem'
            print "plotting            :", fname
            plt.figure(figsize=lplot.size_common)
            ax = plt.gca()
            lplot.nice_axes(ax)
            colors = lcmaps.get_short_color_array(run_param['sweeps']+1)
            for i in xrange(run_param['sweeps']):
                plt.plot(data['t_vec'],
                         data['v_vec_soma'][i],
                         color=colors[i],
                         )
            ax.set_xlabel(r"Time \textbf{[\si{\milli\second}]}")
            # Save plt.
            lplot.save_plt(plt, fname, dir_plot)
            plt.close()
        # }}} 
        # Plotting all sweeps elec {{{ 
        if run_param['sweeps'] <= 10:
            for i in xrange(run_param['n_elec']):
                fname = self.name + '_elec_{}'.format(i)
                print "plotting            :", fname
                plt.figure(figsize=lplot.size_common)
                ax = plt.gca()
                lplot.nice_axes(ax)
                colors = lcmaps.get_short_color_array(run_param['sweeps']+1)
                for j in xrange(run_param['sweeps']):
                    plt.plot(data['t_vec'],
                             data['v_vec_elec'][j, i],
                             color=colors[j],
                             )
                ax.set_xlabel(r"Time \textbf{[\si{\milli\second}]}")
                # Save plt.
                lplot.save_plt(plt, fname, dir_plot)
                plt.close()
        # }}} 
        # {{{ Plot f_i
        fname = self.name + '_f_i'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        plt.plot(data['amps']*1000,
                 data['freqs'],
                 color=lcmaps.get_color(0),
                 marker='o',
                 markersize=5,
                 )
        ax.set_xlabel(r"Stimulus Current \textbf{[\si{\nano\ampere}]}")
        # Save plt.
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # }}} 
        # {{{ Plot spike width over frequency I soma
        fname = self.name + '_soma_width_frequency_I'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        plt.plot(data['freqs'],
                 data['widths_I_soma_mean'],
                 color=lcmaps.get_color(0),
                 marker='o',
                 markersize=5,
                 )
        ax.fill_between(data['freqs'],
                        data['widths_I_soma_mean'] - data['widths_I_soma_std'],
                        data['widths_I_soma_mean'] + data['widths_I_soma_std'],
                        color=lcmaps.get_color(0),
                        alpha=0.2)
        ax.set_ylabel(r"Peak-to-Peak Width \textbf{[\si{\milli\second}]}")
        ax.set_xlabel(r"Frequency \textbf{[\si{\hertz}]}")
        # ax.set_xlabel(r"Stimulus Current \textbf{[\si{\nano\ampere}]}")
        # Save plt.
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # }}} 
        # {{{ Plot spike width over frequency II soma
        fname = self.name + '_soma_width_frequency_II'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        plt.plot(data['freqs'],
                 data['widths_II_soma_mean'],
                 color=lcmaps.get_color(0),
                 marker='o',
                 markersize=5,
                 )
        ax.fill_between(data['freqs'],
                        data['widths_II_soma_mean'] - data['widths_II_soma_std'],
                        data['widths_II_soma_mean'] + data['widths_II_soma_std'],
                        color=lcmaps.get_color(0),
                        alpha=0.2)
        ax.set_ylabel(r"Half Max Width \textbf{[\si{\milli\second}]}")
        ax.set_xlabel(r"Frequency \textbf{[\si{\hertz}]}")
        # ax.set_xlabel(r"Stimulus Current \textbf{[\si{\nano\ampere}]}")
        # Save plt.
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # }}} 
        # {{{ Plot spike width over frequency I elec
        for i in xrange(run_param['n_elec']):
            fname = self.name + '_elec_{}_width_frequency_I'.format(i)
            print "plotting            :", fname
            plt.figure(figsize=lplot.size_common)
            ax = plt.gca()
            lplot.nice_axes(ax)
            plt.plot(data['freqs'],
                     data['widths_I_elec_mean'][:, i],
                     color=lcmaps.get_color(0),
                     marker='o',
                     markersize=5,
                     )
            ax.fill_between(
                    data['freqs'],
                    data['widths_I_elec_mean'][:,i] 
                        - data['widths_I_elec_std'][:,i],
                    data['widths_I_elec_mean'][:,i] 
                        + data['widths_I_elec_std'][:,i],
                    color=lcmaps.get_color(0),
                    alpha=0.2)
            ax.set_ylabel(r"Peak-to-Peak Width \textbf{[\si{\milli\second}]}")
            ax.set_xlabel(r"Frequency \textbf{[\si{\hertz}]}")
            # Save plt.
            lplot.save_plt(plt, fname, dir_plot)
            plt.close()
        # }}} 
        # {{{ Plot spike width over frequency II elec
        for i in xrange(run_param['n_elec']):
            fname = self.name + '_elec_{}_width_frequency_II'.format(i)
            print "plotting            :", fname
            plt.figure(figsize=lplot.size_common)
            ax = plt.gca()
            lplot.nice_axes(ax)
            plt.plot(data['freqs'],
                     data['widths_II_elec_mean'][:,i],
                     color=lcmaps.get_color(0),
                     marker='o',
                     markersize=5,
                     )
            ax.fill_between(
                    data['freqs'],
                    data['widths_II_elec_mean'][:,i] 
                        - data['widths_II_elec_std'][:,i],
                    data['widths_II_elec_mean'][:,i] 
                        + data['widths_II_elec_std'][:,i],
                    color=lcmaps.get_color(0),
                    alpha=0.2)
            ax.set_ylabel(r"Half Max Width \textbf{[\si{\milli\second}]}")
            ax.set_xlabel(r"Frequency \textbf{[\si{\hertz}]}")
            # Save plt.
            lplot.save_plt(plt, fname, dir_plot)
            plt.close()
        # }}} 
        # {{{ Plot spike width over frequency I elec soma
        for i in xrange(run_param['n_elec']):
            fname = self.name + '_elec_{}_soma_width_frequency_I'.format(i)
            print "plotting            :", fname
            plt.figure(figsize=lplot.size_common)
            ax = plt.gca()
            lplot.nice_axes(ax)
            plt.plot(data['freqs'],
                     data['widths_I_soma_mean'],
                     color=lcmaps.get_color(0),
                     marker='o',
                     markersize=5,
                     label='Soma',
                     )
            ax.fill_between(data['freqs'],
                            data['widths_I_soma_mean'] - data['widths_I_soma_std'],
                            data['widths_I_soma_mean'] + data['widths_I_soma_std'],
                            color=lcmaps.get_color(0),
                            alpha=0.2)
            plt.plot(data['freqs'],
                     data['widths_I_elec_mean'][:, i],
                     color=lcmaps.get_color(0.5),
                     marker='o',
                     markersize=5,
                     label='Elec.'
                     )
            ax.fill_between(
                    data['freqs'],
                    data['widths_I_elec_mean'][:,i] 
                        - data['widths_I_elec_std'][:,i],
                    data['widths_I_elec_mean'][:,i] 
                        + data['widths_I_elec_std'][:,i],
                    color=lcmaps.get_color(0.5),
                    alpha=0.2)
            ax.set_ylabel(r"Peak-to-Peak Width \textbf{[\si{\milli\second}]}")
            ax.set_xlabel(r"Frequency \textbf{[\si{\hertz}]}")
            # handles, labels = ax.get_legend_handles_labels()
            # ax.legend(handles,
            #           labels,
            #           loc='upper right',
            #           # bbox_to_anchor=(1, 0.5), 
            #           )
            # Save plt.
            lplot.save_plt(plt, fname, dir_plot)
            plt.close()
        # }}} 
        # {{{ Plot spike width over frequency II elec soma
        for i in xrange(run_param['n_elec']):
            fname = self.name + '_elec_{}_soma_width_frequency_II'.format(i)
            print "plotting            :", fname
            plt.figure(figsize=lplot.size_common)
            ax = plt.gca()
            lplot.nice_axes(ax)
            plt.plot(data['freqs'],
                     data['widths_II_soma_mean'],
                     color=lcmaps.get_color(0),
                     marker='o',
                     markersize=5,
                     label='Soma',
                     )
            ax.fill_between(data['freqs'],
                            data['widths_II_soma_mean'] - data['widths_II_soma_std'],
                            data['widths_II_soma_mean'] + data['widths_II_soma_std'],
                            color=lcmaps.get_color(0),
                            alpha=0.2)
            plt.plot(data['freqs'],
                     data['widths_II_elec_mean'][:,i],
                     color=lcmaps.get_color(0.5),
                     marker='o',
                     markersize=5,
                     label='Elec.',
                     )
            ax.fill_between(
                    data['freqs'],
                    data['widths_II_elec_mean'][:,i] 
                        - data['widths_II_elec_std'][:,i],
                    data['widths_II_elec_mean'][:,i] 
                        + data['widths_II_elec_std'][:,i],
                    color=lcmaps.get_color(0.5),
                    alpha=0.2)
            ax.set_ylabel(r"Half Max Width \textbf{[\si{\milli\second}]}")
            ax.set_xlabel(r"Frequency \textbf{[\si{\hertz}]}")
            # Save plt.
            lplot.save_plt(plt, fname, dir_plot)
            plt.close()
        # }}} 
        # {{{ Plot spike amplitude over frequency I soma
        fname = self.name + '_soma_amp_frequency_I'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        plt.plot(data['freqs'],
                 data['amps_I_soma_mean'],
                 color=lcmaps.get_color(0),
                 marker='o',
                 markersize=5,
                 )
        ax.fill_between(data['freqs'],
                        data['amps_I_soma_mean'] - data['amps_I_soma_std'],
                        data['amps_I_soma_mean'] + data['amps_I_soma_std'],
                        color=lcmaps.get_color(0),
                        alpha=0.2)
        ax.set_ylabel(r"Base-to-Peak Amp.")
        ax.set_xlabel(r"Frequency \textbf{[\si{\hertz}]}")
        # Save plt.
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()
        # }}} 
        # {{{ Plot spike amplitude over frequency I elec
        for i in xrange(run_param['n_elec']):
            fname = self.name + '_elec_{}_amp_frequency_I'.format(i)
            print "plotting            :", fname
            plt.figure(figsize=lplot.size_common)
            ax = plt.gca()
            lplot.nice_axes(ax)
            plt.plot(data['freqs'],
                     data['amps_I_elec_mean'][:, i],
                     color=lcmaps.get_color(0),
                     marker='o',
                     markersize=5,
                     )
            ax.fill_between(
                    data['freqs'],
                    data['amps_I_elec_mean'][:,i] 
                        - data['amps_I_elec_std'][:,i],
                    data['amps_I_elec_mean'][:,i] 
                        + data['amps_I_elec_std'][:,i],
                    color=lcmaps.get_color(0),
                    alpha=0.2)
            ax.set_ylabel(r"Base-to-Peak Amp.")
            ax.set_xlabel(r"Frequency \textbf{[\si{\hertz}]}")
            # Save plt.
            lplot.save_plt(plt, fname, dir_plot)
            plt.close()
        # }}} 
        # {{{ Plot spike amplitude over frequency I elec soma
        for i in xrange(run_param['n_elec']):
            fname = self.name + '_elec_{}_soma_amp_frequency_I'.format(i)
            print "plotting            :", fname
            plt.figure(figsize=lplot.size_common)
            ax = plt.gca()
            lplot.nice_axes(ax)
            plt.plot(data['freqs'],
                     data['amps_I_soma_mean'],
                     color=lcmaps.get_color(0),
                     marker='o',
                     markersize=5,
                     label='Soma',
                     )
            ax.fill_between(data['freqs'],
                            data['amps_I_soma_mean'] - data['amps_I_soma_std'],
                            data['amps_I_soma_mean'] + data['amps_I_soma_std'],
                            color=lcmaps.get_color(0),
                            alpha=0.2)
            plt.plot(data['freqs'],
                     data['amps_I_elec_mean'][:, i],
                     color=lcmaps.get_color(0.5),
                     marker='o',
                     markersize=5,
                     label='Elec.'
                     )
            ax.fill_between(
                    data['freqs'],
                    data['amps_I_elec_mean'][:,i] 
                        - data['amps_I_elec_std'][:,i],
                    data['amps_I_elec_mean'][:,i] 
                        + data['amps_I_elec_std'][:,i],
                    color=lcmaps.get_color(0.5),
                    alpha=0.2)
            ax.set_ylabel(r"Base-to-Peak Amp.")
            ax.set_xlabel(r"Frequency \textbf{[\si{\hertz}]}")
            # Save plt.
            lplot.save_plt(plt, fname, dir_plot)
            plt.close()
        # }}} 


class Worker(Process):
    def __init__(self, input_queue, output_queue, cell, run_param, 
            elec_x, elec_y, elec_z):
        super(Worker, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.cell = cell
        self.run_param = run_param
        self.elec_x = elec_x
        self.elec_y = elec_y
        self.elec_z = elec_z

    def run(self):
        run_param = self.run_param
        cell = self.cell

        duration = run_param['duration']
        delay = run_param['delay']
        pptype = run_param['pptype']

        for sec in cell.allseclist:
            if 'soma' in sec.name():
                if pptype == 'ISyn':
                    syn = neuron.h.ISyn(0.5, sec=sec)
                elif pptype == 'IClamp':
                    syn = neuron.h.IClamp(0.5, sec=sec)
                else:
                    syn = None
                break

        # Will loop until None is in the input_queue.
        for data in iter(self.input_queue.get, None):
            i, amp = data
            if pptype == 'ISyn':
                syn.dur = duration
                syn.delay = delay
                syn.amp = amp
            elif pptype == 'IClamp':
                syn.dur = duration
                syn.delay = delay
                syn.amp = amp
            # A fix for when the stim should be applied before
            # the recording from the simulation.
            if cell.tstartms < 0:
                syn.delay = syn.delay + cell.tstartms

            cell.simulate(rec_imem=True)

            electrode = LFPy.RecExtElectrode(cell,
                                             x=self.elec_x,
                                             y=self.elec_y,
                                             z=self.elec_z,
                                             sigma=run_param['sigma'])
            electrode.method = run_param['ext_method']
            electrode.calc_lfp()

            data_packet = (i, 'somav', cell.somav)
            self.output_queue.put(data_packet)

            data_packet = (i, 'lfp', electrode.LFP)
            self.output_queue.put(data_packet)

            self.output_queue.put(None)

