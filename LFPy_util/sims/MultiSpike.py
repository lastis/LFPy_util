"""
Simulation that calculates the current needed to give one spike over
the simulation time. Also applies that electrode when simulation is finised.
"""
import os
from multiprocessing import Process, Manager
import numpy as np
import matplotlib.pyplot as plt
import LFPy
import LFPy_util
from LFPy_util.sims import Simulation
import LFPy_util.plot as lplot
import LFPy_util.colormaps as cmaps


class MultiSpike(Simulation):
    """docstring for Grid"""

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.name = 'mspike'

        # Used by the custom simulate and plot function.
        self.run_param['threshold'] = 4
        self.run_param['pptype'] = 'IClamp'
        self.run_param['delay'] = 100
        self.run_param['duration'] = 300
        self.run_param['init_amp'] = 1
        self.run_param['pre_dur'] = 16.7 * 0.5
        self.run_param['post_dur'] = 16.7 * 0.5
        self.run_param['spikes'] = 3
        self.apply_electrode_at_finish = True
        self.verbose = False
        self._prev_data = None

    def previous_run(self, dir_data):
        """
        If this simulation has been ran before, try setting init_amp
        to the amp of that file to avoid doing uneccessary simulations.
        """
        fname = self.get_fname_data()
        self._prev_data = os.path.join(dir_data, fname)

    def get_previous_amp(self):
        """
        Loads data from previous simulation and finds which amp was used.
        """
        # String to put before some output to the terminal.
        str_start = self.name
        str_start += " "*(20 - len(self.name)) + ":"
        # If this simulation has been ran before, try setting init_amp
        # to the amp of that file to avoid doing uneccessary simulations.
        path = self._prev_data
        if path is None or not os.path.isfile(path):
            if self.verbose:
                print str_start + " Could not load previous amp."
            return self.run_param['init_amp']
        if self.format_save_data == 'pkl':
            data_tmp = LFPy_util.other.load_kwargs(path)
        elif self.format_save_data == 'js':
            data_tmp = LFPy_util.other.load_kwargs_json(path)
        else:
            raise ValueError("Unsupported format")
        if self.verbose:
            print str_start + " Previous amp = " + str(data_tmp['amp'])
        return data_tmp['amp']

    def simulate(self, cell):
        run_param = self.run_param
        amp = self.get_previous_amp()

        # String to put before some output to the terminal.
        str_start = self.name
        str_start += " "*(20 - len(self.name)) + ":"

        # Find a current that generates n spikes.
        amp_low = 0
        amp_high = 0
        # Copy the run param so they can be given to the "sub" simulation.
        sub_run_param = run_param.copy()
        while True:
            # Gather data from the sub process in a dictionary.
            # manager = Manager()
            sub_data = Manager().dict()
            # Set the new amp.
            sub_run_param['amp'] = amp
            # Run the "sub" simulation.
            target = self.simulate_sub
            args = (cell, sub_data, sub_run_param)
            process = Process(target=target, args=args)
            process.start()
            process.join()

            # Change the amplitude according to the spike cnt.
            spike_cnt = sub_data.get('spike_cnt', 0)
            if self.verbose:
                print str_start + ' Found {} spikes at current {} nA.'.format(spike_cnt,
                                                                 amp)
            if spike_cnt == run_param['spikes']:
                break
            elif spike_cnt < run_param['spikes']:
                amp_low = amp
            elif spike_cnt > run_param['spikes']:
                amp_high = amp
            # Increase the amp until we have more than the desired number of spikes.
            if amp_high == 0:
                amp = 1.25 * amp
                continue
            amp = 0.5 * (amp_high + amp_low)
            if amp < 1e-4 or amp > 1e4:
                print str_start + ' Curent amplitude is above or under threshold, finishing.'
                return

        # Give the data back.
        self.data['amp'] = amp
        self.data['spike_cnt'] = spike_cnt
        self.data['dt'] = cell.timeres_NEURON
        self.data['stimulus_i'] = sub_data['stimulus_i']
        self.data['soma_v'] = sub_data['soma_v']
        self.data['soma_t'] = sub_data['soma_t']

        if self.apply_electrode_at_finish:
            soma_clamp_params = {
                'idx': cell.somaidx,
                'record_current': True,
                'amp': self.data['amp'],  #  [nA]
                'dur': self.run_param['duration'],  # [ms]
                'delay': self.run_param['delay'],  # [ms]
                'pptype': self.run_param['pptype'],
            }
            stim = LFPy.StimIntElectrode(cell, **soma_clamp_params)

    @staticmethod
    def simulate_sub(cell, data, run_param):
        """
        This is run in a new process by the simulation function, and applies 
        a spesific electrode amp.
        """
        amp = run_param['amp']
        duration = run_param['duration']
        delay = run_param['delay']
        threshold = run_param['threshold']
        pptype = run_param['pptype']

        soma_clamp_params = {
            'idx': cell.somaidx,
            'record_current': True,
            'amp': amp,  #  [nA]
            'dur': duration,  # [ms]
            'delay': delay,  # [ms]
            'pptype': pptype,
        }

        stim = LFPy.StimIntElectrode(cell, **soma_clamp_params)
        cell.simulate(rec_vmem=True,
                      rec_imem=True,
                      rec_istim=True,
                      rec_isyn=True)
        # Find spikes.
        max_idx = LFPy_util.data_extraction.find_spikes(cell.tvec,
                                                        cell.somav,
                                                        threshold,
                                                        run_param['pre_dur'],
                                                        run_param['post_dur'],
                                                       )
        # Count local maxima over threshold as spikes.
        spike_cnt = len(max_idx)
        data['stimulus_i'] = stim.i
        data['spike_cnt'] = spike_cnt
        data['soma_v'] = cell.somav
        data['soma_t'] = cell.tvec

    def process_data(self):
        data = self.data
        run_param = self.run_param

        data['soma_v'] = np.array(data['soma_v'])
        data['soma_t'] = np.array(data['soma_t'])
        data['stimulus_i'] = np.array(data['stimulus_i'])

        # Extract the shape around the first spike.
        spikes, t_vec_spike, _ = LFPy_util.data_extraction.extract_spikes(
            data['soma_t'],
            data['soma_v'],
            pre_dur=run_param['pre_dur'],
            post_dur=run_param['post_dur'],
            threshold=run_param['threshold'])
        data['spikes'] = spikes
        data['t_vec_spike'] = t_vec_spike

    def plot(self, dir_plot):
        data = self.data
        run_param = self.run_param

        # String to put before output to the terminal.
        str_start = self.name
        str_start += " "*(20 - len(self.name)) + ":"

        # Set global matplotlib parameters.
        LFPy_util.plot.set_rc_param()

        # New plot.
        fname = self.name + '_all_spikes'
        if run_param['pre_dur'] != 0 and run_param['post_dur'] != 0:
            print "plotting            :", fname
            plt.figure(figsize=lplot.size_common)
            ax = plt.gca()
            ax.set_ylabel(r'Membrane Potential \textbf[$\mathbf{mV}$\textbf]')
            ax.set_xlabel(r'Time \textbf[$\mathbf{ms}$\textbf]')
            lplot.nice_axes(ax)
            rows = data['spikes'].shape[0]
            colors = cmaps.get_short_color_array(rows)
            for row in xrange(rows):
                plt.plot(
                    data['t_vec_spike'], 
                    data['spikes'][row], 
                    color=colors[row],
                    label="Spike {}".format(row)
                    )
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels)
            lplot.save_plt(plt, fname, dir_plot)
        else:
            if self.verbose:
                print str_start + " Missing pre and post dur, not plotting " + fname

        # New plot.
        fname = self.name + '_soma_mem'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        ax.set_ylabel(r'Membrane Potential \textbf[$\mathbf{mV}$\textbf]')
        ax.set_xlabel(r'Time \textbf[$\mathbf{ms}$\textbf]')
        lplot.nice_axes(ax)
        plt.plot(data['soma_t'], data['soma_v'], color=cmaps.get_color(0))
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()

        # New plot.
        fname = self.name + '_soma_v_mem_i_mem'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)

        ax = plt.subplot(2, 1, 1)
        lplot.nice_axes(ax)
        plt.plot(data['soma_t'], data['soma_v'], color=cmaps.get_color(0))
        ax.set_ylabel(r'Membrane Potential \textbf[$\mathbf{mV}$\textbf]')

        ax = plt.subplot(2, 1, 2)
        lplot.nice_axes(ax)
        plt.plot(data['soma_t'], data['stimulus_i'], color=cmaps.get_color(0))

        ax.set_ylabel(r'Stimulus Current \textbf[$\mathbf{nA}$\textbf]')
        ax.set_xlabel(r'Time \textbf[$\mathbf{ms}$\textbf]')
        lplot.save_plt(plt, fname, dir_plot)
        plt.close()

