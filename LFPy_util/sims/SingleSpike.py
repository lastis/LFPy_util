from Simulation import Simulation
import LFPy
import LFPy_util
import numpy as np
import os
from multiprocessing import Process, Manager

fname_current_plot_soma_spike       = 'current_soma_spike'
fname_current_plot_soma_mem         = 'current_soma_mem'
fname_current_plot_soma_i_mem_v_mem = 'current_soma_i_mem_v_mem'
fname_current_run_info              = 'current_run_info.md'

class SingleSpike(Simulation):
    """docstring for Grid"""

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.ID = 'current'
        
        # Used by the custom simulate and plot function.
        self.run_param['threshold'] = 3
        self.run_param['pptype'] = 'IClamp'
        self.run_param['delay'] = 50
        self.run_param['duration'] = 200
        self.run_param['init_amp'] = 1
        self.run_param['pre_dur'] = 16.7*0.5
        self.run_param['post_dur'] = 16.7*0.5
        self.apply_electrode_at_finish = True
        self.debug = False
        self.prev_data = None

        # Used by the custom plot function.
        self.show = False

    def __str__(self):
        return "SingleSpike"

    def get_previous_amp(self):
        # If this simulation has been ran before, try setting init_amp
        # to the amp of that file to avoid doing uneccessary simulations.
        path = self.prev_data
        if path is None or not os.path.isfile(path):
            if self.debug:
                print "Could not load previous one spike amp."
            return self.run_param['init_amp']
        if self.format_save_data == 'pkl':
            data_tmp = LFPy_util.other.load_kwargs(path)
        elif self.format_save_data == 'js':
            data_tmp = LFPy_util.other.load_kwargs_json(path)
        else:
            raise ValueError("Unsupported format")
        if self.debug:
            print "Previous one spike amp: " + str(data_tmp['amp'])
        return data_tmp['amp']

    def simulate(self,cell):
        run_param = self.run_param
        run_param['init_amp'] = self.get_previous_amp();
        # Find a current that generates one spike.
        amp = run_param['init_amp']
        spike_cnt_low  = 0
        amp_low        = 0
        amp_high       = 0
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
            args = (cell,sub_data,sub_run_param)
            process = Process(target=target,args=args)
            process.start()
            process.join()

            # Change the amplitude according to the spike cnt. 
            spike_cnt = sub_data['spike_cnt']
            if self.debug:
                print 'Found {} spikes at current {} nA.'.format(spike_cnt,amp)
            if spike_cnt == 1:
                break
            elif spike_cnt < 1:
                spike_cnt_low = spike_cnt
                amp_low = amp
            elif spike_cnt > 1:
                spike_cnt_high = spike_cnt
                amp_high = amp
            # Double the amp until we have more than one spike. 
            if amp_high == 0:
                amp = 2*amp
                continue
            amp = 0.5*(amp_high+amp_low)
            if amp < 1e-9 or amp > 1e9:
                print 'Curent amplitude is above or under threshold, finishing.'
                return

        # Give the data back.
        self.data['amp']          = amp
        self.data['spike_cnt']    = spike_cnt
        self.data['dt']           = cell.timeres_NEURON
        self.data['poly_morph']   = cell.get_idx_polygons(('x','y'))
        self.data['stimulus_i']   = sub_data['stimulus_i']
        self.data['spike_cnt']    = sub_data['spike_cnt']
        self.data['soma_v']       = sub_data['soma_v']
        self.data['soma_t']       = sub_data['soma_t']

        self.process_data()

        if self.apply_electrode_at_finish:
            soma_clamp_params = {
                'idx': cell.somaidx,
                'record_current': True,
                'amp': self.data['amp'], #  [nA]
                'dur': self.run_param['duration'],  # [ms]
                'delay': self.run_param['delay'],  # [ms]
                'pptype': self.run_param['pptype'],
            }
            stim = LFPy.StimIntElectrode(cell, **soma_clamp_params)


    def simulate_sub(self, cell, data, run_param):
        amp         = run_param.get('amp')
        duration    = run_param.get('duration')
        delay       = run_param.get('delay')
        threshold   = run_param.get('threshold')
        pptype      = run_param.get('pptype','IClamp')
        
        soma_clamp_params = {
            'idx': cell.somaidx,
            'record_current': True,
            'amp': amp, #  [nA]
            'dur': duration,  # [ms]
            'delay': delay,  # [ms]
            'pptype': pptype,
        }

        stim = LFPy.StimIntElectrode(cell, **soma_clamp_params)
        cell.simulate(rec_vmem=True,rec_imem=True,rec_istim=True,rec_isyn=True)
        # Find local maxima.
        max_idx = LFPy_util.data_extraction.find_spikes(
                cell.somav, threshold=threshold)
        # Count local maxima over threshold as spikes.
        spike_cnt = len(max_idx)
        data['stimulus_i']   = stim.i
        data['spike_cnt']    = spike_cnt
        data['soma_v']       = cell.somav
        data['soma_t']       = cell.tvec


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
                threshold=run_param['threshold']
        )
        if len(spikes) == 0:
            print "Could not find any spikes at threshold = {}.".format(run_param['threshold'])
            spikes, t_vec_spike, _ = LFPy_util.data_extraction.extract_spikes(
                    data['soma_t'],
                    data['soma_v'],
                    pre_dur=run_param['pre_dur'],
                    post_dur=run_param['post_dur'],
                    threshold=run_param['threshold']
            )
        # Hope everything is correct and use the first (and only) spike.
        v_vec_spike = spikes[0]
        data['spikes'] = spikes
        data['v_vec_spike'] = v_vec_spike
        data['t_vec_spike'] = t_vec_spike

    def plot(self,dir_plot):
        data = self.data
        run_param = self.run_param

        # Plot data.
        LFPy_util.plot.soma(
                data['t_vec_spike'],
                data['v_vec_spike'],
                fname_current_plot_soma_spike, 
                plot_save_dir=dir_plot,
                show = self.show,
        )
        LFPy_util.plot.soma(
                data['soma_t'],
                data['soma_v'],
                fname_current_plot_soma_mem, 
                plot_save_dir=dir_plot,
                show = self.show,
        )

        LFPy_util.plot.i_mem_v_mem(
                data['soma_v'],
                data['stimulus_i'],
                data['soma_t'],
                fname_current_plot_soma_i_mem_v_mem,
                plot_save_dir=dir_plot,
                show = self.show,
        )
