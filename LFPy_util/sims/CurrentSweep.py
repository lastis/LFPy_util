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
import LFPy_util.colormaps as cmaps

from multiprocessing import Process, Manager, Queue
from LFPy_util.sims import Simulation

class Worker(Process):
    def __init__(self, input_queue, output_queue, cell, run_param):
        super(Worker, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.cell = cell
        self.run_param = run_param

    def run(self):
        run_param = self.run_param
        cell = self.cell

        duration = run_param['duration']
        delay = run_param['delay']
        threshold = run_param['threshold']
        pptype = run_param['pptype']

        for sec in cell.allseclist:
            if 'soma' in sec.name():
                if pptype == 'ISyn':
                    syn = neuron.h.ISyn(0.5, sec=sec)
                if pptype == 'IClamp':
                    syn = neuron.h.IClamp(0.5, sec=sec)
                else:
                    syn = None

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

            cell.simulate(rec_imem=True)

            # Find spikes.
            spike_indices = LFPy_util.data_extraction.find_spikes(cell.tvec,
                                                            cell.somav,
                                                            threshold,
                                                            )
            isi = np.diff(spike_indices) * cell.timeres_NEURON
            if isi.size != 0:
                isi = np.mean(isi)
                freq = 1000/isi # Hz
            else:
                isi = 0
                freq = 0
            output = (i, freq)
            self.output_queue.put(output)

class CurrentSweep(Simulation):

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.set_name('sweep')

        # Used by the custom simulate and plot function.
        self.run_param['threshold'] = 4
        self.run_param['pptype'] = 'IClamp'
        self.run_param['delay'] = 0
        self.run_param['duration'] = 1000
        self.run_param['amp_start'] = 0.0
        self.run_param['amp_end'] = 3
        self.run_param['sweeps'] = 20
        self.run_param['processes'] = 4
        self.verbose = False

    def simulate(self, cell):
        run_param = self.run_param

        # String to put before output to the terminal.
        str_start = self.name
        str_start += " "*(20 - len(self.name)) + ": "

        amps = np.linspace(run_param['amp_start'], 
                           run_param['amp_end'], 
                           run_param['sweeps'])
        amps = amps.tolist()
        
        input_queue = Queue()
        output_queue = Queue()
        workers = []
        for i in xrange(run_param['processes']):
            worker = Worker(input_queue, output_queue, cell, run_param)
            worker.start()
            workers.append(worker)
        for i, amp in enumerate(amps):
            input_queue.put((i, amp))
        for i in xrange(run_param['processes']):
            input_queue.put(None)
        for worker in workers:
            worker.join()

        isi = np.zeros(run_param['sweeps'])
        for i in xrange(run_param['sweeps']):
            data = output_queue.get()
            i, freq = data
            isi[i] = freq
        print isi


    def process_data(self):
        pass

    def plot(self, dir_plot):
        data = self.data
        run_param = self.run_param
