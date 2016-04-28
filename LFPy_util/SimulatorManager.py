"""
Simulator class
"""
import os
import inspect
from multiprocessing import Process, Manager
import LFPy_util

class SimulatorManager(object):
    """docstring for SimulationHandler"""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, cell=None):
        self._neuron_names = []
        self._get_simulator = None

        self.verbose = True
        self.concurrent_neurons = 1

    def __str__(self):
        text = ''
        for neuron_name in self._neuron_names:
            text += self._get_simulator(neuron_name).__str__() + "\n"
        text += "concurrent neurons  : " + str(self.concurrent_neurons) + "\n"
        text += "verbose             : " + str(self.verbose)
        return text

    def set_neuron_names(self, names):
        """
        """
        # If input is a string, make it a list of strings.
        if not isinstance(names, list):
            names = [names]
        self._neuron_names = names

    def set_sim_load_func(self, func):
        """
        Set function to load a simulator.
        """
        if len(inspect.getargspec(func)[0]) != 1:
            raise ValueError("The load cell function must have one argument" +
                             " and return a Simulator object.")
        self._get_simulator = func

    def _is_ready(self):
        if self._get_simulator is None:
            raise ValueError("Load simulator function missing.")
        return

    def simulate(self):
        """
        Start simulations.
        """
        self._is_ready()

        process_list = []
        for i in xrange(len(self._neuron_names)):
            # For each neuron make a new process.
            neuron_name = self._neuron_names[i]
            simulator = self._get_simulator(neuron_name)
            process = Process(target=simulator.simulate)
            process_list.append(process)

        cnt = 0
        finished = False
        while not finished:
            for i in xrange(self.concurrent_neurons):
                if cnt + i >= len(self._neuron_names):
                    continue
                process = process_list[cnt + i]
                process.start()
            for i in xrange(self.concurrent_neurons):
                if cnt + i >= len(self._neuron_names):
                    continue
                process = process_list[cnt + i]
                process.join()
            cnt += self.concurrent_neurons
            if cnt >= len(self._neuron_names):
                finished = True

    def plot(self):
        """
        Start plotting.
        Will run the number of concurrent neurons in parallel. 
        If parallel flag is true, each Simulation.plot function is run in parallel.
        """
        self._is_ready()
        process_list = []
        for i in xrange(len(self._neuron_names)):
            # For each neuron make a new process.
            neuron_name = self._neuron_names[i]
            simulator = self._get_simulator(neuron_name)
            process = Process(target=simulator.plot)
            process_list.append(process)

        cnt = 0
        finished = False
        while not finished:
            for i in xrange(self.concurrent_neurons):
                if cnt + i >= len(self._neuron_names):
                    continue
                process = process_list[cnt + i]
                process.start()
            for i in xrange(self.concurrent_neurons):
                if cnt + i >= len(self._neuron_names):
                    continue
                process = process_list[cnt + i]
                process.join()
            cnt += self.concurrent_neurons
            if cnt >= len(self._neuron_names):
                finished = True
