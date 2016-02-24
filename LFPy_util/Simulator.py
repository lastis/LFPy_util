"""
Simulator class
"""
import os
import inspect
from multiprocessing import Process, Manager
import LFPy_util


class Simulator(object):
    """docstring for SimulationHandler"""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, cell=None):
        self._neuron_list = None
        self._dir_neurons = None
        self._sim_stack = []
        self._sim_stack_flag = []
        self._str_data_dir = "data"
        self._str_plot_dir = "plot"
        self._get_cell = None
        self._cell = None

        self.save = True
        self.verbose = True
        self.parallel = True
        self.concurrent_neurons = 1

        self.set_cell(cell)
        self.set_dir_neurons("neuron")
        self.set_neuron_name("unnamed_neuron")

    def __str__(self):
        text = ""
        text += "## Simulator ##" + "\n"
        text += "neurons             : " + self._neuron_list[0] + "\n"
        for neuron in self._neuron_list[1:]:
            text += "                    : " + neuron + "\n"
        text += "concurrent neurons  : " + str(self.concurrent_neurons) + "\n"
        text += "parallel            : " + str(self.parallel)
        return text

    def set_neuron_name(self, name):
        """
        Set the name(s) of neuron(s) that will be simulated.
        The list will be sent to the cell load function if used.
        """
        # If input is a stiring, make it a list of strings.
        if not isinstance(name, list):
            name = [name]
        self._neuron_list = name

    def set_dir_neurons(self, path):
        """
        Set directory where simulation data and plots will be stored.
        """
        self._dir_neurons = path

    def set_cell(self, cell):
        """
        Set the cell model. Only used if just one neuron is simulated.
        """
        self._cell = cell

    def set_cell_load_func(self, func):
        """
        Set function to load a cell model. Must return a cell object
        and take a string as argument.
        """
        if len(inspect.getargspec(func)[0]) != 1:
            raise ValueError("The load cell function must have one argument" +
                             " and return a Cell object.")
        self._get_cell = func

    def _is_ready(self):
        if self._get_cell is None and len(self._neuron_list) != 1:
            raise ValueError("List of neurons is used but load cell "
                             "functionis not set.")
        if self._cell is None and self._get_cell is None:
            raise ValueError("Load cell function and cell object missing."
                             "One must be supplied.")

    def push(self, sim_or_func, own_process=True):
        """
        Push simulation or function to the simulator.
        """
        # if isinstance(sim_or_func, LFPy_util.sims.Simulation):
        #     sim_or_func = copy.deepcopy(sim_or_func)
        self._sim_stack.append(sim_or_func)
        self._sim_stack_flag.append(own_process)

    def clear_list(self):
        """
        Clear the list of simulation objects.
        """
        self._sim_stack = []
        self._sim_stack_flag = []

    def get_path_neuron(self, index=0):
        """
        Get directory path to neuron by index.
        """
        return os.path.join(self._dir_neurons, self._neuron_list[index])

    def get_dir_neuron_data(self, index=0):
        """
        Get the directory where the data of the simulation is stored.
        Index corresponds to list index in the names of the neurons.
        """
        dir_neuron = self.get_path_neuron(index)
        return os.path.join(dir_neuron, self._str_data_dir)

    def get_dir_neuron_plot(self, index=0):
        """
        Get the directory where the plots of the simulation is stored.
        Index corresponds to list index in the names of the neurons.
        """
        dir_neuron = self.get_path_neuron(index)
        return os.path.join(dir_neuron, self._str_plot_dir)

    def get_path_sim_data(self, sim, index=0):
        """
        Get the path to the data file from a spesific simulation and neuron.
        """
        dir_data = self.get_dir_neuron_data(index)
        fname = sim.get_fname_data()
        path = os.path.join(dir_data, fname)
        return path

    def get_path_sim_run_param(self, sim, index=0):
        """
        Get the path to the run_param file from a spesific
        simulation and neuron.
        """
        dir_data = self.get_dir_neuron_data(index)
        fname = sim.get_fname_run_param()
        path = os.path.join(dir_data, fname)
        return path

    def plot(self):
        """
        Start plotting.
        Will run the number of concurrent neurons in parallel. 
        If parallel flag is true, each Simulation.plot function is run in parallel.
        """
        self._is_ready()
        process_list = []
        for i in xrange(len(self._neuron_list)):
            # For each neuron make a new process.
            process = Process(target=Simulator._plot_neuron, args=(self, i), )
            process_list.append(process)

        cnt = 0
        finished = False
        while not finished:
            for i in xrange(self.concurrent_neurons):
                if cnt + i >= len(self._neuron_list):
                    continue
                process = process_list[cnt + i]
                process.start()
            for i in xrange(self.concurrent_neurons):
                if cnt + i >= len(self._neuron_list):
                    continue
                process = process_list[cnt + i]
                process.join()
            cnt += self.concurrent_neurons
            if cnt >= len(self._neuron_list):
                finished = True

    def simulate(self):
        """
        Start simulations.
        """
        self._is_ready()
        process_list = []
        for i in xrange(len(self._neuron_list)):
            # For each neuron make a new process.
            process = Process(target=Simulator._simulate_neuron, args=(self, i), )
            process_list.append(process)

        cnt = 0
        finished = False
        while not finished:
            for i in xrange(self.concurrent_neurons):
                if cnt + i >= len(self._neuron_list):
                    continue
                process = process_list[cnt + i]
                process.start()
            for i in xrange(self.concurrent_neurons):
                if cnt + i >= len(self._neuron_list):
                    continue
                process = process_list[cnt + i]
                process.join()
            cnt += self.concurrent_neurons
            if cnt >= len(self._neuron_list):
                finished = True

    @staticmethod
    def _simulate(sim, cell, dir_data, save):
        sim.previous_run(dir_data)
        sim.simulate(cell)
        if save:
            print "saving data to      : " \
                + os.path.join(dir_data, sim.get_fname_data()) 
            sim.save(dir_data)

    @staticmethod
    def _plot(sim, dir_plot, dir_data):
        if not sim.data:
            print "loading data from   : " \
                + os.path.join(dir_data, sim.get_fname_data()) 
            sim.load(dir_data)
            if not sim.data:
                raise ValueError("No data to plot.")
        sim.process_data()
        sim.plot(dir_plot)

    def _simulate_neuron(self, index=0):
        # This function (can be) is run in
        # parallel. Which means there are multiple instances of self and
        # hopefully all of its parameters have been deep copied.
        cell = self._cell
        if cell is None:
            cell = self._get_cell(self._neuron_list[index])

        manager = Manager()
        if self.verbose:
            print "starting simulation : " + self._neuron_list[index]
        # Run the simulations.
        process_list = []
        for i, sim_or_func in enumerate(self._sim_stack):
            flag = self._sim_stack_flag[i]
            if isinstance(sim_or_func, LFPy_util.sims.Simulation):
                sim = sim_or_func
                dir_data = self.get_dir_neuron_data(index)
                # Start the simulation in a new process if the 
                # flag is true.
                if flag:
                    if self.verbose:
                        print "new process         : "\
                            + self._neuron_list[index] \
                            + " " + sim.__str__()
                    # Replace sim.data and sim.run_param with shared memory
                    # versions so data can be retrived from subprocesses.
                    sim.data = manager.dict(sim.data)
                    sim.run_param = manager.dict(sim.run_param)

                    process = Process(
                        target=Simulator._simulate,
                        args=(sim, cell, dir_data, self.save), )
                    process.start()
                    # End the simulation here if parallel is not enabled.
                    if self.parallel:
                        process_list.append(process)
                    else:
                        process.join()
                else:
                    if self.verbose:
                        print "current process     : " \
                            + self._neuron_list[index] +\
                            " " + sim.__str__()
                    self._simulate(sim, cell, dir_data, self.save)
            # If not a Simulation object assume it is a function.
            else:
                func = sim_or_func
                if flag:
                    if self.verbose:
                        print "new process         : "\
                            + self._neuron_list[index] \
                            + " " + func.__name__
                    process = Process(
                        target=func,
                        args=(cell,)
                        )
                    process.start()
                    # End the function here if parallel is not enabled.
                    if self.parallel:
                        process_list.append(process)
                    else:
                        process.join()
                else:
                    if self.verbose:
                        print "current process     : " \
                            + self._neuron_list[index] \
                            + " " + func.__name__
                    func(cell)
        # Join all processes.
        for process in process_list:
            process.join()

    def _plot_neuron(self, index):
        if self.verbose:
            print "starting plotting   : "
        process_list = []
        for i, sim_or_func in enumerate(self._sim_stack):
            if isinstance(sim_or_func, LFPy_util.sims.Simulation):
                sim = sim_or_func
                dir_plot = self.get_dir_neuron_plot(index)
                dir_data = self.get_dir_neuron_data(index)
                # Start each Simulation.plot in a new process.
                process = Process(target=self._plot, args=(sim, dir_plot, dir_data))
                process_list.append(process)

        # Start and end plotting.
        for process in process_list:
            process.start()
            if not self.parallel:
                process.join()
        if self.parallel:
            process.join()

