"""
Simulator class
"""
import os
import inspect
from multiprocessing import Process, Manager
import LFPy_util
import numpy.random as random


class Simulator(object):

    # pylint: disable=too-many-instance-attributes
    def __init__(self):
        self._neuron_name = None
        self._dir_output = None
        self._sim_stack = []
        self._sim_stack_flag = []
        self._str_data_dir = "data"
        self._str_plot_dir = "plot"
        self._get_cell = None

        self.verbose = True
        self.parallel = True

        self.set_output_dir(".")
        self.set_neuron_name("unnamed_neuron")

    def __str__(self):
        text = ""
        # text += "## Simulator ##" + "\n"
        text += "neuron              : " + self._neuron_name + "\n"
        if len(self._sim_stack) > 0 :
            text += "simulation list     : " + self._sim_stack[0].__str__() + "\n"
            for sim in self._sim_stack[1:] :
                text += "                    : " + sim.__str__() + "\n"
        text += "parallel            : " + str(self.parallel)
        return text

    def set_neuron_name(self, name):
        """
        """
        self._neuron_name = name

    def set_output_dir(self, path):
        """
        Set directory where simulation data and plots will be stored.
        """
        self._dir_output = path

    def set_cell_load_func(self, func):
        """
        Set function to load a cell model. Must return a cell object
        and take a string as argument.
        """
        if len(inspect.getargspec(func)[0]) != 1:
            raise ValueError("The load cell function must have one argument" +
                             " and return a LFPy Cell object.")
        self._get_cell = func

    def _is_ready(self):
        if self._get_cell is None:
            raise ValueError("Load cell function missing.")

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

    def get_dir_neuron(self):
        """
        Get directory path to neuron.
        """
        return os.path.join(self._dir_output, self._neuron_name)

    def get_dir_neuron_data(self):
        """
        Get the directory where the data of the simulation is stored.
        """
        dir_neuron = self.get_dir_neuron()
        return os.path.join(dir_neuron, self._str_data_dir)

    def get_dir_neuron_plot(self):
        """
        Get the directory where the plots of the simulation is stored.
        """
        dir_neuron = self.get_dir_neuron()
        return os.path.join(dir_neuron, self._str_plot_dir)

    def get_dir_sim_plot(self, sim):
        """
        Get the directory the plotting folder of the spesific simulation.
        """
        dir_plot = self.get_dir_neuron_plot()
        return os.path.join(dir_plot, sim.name)

    def get_path_sim_data(self, sim):
        """
        Get the path to the data file from a spesific simulation.
        """
        dir_data = self.get_dir_neuron_data()
        fname = sim.get_fname_data()
        path = os.path.join(dir_data, fname)
        return path

    def get_path_sim_run_param(self, sim):
        """
        Get the path to the run_param file from a spesific
        simulation and neuron.
        """
        dir_data = self.get_dir_neuron_data()
        fname = sim.get_fname_run_param()
        path = os.path.join(dir_data, fname)
        return path


    @staticmethod
    def _simulate(sim, cell, dir_data):
        sim.previous_run(dir_data)
        sim.simulate(cell)
        if sim.data:
            print "saving data to      : " \
                + os.path.join(dir_data, sim.get_fname_data()) 
            sim.save(dir_data)
        else:
            print 'nothing to save     :'

    @staticmethod
    def _plot(sim, dir_plot, dir_data):
        print "loading data from   : " \
            + os.path.join(dir_data, sim.get_fname_data()) 
        sim.load(dir_data)
        # Commented out code to allow loading empty data.
        # if not sim.data:
        #     raise ValueError("No data to plot.")
        sim.process_data()
        sim.save_info(dir_plot)
        sim.plot(dir_plot)

    def simulate(self):
        cell = self._get_cell(self._neuron_name)

        manager = Manager()
        # if self.verbose:
        #     print "starting simulation : " + self._neuron_name
        # Run the simulations.
        process_list = []
        for i, sim_or_func in enumerate(self._sim_stack):
            flag = self._sim_stack_flag[i]
            if isinstance(sim_or_func, LFPy_util.sims.Simulation):
                sim = sim_or_func

                dir_data = self.get_dir_neuron_data()
                # Start the simulation in a new process if the 
                # flag is true.
                if flag:
                    if self.verbose:
                        print "new process         : "\
                            + self._neuron_name \
                            + " " + sim.__str__()
                    # # Replace dictionaries with shared memory
                    # # versions so data can be retrived from subprocesses.
                    sim.data = manager.dict(sim.data)
                    sim.run_param = manager.dict(sim.run_param)
                    sim.info = manager.dict(sim.info)

                    process = Process(
                        target=self._simulate,
                        args=(sim, cell, dir_data), 
                        )
                    process.start()
                    # End the simulation here if parallel is not enabled.
                    if self.parallel:
                        process_list.append(process)
                    else:
                        process.join()
                else:
                    if self.verbose:
                        print "current process     : " \
                            + self._neuron_name +\
                            " " + sim.__str__()
                    self._simulate(sim, cell, dir_data)
            # If not a Simulation object, assume it is a function.
            else:
                func = sim_or_func
                if flag:
                    if self.verbose:
                        print "new process         : "\
                            + self._neuron_name \
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

    def plot(self):
        # if self.verbose:
        #     print "starting plotting   : " + self._neuron_name
        process_list = []
        for i, sim_or_func in enumerate(self._sim_stack):
            if isinstance(sim_or_func, LFPy_util.sims.Simulation):
                sim = sim_or_func
                dir_plot = self.get_dir_sim_plot(sim)
                dir_data = self.get_dir_neuron_data()
                if self.verbose:
                    print "plotting            : "\
                        + self._neuron_name \
                        + " " + sim.__str__()
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

