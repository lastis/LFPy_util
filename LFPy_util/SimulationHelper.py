import os 
from multiprocessing import Process, Manager
import LFPy_util

class SimulationHelper(object):
    """docstring for SimulationHandler"""
    def __init__(self, cell=None):
        self._neuron_name = "unnamed_neuron"
        self._dir_neurons = None
        self._dir_neuron = None
        self._dir_plot = None
        self._dir_data = None
        self._simulation_stack = []
        self._simulation_stack_flag = []

        self.cell = None
        self.save = True
        self.plot_at_runtime = False
        self.verbatim = True
        self.parallel = True
        self.parallel_plot = True

        self.set_cell(cell)
        self.set_dir_neurons(".")

    def __str__(self):
        text = ""
        text += "## SimulationHelper ##" + "\n"
        text += "neuron name         : " + self._neuron_name + "\n"
        text += "dir_neuron          : " + self._dir_neuron + "\n" 
        text += "verbatim            : " + str(self.verbatim)+ "\n"
        text += "run in parallel     : " + str(self.parallel)+ "\n"
        text += "plot at runtime     : " + str(self.plot_at_runtime)+ "\n"
        text += "plot in parallel    : " + str(self.parallel_plot)
        return text

    def set_neuron_name(self, name):
        self._neuron_name = name
        # Update directories.
        self.set_dir_neurons(self._dir_neurons)

    def set_dir_neurons(self, dir):
        self._dir_neurons = dir
        # Child of _dir_neurons.
        self._dir_neuron = os.path.join(self._dir_neurons,self._neuron_name)
        # Child of _dir_neuron.
        self._dir_plot = os.path.join(self._dir_neuron,"plot")
        self._dir_data = os.path.join(self._dir_neuron,"data")

    def set_cell(self, cell):
        self.cell = cell

    def push(self, sim_or_func, own_process):
        self._simulation_stack.append(sim_or_func)
        self._simulation_stack_flag.append(own_process)

    def simulate(self):
        if self.verbatim:
            print "starting simulation : " 
        # Store variables so they can be reset later.
        for i, sim_or_func in enumerate(self._simulation_stack):
            if isinstance(sim_or_func, LFPy_util.sims.Simulation):
                simulation = sim_or_func
                a = simulation.plot_at_runtime
                b = simulation.dir_data
                c = simulation.dir_plot
                d = simulation.save

        # Run the simulations.
        simulation_list = []
        for i, sim_or_func in enumerate(self._simulation_stack):
            flag = self._simulation_stack_flag[i]
            if isinstance(sim_or_func, LFPy_util.sims.Simulation):
                simulation = sim_or_func
                simulation.plot_at_runtime = self.plot_at_runtime
                simulation.dir_data = self._dir_data
                simulation.dir_plot = self._dir_plot
                simulation.save_data = self.save
                simulation.cell = self.cell
                # Start in new process.
                if flag:
                    if self.verbatim:
                        print "new process         : " + simulation.__str__()
                    simulation.start()
                    # If running in parallel, start all processes before
                    # joining them.
                    if self.parallel:
                        simulation_list.append(simulation)
                    else:
                        simulation.join()
                else:
                    if self.verbatim:
                        print "current process     : " + simulation.__str__()
                    simulation.run()
            # If not a Simulation object assume it is a function.
            else:
                func = sim_or_func
                func(cell)
        # If running in parallel, join all the processes here instead.
        if self.parallel:
            for simulation in simulation_list:
                simulation.join()

        # Reset some variables.
        for i, sim_or_func in enumerate(self._simulation_stack):
            if isinstance(sim_or_func, LFPy_util.sims.Simulation):
                simulation.plot_at_runtime = a
                simulation.dir_data = b
                simulation.dir_plot = c
                simulation.save = d

    def plot(self):
        if self.verbatim:
            print "starting plotting   : " 
        # Store variables so they can be reset later.
        for i, sim_or_func in enumerate(self._simulation_stack):
            if isinstance(sim_or_func, LFPy_util.sims.Simulation):
                simulation = sim_or_func
                a = simulation.dir_data
                b = simulation.dir_plot

        process_list = []
        for i, sim_or_func in enumerate(self._simulation_stack):
            if isinstance(sim_or_func, LFPy_util.sims.Simulation):
                simulation = sim_or_func
                simulation.dir_data = self._dir_data
                simulation.dir_plot = self._dir_plot

                if not simulation.results:
                    print "loading data        : " \
                            + self._dir_data + "/" + simulation.fname_results \
                            + "." + simulation.format_save_results
                    simulation.load()
                    if not simulation.results:
                        raise ValueError("No results to plot.")
                if self.parallel_plot:
                    process = Process(target=simulation.plot)
                    process.start()
                    process_list.append(process)
                else:
                    simulation.plot()
        for process in process_list:
            process.join()
        # Reset some variables.
        for i, sim_or_func in enumerate(self._simulation_stack):
            if isinstance(sim_or_func, LFPy_util.sims.Simulation):
                simulation.dir_data = a
                simulation.dir_plot = b




        
