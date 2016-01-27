import os 
import inspect
from multiprocessing import Process, Manager
import LFPy_util

class Simulator(object):
    """docstring for SimulationHandler"""
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
        self.plot = True
        self.simulate = True
        self.verbatim = True
        self.parallel = True

        self.set_cell(cell)
        self.set_dir_neurons("neuron")
        self.set_neuron_name("unnamed_neuron")

    def __str__(self):
        text = ""
        text += "## Simulator ##" + "\n"
        text += "neurons             : " + self._neuron_list[0] + "\n"
        for neuron in self._neuron_list[1:]:
            text += "                    : " + neuron + "\n"
        text += "parallel            : " + str(self.parallel)+ "\n"
        text += "simulate            : " + str(self.simulate)+ "\n"
        text += "plot                : " + str(self.plot)
        return text

    def set_neuron_name(self, name):
        # If input is a stiring, make it a list of strings.
        if not isinstance(name,list):
            name = [name]
        self._neuron_list = name

    def set_dir_neurons(self, path):
        self._dir_neurons = path

    def set_cell(self, cell):
        self._cell = cell

    def set_cell_load_func(self, func):
        if len(inspect.getargspec(func)[0]) != 1:
            raise ValueError("The load cell function must have one argument and return a Cell object.")
        self._get_cell = func

    def _is_ready(self):
        if self._get_cell is None and len(self._neuron_list) != 1:
            raise ValueError("List of neurons is used but load cell function is not set.")
        if self._cell is None and self._get_cell is None:
            raise ValueError("Load cell function and cell object missing. One must be supplied.")

    def push(self, sim_or_func, own_process):
        self._sim_stack.append(sim_or_func)
        self._sim_stack_flag.append(own_process)


    def get_path_neuron(self, index=0):
        return os.path.join(self._dir_neurons,self._neuron_list[index])

    def get_dir_neuron_data(self,index=0):
        dir_neuron = self.get_path_neuron(index)
        return os.path.join(dir_neuron,self._str_data_dir)

    def get_dir_neuron_plot(self,index=0):
        dir_neuron = self.get_path_neuron(index)
        return os.path.join(dir_neuron,self._str_plot_dir)

    def get_path_sim_data(self, sim, index=0):
        dir_data = self.get_dir_neuron_data(index)
        fname = sim.get_fname_data()
        path = os.path.join(dir_data,fname)
        return path

    def get_path_sim_run_param(self,sim, index=0):
        dir_data = self.get_dir_neuron_data(index)
        fname = sim.get_fname_run_param()
        path = os.path.join(dir_data,fname)
        return path

    @staticmethod
    def _simulate(sim,cell,dir_data):
        sim.simulate(cell)
        if dir_data is not None:
            sim.save(dir_data)

    @staticmethod
    def _plot(sim,dir_plot):
        sim.process_data()
        sim.plot(dir_plot)

    def _run_neuron(self,index=0):
        # If everything works as intended, this function (can) is run in 
        # parallel. Which means there are multiple instances of self and
        # hopefully all of its parameters have been deep copied.
        if self.simulate:
            cell = self._cell
            if cell is None:
                cell = self._get_cell(self._neuron_list[index])
                
            manager = Manager()
            if self.verbatim:
                print "starting simulation : " + self._neuron_list[index]
            # Run the simulations.
            process_list = []
            for i, sim_or_func in enumerate(self._sim_stack):
                flag = self._sim_stack_flag[i]
                if isinstance(sim_or_func, LFPy_util.sims.Simulation):
                    sim = sim_or_func
                    dir_data = None
                    if self.save:
                        dir_data = self.get_dir_neuron_data(index)
                    if flag:
                        # Start in new process.
                        if self.verbatim:
                            print "new process         : " \
                                    + self._neuron_list[index] + " " + sim.__str__()
                        # Replace sim.data and sim.run_param with shared memory
                        # versions so data can be retrived from subprocesses.
                        sim.data = manager.dict(sim.data)
                        sim.run_param = manager.dict(sim.run_param)

                        process = Process(
                                target=Simulator._simulate,
                                args=(sim,cell,dir_data),
                                )
                        process.start()
                        process_list.append(process)
                    else:
                        # Start in current process.
                        if self.verbatim:
                            print "current process     : " \
                                    + self._neuron_list[index] + " " + sim.__str__()
                        self._simulate(sim,cell,dir_data)
                # If not a Simulation object assume it is a function.
                else:
                    func = sim_or_func
                    func(cell)
            # Join all processes.
            for process in process_list:
                process.join()

        if self.plot:
            if self.verbatim:
                print "starting plotting   : " 
            # process_list = []
            for i, sim_or_func in enumerate(self._sim_stack):
                if isinstance(sim_or_func, LFPy_util.sims.Simulation):
                    sim = sim_or_func
                    dir_plot = self.get_dir_neuron_plot(index)
                    if not sim.data:
                        print "loading data from   : " \
                                + self.get_path_sim_data(sim,index)
                        sim.load(self.get_dir_neuron_data(index))
                        if not sim.data:
                            raise ValueError("No data to plot.")
                    # Start each Simulation.plot in a new process.
                    process = Process(target=self._plot,args=(sim,dir_plot))
                    process.start()
                    process.join()


    def run(self):
        self._is_ready()
        process_list = []
        for i, neuron in enumerate(self._neuron_list):
            # For each neuron start a new process.
            process = Process(
                    target=Simulator._run_neuron,
                    args=(self, i),
                    )
            process.start()
            if self.parallel and len(self._neuron_list) != 1:
                process_list.append(process)
            else:
                process.join()
        for process in process_list:
            process.join()




        
