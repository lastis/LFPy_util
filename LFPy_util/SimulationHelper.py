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
        self._sim_stack = []
        self._sim_stack_flag = []

        self.cell = None
        self.save = True
        # self.plot_at_runtime = False
        self.verbatim = True
        self.parallel = True
        self.parallel_plot = True

        self.set_cell(cell)
        self.set_dir_neurons("neuron")

    def __str__(self):
        text = ""
        text += "## SimulationHelper ##" + "\n"
        text += "neuron name         : " + self._neuron_name + "\n"
        text += "dir_neuron          : " + self._dir_neuron + "\n" 
        text += "verbatim            : " + str(self.verbatim)+ "\n"
        text += "run in parallel     : " + str(self.parallel)+ "\n"
        text += "plot in parallel    : " + str(self.parallel_plot)
        return text

    def set_neuron_name(self, name):
        self._neuron_name = name
        # Update directories.
        self.set_dir_neurons(self._dir_neurons)

    def set_dir_neurons(self, path):
        self._dir_neurons = path
        # Child of _dir_neurons.
        self._dir_neuron = os.path.join(self._dir_neurons,self._neuron_name)
        # Child of _dir_neuron.
        self._dir_plot = os.path.join(self._dir_neuron,"plot")
        self._dir_data = os.path.join(self._dir_neuron,"data")

    def set_cell(self, cell):
        self.cell = cell

    def push(self, sim_or_func, own_process):
        self._sim_stack.append(sim_or_func)
        self._sim_stack_flag.append(own_process)

    @staticmethod
    def _simulate(sim,cell,dir_data):
        sim.simulate(cell)
        if dir_data is not None:
            sim.save(dir_data)

    @staticmethod
    def _plot(sim,dir_plot):
        sim.process_data()
        sim.plot(dir_plot)

    def get_path_data(self,sim):
        fname = sim.ID+"_data"
        path = os.path.join(self._dir_data,fname)
        return path

    def get_path_run_param(self,sim):
        fname = sim.ID+"_run_param"
        path = os.path.join(self._dir_data,fname)
        return path

    def simulate(self):
        manager = Manager()
        if self.verbatim:
            print "starting simulation : " + self._neuron_name
        # Run the simulations.
        process_list = []
        for i, sim_or_func in enumerate(self._sim_stack):
            flag = self._sim_stack_flag[i]
            if isinstance(sim_or_func, LFPy_util.sims.Simulation):
                sim = sim_or_func
                if flag:
                    # Start in new process.
                    if self.verbatim:
                        print "new process         : " \
                                + self._neuron_name + " " + sim.__str__()
                    # Replace sim.data and sim.run_param with shared memory
                    # versions so data can be retrived from subprocesses.
                    sim.data = manager.dict(sim.data)
                    sim.run_param = manager.dict(sim.run_param)
                    if self.save:
                        process = Process(
                                target=self._simulate,
                                args=(sim,self.cell,self._dir_data),
                                )
                    else:
                        process = Process(
                                target=self._simulate,
                                args=(sim,self.cellNone),
                                )
                    process.start()
                    # If running in parallel, start all processes before
                    # joining them.
                    if self.parallel:
                        process_list.append(process)
                    else:
                        process.join()
                else:
                    # Start in current process.
                    if self.verbatim:
                        print "current process     : " \
                                + self._neuron_name + " " + sim.__str__()
                    if self.save:
                        self._simulate(sim,self.cell,self._dir_data)
                    else:
                        self._simulate(sim,self.cell,None)
            # If not a Simulation object assume it is a function.
            else:
                func = sim_or_func
                func(self.cell)
        # If running in parallel, join all the processes here instead.
        if self.parallel:
            for process in process_list:
                process.join()


    def plot(self):
        if self.verbatim:
            print "starting plotting   : " 

        process_list = []
        for i, sim_or_func in enumerate(self._sim_stack):
            if isinstance(sim_or_func, LFPy_util.sims.Simulation):
                sim = sim_or_func
                if not sim.data:
                    print "loading data from : " \
                            + self._dir_data + "/" + sim.ID + "_data."\
                            + sim.format_save_data
                    sim.load(self._dir_data)
                    if not sim.data:
                        raise ValueError("No data to plot.")
                if self.parallel_plot:
                    process = Process(target=self._plot,args=(sim,self._dir_plot))
                    process.start()
                    process_list.append(process)
                else:
                    self._plot(sim,self._dir_plot)
        for process in process_list:
            process.join()



        
