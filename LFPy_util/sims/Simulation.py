import LFPy_util
import os
from multiprocessing import Process, Manager
class Simulation(Process):
    """docstring for ClassName"""
    def __init__(self):
        super(Simulation,self).__init__()

        manager = Manager()
        self.results = manager.dict()
        self.run_param = manager.dict()
        self.fname_run_param = "unnamed_run_param"
        self.fname_results = "unnamed_data"

        self.format_save_results = 'pkl'
        self.format_save_run_param = 'js'
        # Can be set to false to avoid overriding run_param on load.
        self.load_run_param = True

        # These values are normally set by SimulationHelper.
        self.cell = None
        self.plot_at_runtime = False
        self.dir_data = "."
        self.dir_plot = "."
        self.save_data = True


    def __str__(self):
        return "unnamed"

    def set_cell(cell):
        self.cell = cell

    def run(self):
        self.simulate()
        if self.plot_at_runtime:
            self.plot()
        if self.save_data:
            self.save()

    def simulate(self):
        pass

    def plot(self):
        pass

    def save(self):
        # Save results.
        path = os.path.join(self.dir_data,self.fname_results)
        if self.format_save_results == 'pkl':
            LFPy_util.other.save_kwargs(path,**self.results)
        elif self.format_save_results == 'js':
            LFPy_util.other.save_kwargs_json(path,**self.results)
        else:
            raise ValueError("Unsupported format")
        # Save run param.
        path = os.path.join(self.dir_data,self.fname_run_param)
        if self.format_save_run_param == 'pkl':
            LFPy_util.other.save_kwargs(path,**self.run_param)
        elif self.format_save_run_param == 'js':
            LFPy_util.other.save_kwargs_json(path,**self.run_param)
        else:
            raise ValueError("Unsupported format")

    def load(self):
        manager = Manager()
        # Load results.
        path = os.path.join(self.dir_data,self.fname_results)
        if self.format_save_results == 'pkl':
            self.results = LFPy_util.other.load_kwargs(path)
        elif self.format_save_results == 'js':
            self.results = LFPy_util.other.load_kwargs_json(path)
        else:
            raise ValueError("Unsupported format")
        # Change the type of dict.
        self.results = manager.dict(self.results)

        if self.load_run_param:
            # Load the run parameters.
            path = os.path.join(self.dir_data,self.fname_run_param)
            if self.format_save_run_param == 'pkl':
                self.run_param = LFPy_util.other.load_kwargs(path)
            elif self.format_save_run_param == 'js':
                self.run_param = LFPy_util.other.load_kwargs_json(path)
            else:
                raise ValueError("Unsupported format")
            # Change the type of dict.
            self.run_param = manager.dict(self.run_param)

            
