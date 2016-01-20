import LFPy_util
import os
class Simulation():
    """docstring for ClassName"""
    def __init__(self):
        self.ID = "unnamed"
        self.format_save_data = 'pkl'
        self.format_save_run_param = 'js'
        self.data = {}
        self.run_param = {}

    def __str__(self):
        return self.ID

    def simulate(self,cell):
        raise NotImplementedError("Function must be overrided.")

    def process_data(self):
        raise NotImplementedError("Function must be overrided.")

    def plot(self,dir_plot):
        raise NotImplementedError("Function must be overrided.")

    def save(self,dir_data):
        # Save data.
        fname = self.ID+"_data"
        path = os.path.join(dir_data,fname)
        if self.format_save_data == 'pkl':
            LFPy_util.other.save_kwargs(path,**self.data)
        elif self.format_save_data == 'js':
            LFPy_util.other.save_kwargs_json(path,**self.data)
        else:
            raise ValueError("Unsupported format")
        # Save run param.
        fname = self.ID+"_run_param"
        path = os.path.join(dir_data,fname)
        if self.format_save_run_param == 'pkl':
            LFPy_util.other.save_kwargs(path,**self.run_param)
        elif self.format_save_run_param == 'js':
            LFPy_util.other.save_kwargs_json(path,**self.run_param)
        else:
            raise ValueError("Unsupported format")

    def load(self,dir_data):
        # Load data.
        fname = self.ID+"_data"
        path = os.path.join(dir_data,fname)
        if self.format_save_data == 'pkl':
            self.data = LFPy_util.other.load_kwargs(path)
        elif self.format_save_data == 'js':
            self.data = LFPy_util.other.load_kwargs_json(path)
        else:
            raise ValueError("Unsupported format")

        # Load the run parameters.
        fname = self.ID+"_run_param"
        path = os.path.join(dir_data,fname)
        if self.format_save_run_param == 'pkl':
            self.run_param = LFPy_util.other.load_kwargs(path)
        elif self.format_save_run_param == 'js':
            self.run_param = LFPy_util.other.load_kwargs_json(path)
        else:
            raise ValueError("Unsupported format")

            
