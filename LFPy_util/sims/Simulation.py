"""
Module holds only one class.
"""
import os
import LFPy_util


class Simulation(object):
    """
    Some guidelines:
    * run_param should not contain values that are used to process_data.
    Parameters that effect the proccesing of the data should be saved as
    class variables.

    If this is done it is possible to process data in different ways
    using the saved data from one simulation. The opposite is true for the
    simulate function, see below.
    * The run_param should uniquely define the data in the simulate function.
    If class variables are used to create the data in the simulate function it
    will not be possible to recreate the data from only the run_param.
    * Simulation.name should have its value set at initiation. This
    makes it more easy to use the LFPy_util.other.collect_data function.

    """

    def __init__(self):
        """
        """
        self.name = "unnamed"
        self.format_save_data = 'pkl'
        self.format_save_run_param = 'js'
        self.data = {}
        self.run_param = {}
        self.process_param = {}
        self.plot_param = {}
        self._str_data = "_data"
        self._str_run_param = "_run_param"

    def __str__(self):
        return self.name

    def previous_run(self, dir_data):
        """
        Called by Simulator during each neuron simulation.
        Is usually empty.
        """
        pass

    def simulate(self, cell):
        """
        Start simulation
        """
        raise NotImplementedError("Function must be overrided.")

    def process_data(self):
        """
        Process data from the simulation function.
        """
        raise NotImplementedError("Function must be overrided.")

    def plot(self, dir_plot):
        """
        Plot from simulation. Usually using self.run_param and self.data.
        """
        raise NotImplementedError("Function must be overrided.")

    def get_fname_data(self):
        """
        Get the filename of the data that is stored by the simulation.
        """
        return self.name + self._str_data + "." + self.format_save_data

    def get_fname_run_param(self):
        """
        Get the filename of the run_param that is stored by the simulation.
        """
        return self.name + self._str_run_param + "." \
            + self.format_save_run_param

    def save(self, dir_data):
        """
        Store run_param and data to files specified by .format_save_data and
        .format_save_run_param.
        """
        # Save data.
        fname = self.get_fname_data()
        path = os.path.join(dir_data, fname)
        if self.format_save_data == 'pkl':
            LFPy_util.other.save_kwargs(path, **self.data)
        elif self.format_save_data == 'js':
            LFPy_util.other.save_kwargs_json(path, **self.data)
        else:
            raise ValueError("Unsupported format")
        # Save run param.
        fname = self.get_fname_run_param()
        path = os.path.join(dir_data, fname)
        if self.format_save_run_param == 'pkl':
            LFPy_util.other.save_kwargs(path, **self.run_param)
        elif self.format_save_run_param == 'js':
            LFPy_util.other.save_kwargs_json(path, **self.run_param)
        else:
            raise ValueError("Unsupported format")

    def load(self, dir_data):
        """
        Load run_param and data that has been stored according to the
        save function.
        """
        # Load data.
        fname = self.get_fname_data()
        path = os.path.join(dir_data, fname)
        if self.format_save_data == 'pkl':
            self.data = LFPy_util.other.load_kwargs(path)
        elif self.format_save_data == 'js':
            self.data = LFPy_util.other.load_kwargs_json(path)
        else:
            raise ValueError("Unsupported format")

        # Load the run parameters.
        fname = self.get_fname_run_param()
        path = os.path.join(dir_data, fname)
        if self.format_save_run_param == 'pkl':
            self.run_param = LFPy_util.other.load_kwargs(path)
        elif self.format_save_run_param == 'js':
            self.run_param = LFPy_util.other.load_kwargs_json(path)
        else:
            raise ValueError("Unsupported format")
