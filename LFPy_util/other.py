import os
import neuron
import subprocess
import LFPy_util
import pickle
import json
import numpy as np
import inspect
from neuron import hoc


def collect_data(dir_neurons, sim, func):
    if len(inspect.getargspec(func)[0]) != 4:
        raise ValueError("DataCollection function must have 3 arguments.")
    neurons = os.listdir(dir_neurons)
    s = LFPy_util.Simulator()
    s.set_dir_neurons(dir_neurons)
    for neuron in neurons:
        s.set_neuron_name(neuron)
        path_data = s.get_path_sim_data(sim)
        path_run_param = s.get_path_sim_run_param(sim)
        sim.data = LFPy_util.other.load_kwargs(path_data)
        sim.run_param = LFPy_util.other.load_kwargs_json(path_run_param)
        sim.process_data()
        func(neuron, sim.ID, sim.run_param, sim.data)


def nrnivmodl(directory='.', suppress=False):
    """
    Should avoid using relative paths as neuron will complain on
    running neuron.load_mechanisms twice on the same directory path.
    """
    tmp = os.getcwd()
    with suppress_stdout_stderr(suppress):
        os.chdir(directory)
        devnull = open(os.devnull, 'w')
        subprocess.call(['nrnivmodl'], stdout=devnull, shell=True)
        neuron.load_mechanisms(directory)
    os.chdir(tmp)


def save_kwargs(path, **kwargs):
    # Create the directory path if it doesn't exist yet.
    directory = os.path.dirname(path)
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)
    _, ext = os.path.splitext(path)
    if ext == '':
        path = path + '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(kwargs, f, pickle.HIGHEST_PROTOCOL)


def load_kwargs(path):
    _, ext = os.path.splitext(path)
    if ext == '':
        path = path + '.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_kwargs_json(path, **kwargs):
    # Create the directory path if it doesn't exist yet.
    directory = os.path.dirname(path)
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)
    _, ext = os.path.splitext(path)
    if ext == '':
        path = path + '.js'

    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, hoc.HocObject):
            return obj.to_python()
        raise TypeError('Not serializeable')
    # Save the kwargs to json.
    data_string = json.dumps(kwargs, default=default, indent=4, sort_keys=True)
    f = open(path, 'w')
    f.write(data_string)
    f.close()


def load_kwargs_json(path):
    """
    Loads a json file.

    Should probably be renamed load something. 

    :param string path: 
        Absolute or relative path. 
    :returns: 
        :class:`dict` -- json file.
    """
    _, ext = os.path.splitext(path)
    if ext == '':
        path = path + '.js'
    f = open(path, 'r')
    param_dict = json.load(f)
    f.close()
    return param_dict


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    Added supress as a boolean keyword, if false it will not supress anything.
    Done to enable easier usage.

    '''

    def __init__(self, suppress=True):
        self.suppress = suppress
        if self.suppress:
            # Open a pair of null files
            self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
            # Save the actual stdout (1) and stderr (2) file descriptors.
            self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        if self.suppress:
            # Assign the null pointers to stdout and stderr.
            os.dup2(self.null_fds[0], 1)
            os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        if self.suppress:
            # Re-assign the real stdout/stderr back to (1) and (2)
            os.dup2(self.save_fds[0], 1)
            os.dup2(self.save_fds[1], 2)
            # Close the null files
            os.close(self.null_fds[0])
            os.close(self.null_fds[1])
