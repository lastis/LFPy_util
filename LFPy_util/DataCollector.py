import os 
from glob import glob
import inspect
import LFPy_util

class DataCollector(object):
    """docstring for DataCollector"""
    def __init__(self):
        self._dir_neurons = None
        self._data_func = None

        self.verbatim = True
        self.set_dir_neurons(".")
        
    def set_dir_neurons(self, path):
        self._dir_neurons = path

    def set_collection_func(self, func):
        if len(inspect.getargspec(func)[0]) != 3:
            raise ValueError("DataCollection function must have 3 arguments.")
        self._data_func = func

    def run(self):
        if self._data_func is None:
            raise TypeError("DataCollector has not been assiged collection function.")
        neurons = os.listdir(self._dir_neurons)
        for neuron in neurons:
            path = os.path.join(self._dir_neurons,neuron)
            if not os.path.isdir(path):
                continue
            dir_data = os.path.join(path,"data")
            expr_pkl = os.path.join(dir_data,"*.pkl")
            expr_js = os.path.join(dir_data,"*.js")

            files_pkl = glob(expr_pkl)
            files_js = glob(expr_js)

            for file in files_pkl:
                path = os.path.join(path,file)
                data = LFPy_util.other.load_kwargs(path)
                # Remove extention and parent dirs.
                file = os.path.splitext(path)[0]
                file = os.path.basename(file)
                self._data_func(neuron,file,data)



