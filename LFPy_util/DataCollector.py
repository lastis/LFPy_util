import os 
from glob import glob

class DataCollector(object):
    """docstring for DataCollector"""
    def __init__(self):
        self._dir_neurons = None

        self.verbatim = True

        self.set_dir_neurons(".")
        
    def set_dir_neurons(self, path):
        self._dir_neurons = path

    def run(self):
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



