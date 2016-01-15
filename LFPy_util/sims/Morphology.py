from Simulation import Simulation
import LFPy
import LFPy_util
import numpy as np
import os
from multiprocessing import Process, Manager



class Morphology(Simulation):
    """docstring for Grid"""

    def __init__(self):
        super(Morphology,self).__init__()
        # Used by the super save and load function.
        self.fname_run_param = 'morph_run_param'
        self.fname_results = 'morph_results'

        self.debug = False

        # Used by the custom plot function.
        self.show = False

        # Plot names.
        self.fname_morph_plot_xy     = 'morph_xy'
        self.fname_morph_plot_xz     = 'morph_xz'
        self.fname_morph_plot_yz     = 'morph_yz'

    def __str__(self):
        return "Morphology"

    def simulate(self):
        self.results['poly_morph_x_y'] = self.cell.get_idx_polygons(('x','y'))
        self.results['poly_morph_x_z'] = self.cell.get_idx_polygons(('x','z'))
        self.results['poly_morph_y_z'] = self.cell.get_idx_polygons(('z','y'))

    def plot(self):
        results = self.results
        run_param = self.run_param

        # Plot.
        LFPy_util.plot.morphology(
            results['poly_morph_x_y'],
            fname=self.fname_morph_plot_xy,
            plot_save_dir=self.dir_plot,
            show=self.show,
            mirror=True,
            x_label='y',
            y_label='x',
        )
        LFPy_util.plot.morphology(
            results['poly_morph_x_z'],
            fname=self.fname_morph_plot_xz,
            plot_save_dir=self.dir_plot,
            show=self.show,
            x_label='x',
            y_label='z',
        )
        LFPy_util.plot.morphology(
            results['poly_morph_y_z'],
            fname=self.fname_morph_plot_yz, 
            plot_save_dir=self.dir_plot,
            mirror=True,
            x_label='y',
            y_label='z',
            show=self.show,
        )
