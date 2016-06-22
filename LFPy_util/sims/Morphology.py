import os
from Simulation import Simulation
import LFPy_util
import LFPy_util.data_extraction as de
import LFPy_util.plot as lplot
import LFPy_util.colormaps as cm
import matplotlib.pyplot as plt
import matplotlib as mpl


class Morphology(Simulation):
    """docstring for Grid"""

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.set_name("morph")

        self.debug = False

        self.plot_param['use_tex'] = True

    def simulate(self, cell):
        self.data['poly_morph_x_y'] \
                = de.get_polygons_no_axon(cell,('x','y'))
        self.data['poly_morph_x_y_axon'] \
                = de.get_polygons_axon(cell,('x','y'))
        self.data['poly_morph_x_z'] \
                = de.get_polygons_no_axon(cell,('x','z'))
        self.data['poly_morph_x_z_axon'] \
                = de.get_polygons_axon(cell,('x','z'))
        self.data['poly_morph_y_z'] \
                = de.get_polygons_no_axon(cell,('y','z'))
        self.data['poly_morph_y_z_axon'] \
                = de.get_polygons_axon(cell,('y','z'))

    def process_data(self):
        pass

    def plot(self, dir_plot):
        data = self.data

        LFPy_util.plot.set_rc_param(self.plot_param['use_tex'])

        LFPy_util.plot.morphology(data['poly_morph_x_y'],
                                  data['poly_morph_x_y_axon'],
                                  fname=self.name + '_xy_up',
                                  plot_save_dir=dir_plot,
                                  fig_size=lplot.size_tall,
                                  show=False,
                                  mirror=False,
                                  x_label='x',
                                  y_label='y', )

        LFPy_util.plot.morphology(data['poly_morph_x_y'],
                                  data['poly_morph_x_y_axon'],
                                  fname=self.name + '_xy',
                                  plot_save_dir=dir_plot,
                                  show=False,
                                  mirror=True,
                                  x_label='y',
                                  y_label='x', )

        LFPy_util.plot.morphology(data['poly_morph_x_z'],
                                  data['poly_morph_x_z_axon'],
                                  fname=self.name + '_xz',
                                  plot_save_dir=dir_plot,
                                  show=False,
                                  x_label='x',
                                  y_label='z', )

        LFPy_util.plot.morphology(data['poly_morph_y_z'],
                                  data['poly_morph_y_z_axon'],
                                  fname=self.name + '_yz',
                                  plot_save_dir=dir_plot,
                                  x_label='y',
                                  y_label='z',
                                  show=False, )
