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

        # # Plot morph no grid {{{1 #
        # fname = self.name + '_xy_nogrid'
        # lplot.plot_format = ['png']
        # print "plotting            :", fname
        # plt.figure()
        # ax = plt.gca()
        # zips = []
        # xmin, xmax, ymin, ymax = 0, 0, 0, 0
        # for a, b in data['poly_morph_x_y']:
        #     xmin = min(xmin, min(a))
        #     xmax = max(xmax, max(a))
        #     ymin = min(ymin, min(b))
        #     ymax = max(ymax, max(b))
        #     zips.append(zip(a, b))
        # polycol = mpl.collections.PolyCollection(zips,
        #                                          edgecolors='none',
        #                                          facecolors=cm.get_color(0.0))
        # ax.add_collection(polycol, )
        # plt.axis('equal')
        # plt.axis([xmin, xmax, ymin, ymax])
        # # Create the directory if it does not exist.
        # if not os.path.exists(dir_plot):
        #     os.makedirs(dir_plot)
        # # Save.
        # name = fname + '.png'
        # path = os.path.join(dir_plot, name)
        # plt.savefig(path, format='png', bbox_inches='tight',
        #             transparent=True, dpi=900)
        # plt.close()
        # # Reset to pdf format.
        # lplot.plot_format = ['pdf']
        # # 1}}} #
        # Plot.
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
