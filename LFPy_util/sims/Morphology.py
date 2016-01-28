from Simulation import Simulation
import LFPy_util
import LFPy_util.data_extraction as de


class Morphology(Simulation):
    """docstring for Grid"""

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.name = "morph"

        self.debug = False

        # Used by the custom plot function.
        self.show = False

        # Plot names.
        self.fname_morph_plot_xy = 'morph_xy'
        self.fname_morph_plot_xz = 'morph_xz'
        self.fname_morph_plot_yz = 'morph_yz'

    def __str__(self):
        return "Morphology"

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

        # Plot.
        LFPy_util.plot.morphology(data['poly_morph_x_y'],
                                  data['poly_morph_x_y_axon'],
                                  fname=self.fname_morph_plot_xy,
                                  plot_save_dir=dir_plot,
                                  show=self.show,
                                  mirror=True,
                                  x_label='y',
                                  y_label='x', )
        LFPy_util.plot.morphology(data['poly_morph_x_z'],
                                  data['poly_morph_x_z_axon'],
                                  fname=self.fname_morph_plot_xz,
                                  plot_save_dir=dir_plot,
                                  show=self.show,
                                  x_label='x',
                                  y_label='z', )
        LFPy_util.plot.morphology(data['poly_morph_y_z'],
                                  data['poly_morph_y_z_axon'],
                                  fname=self.fname_morph_plot_yz,
                                  plot_save_dir=dir_plot,
                                  x_label='y',
                                  y_label='z',
                                  show=self.show, )
