from Simulation import Simulation
import os
import LFPy
import LFPy_util
import LFPy_util.data_extraction as de
import LFPy_util.plot as lplot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Symmetry(Simulation):

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.ID = "sym"

        self.debug = False
        self.run_param['n'] = 11
        self.run_param['n_phi'] = 8
        self.run_param['theta'] = [22.5,45,67.5,90]
        self.run_param['R'] = 50
        self.run_param['sigma'] = 0.3

        self.amp_option = 'both'
        self.pre_dur = 16.7*0.5
        self.post_dur = 16.7*0.5

    def __str__(self):
        return "Symmetry"

    def simulate(self,cell):
        data = self.data
        run_param = self.run_param
        elec_x = np.empty([
            len(run_param['theta']),
            run_param['n_phi'],
            run_param['n'],
            ])
        elec_y = np.empty(elec_x.shape)
        elec_z = np.empty(elec_x.shape)
        for i,theta in enumerate(run_param['theta']):
            theta = theta*np.pi/180
            for j in xrange(run_param['n_phi']):
                phi = float(j)/run_param['n_phi']*np.pi*2
                y = run_param['R']*np.cos(theta)
                x = run_param['R']*np.sin(theta)*np.sin(phi)
                z = run_param['R']*np.sin(theta)*np.cos(phi)
                elec_x[i,j] = np.linspace(0,x,run_param['n'])
                elec_y[i,j] = np.linspace(0,y,run_param['n'])
                elec_z[i,j] = np.linspace(0,z,run_param['n'])
        data['elec_x'] = elec_x
        data['elec_y'] = elec_y
        data['elec_z'] = elec_z


    def process_data(self):
        data = self.data
        run_param = self.run_param

    def plot(self,dir_plot):
        data = self.data
        run_param = self.run_param
        format = 'pdf'
        # Set global matplotlib parameters.
        LFPy_util.plot.set_rc_param()
        # Create the directory if it does not exist.
        if not os.path.exists(dir_plot):
            os.makedirs(dir_plot)

        # 3D plot.
        fname = "sym_elec_pos"
        c = lplot.get_short_color_array(5)[2]
        fig = plt.figure(figsize=lplot.size_common)
        ax = fig.add_subplot(111,projection='3d')
        for i,theta in enumerate(run_param['theta']):
            for j in xrange(run_param['n_phi']):
                ax.scatter(data['elec_x'][i,j],data['elec_y'][i,j],data['elec_z'][i,j],c=c)
        ax.set_xlim(-run_param['R'],run_param['R'])
        ax.set_ylim(-run_param['R'],run_param['R'])
        ax.set_zlim(-run_param['R'],run_param['R'])
        # Save plt.
        path = os.path.join(dir_plot,fname+"."+format)
        plt.savefig(path,format=format,bbox_inches='tight')
        plt.close()
