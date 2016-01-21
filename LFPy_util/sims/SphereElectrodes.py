from Simulation import Simulation
import LFPy
import LFPy_util
import LFPy_util.data_extraction as de
import numpy as np

class SphereElectrodes(Simulation):
    """docstring for Grid"""

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.ID = "sphere"

        self.debug = False
        self.run_param['N'] = 100
        self.run_param['R'] = 100
        self.run_param['sigma'] = 0.3

    def __str__(self):
        return "Sphere Electrodes"

    def simulate(self,cell):
        data = self.data
        run_param = self.run_param
        # Calculate random numbers in a sphere.
        l = np.random.uniform(0,1,run_param['N'])
        u = np.random.uniform(-1,1,run_param['N'])
        phi = np.random.uniform(-1,1,run_param['N'])
        x = run_param['R']*np.power(l,1/3.0)*np.sqrt(1-u*u)*np.cos(phi),
        y = run_param['R']*np.power(l,1/3.0)*np.sqrt(1-u*u)*np.sin(phi),
        z = run_param['R']*np.power(l,1/3.0)*u

        self.cell.simulate(rec_imem=True)

        # Record the LFP of the electrodes. 
        electrode = LFPy.RecExtElectrode(self.cell,x,y,z,sigma=run_param['sigma'])
        electrode.calc_lfp()
        data['LFP'] = electrode.LFP
        data['elec_x'] = x
        data['elec_y'] = y
        data['elec_z'] = z

    def process_data(self):
        pass

    def plot(self,dir_plot):
        data = self.data
        import matplotlib.pyplot as plt
        plt.plot(data['elec_x'],data['elec_y'])
        plt.show()
