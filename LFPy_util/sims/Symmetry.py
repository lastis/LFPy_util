import os
import numpy as np
import LFPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import LFPy_util
import LFPy_util.data_extraction as de
import LFPy_util.plot as lplot
from LFPy_util.sims.Simulation import Simulation


class Symmetry(Simulation):
    """
    Symmetry simulation
    """

    def __init__(self):
        Simulation.__init__(self)
        # Used by the super save and load function.
        self.name = "sym"

        self.debug = False
        self.run_param['n'] = 9
        self.run_param['n_phi'] = 8
        self.run_param['theta'] = [10, 50, 90, 130, 170]
        self.run_param['R'] = 50
        self.run_param['sigma'] = 0.3
        self.run_param['R_0'] = 10

        self.amp_option = 'both'
        self.pre_dur = 16.7 * 0.5
        self.post_dur = 16.7 * 0.5
        self.threshold = 3
        self.plot_detailed = False

    def __str__(self):
        return "Symmetry"

    def simulate(self, cell):
        # pylint: disable=invalid-name,no-member
        data = self.data
        run_param = self.run_param
        elec_x = np.empty([
            len(run_param['theta']),
            run_param['n_phi'],
            run_param['n'],
        ])
        elec_y = np.empty(elec_x.shape)
        elec_z = np.empty(elec_x.shape)
        for i, theta in enumerate(run_param['theta']):
            theta = theta * np.pi / 180
            for j in xrange(run_param['n_phi']):
                phi = float(j) / run_param['n_phi'] * np.pi * 2
                y1 = run_param['R'] * np.cos(theta)
                x1 = run_param['R'] * np.sin(theta) * np.sin(phi)
                z1 = run_param['R'] * np.sin(theta) * np.cos(phi)
                y0 = run_param['R_0'] * np.cos(theta)
                x0 = run_param['R_0'] * np.sin(theta) * np.sin(phi)
                z0 = run_param['R_0'] * np.sin(theta) * np.cos(phi)
                elec_x[i, j] = np.linspace(x0, x1, run_param['n'])
                elec_y[i, j] = np.linspace(y0, y1, run_param['n'])
                elec_z[i, j] = np.linspace(z0, z1, run_param['n'])
        x = elec_x.flatten()
        y = elec_y.flatten()
        z = elec_z.flatten()
        cell.simulate(rec_imem=True)

        # Record the LFP of the electrodes.
        electrode = LFPy.RecExtElectrode(cell,
                                         x=x,
                                         y=y,
                                         z=z,
                                         sigma=run_param['sigma'])
        electrode.calc_lfp()

        data['LFP'] = electrode.LFP
        data['elec_x'] = x
        data['elec_y'] = y
        data['elec_z'] = z
        data['t_vec'] = cell.tvec
        data['dt'] = cell.timeres_NEURON
        data['poly_morph'] \
                = de.get_polygons_no_axon(cell,['x','y'])
        data['poly_morph_axon'] \
                = de.get_polygons_axon(cell,['x','y'])

    def process_data(self):
        data = self.data
        run_param = self.run_param

        # Get the signal from the first electrode.
        signal = data['LFP'][0]
        spike, spikes_t_vec, I = de.extract_spikes(
            data['t_vec'],
            signal,
            pre_dur=self.pre_dur,
            post_dur=self.post_dur,
            threshold=self.threshold,
            amp_option=self.amp_option, )
        spikes = data['LFP'][:, I[0, 0]:I[0, 1]]

        amps_I = de.find_amplitude_type_I(spikes, amp_option=self.amp_option)
        amps_II = de.find_amplitude_type_II(spikes)
        widths_I, widths_I_trace = de.find_wave_width_type_I(spikes,
                                                             dt=data['dt'])
        widths_II, widths_II_trace = de.find_wave_width_type_II(
            spikes,
            dt=data['dt'],
            amp_option=self.amp_option)

        t = len(run_param['theta'])
        p = run_param['n_phi']
        n = run_param['n']

        amps_I = np.reshape(amps_I, (t, p, n))
        amps_II = np.reshape(amps_II, (t, p, n))
        widths_I = np.reshape(widths_I, (t, p, n))
        widths_II = np.reshape(widths_II, (t, p, n))

        # Becomes [t x n] matrix.
        amps_I_mean = np.mean(amps_I, 1)
        amps_I_std = np.std(amps_I, 1)
        amps_II_mean = np.mean(amps_II, 1)
        amps_II_std = np.std(amps_II, 1)
        widths_I_mean = np.mean(widths_I, 1)
        widths_I_std = np.std(widths_I, 1)
        widths_II_mean = np.mean(widths_II, 1)
        widths_II_std = np.std(widths_II, 1)

        data['amps_I_mean'] = amps_I_mean * 1000
        data['amps_I_std'] = amps_I_std * 1000
        data['amps_I'] = amps_I * 1000
        data['amps_II_mean'] = amps_II_mean * 1000
        data['amps_II_std'] = amps_II_std * 1000
        data['amps_II'] = amps_II * 1000

        data['widths_I_mean'] = widths_I_mean
        data['widths_I_std'] = widths_I_std
        data['widths_I'] = widths_I
        data['widths_I_trace'] = widths_I_trace * 1000

        data['widths_II_mean'] = widths_II_mean
        data['widths_II_std'] = widths_II_std
        data['widths_II'] = widths_II
        data['widths_II_trace'] = widths_II_trace * 1000

        data['spikes'] = spikes * 1000
        data['spikes_t_vec'] = spikes_t_vec
        data['r_vec'] = np.linspace(run_param['R_0'], run_param['R'],
                                    run_param['n'])

    def plot(self, dir_plot):
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
        ax = fig.add_subplot(111, projection='3d')
        cnt = 0
        for i, theta in enumerate(run_param['theta']):
            for j in xrange(run_param['n_phi']):
                for k in xrange(run_param['n']):
                    ax.scatter(data['elec_x'][cnt],
                               data['elec_y'][cnt],
                               data['elec_z'][cnt],
                               c=c)
                    cnt += 1
        ax.set_xlim(-run_param['R'], run_param['R'])
        ax.set_ylim(-run_param['R'], run_param['R'])
        ax.set_zlim(-run_param['R'], run_param['R'])
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        # Save plt.
        path = os.path.join(dir_plot, fname + "." + format)
        plt.savefig(path, format=format, bbox_inches='tight')
        plt.close()

        # New plot.
        fname = 'sym_spike_amps_I'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        c = lplot.get_short_color_array(data['amps_I_mean'].shape[0] + 1)
        for t in xrange(data['amps_I_mean'].shape[0]):
            label = r'$\theta = {}\degree$'.format(run_param['theta'][t])
            plt.plot(data['r_vec'],
                     data['amps_I_mean'][t],
                     color=c[t],
                     marker='o',
                     markersize=5,
                     label=label)
            ax.fill_between(data['r_vec'],
                            data['amps_I_mean'][t] - data['amps_I_std'][t],
                            data['amps_I_mean'][t] + data['amps_I_std'][t],
                            color=c[t],
                            alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        # Position the legen on the right side of the plot.
        ax.legend(handles,
                  labels,
                  loc='upper left',
                  bbox_to_anchor=(1.0, 1.04), )
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Distance from Soma")
        # Save plt.
        path = os.path.join(dir_plot, fname + "." + format)
        plt.savefig(path, format=format, bbox_inches='tight')
        plt.close()

        # New plot.
        fname = 'sym_spike_amps_II'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        c = lplot.get_short_color_array(data['amps_II_mean'].shape[0] + 1)
        for t in xrange(data['amps_II_mean'].shape[0]):
            label = r'$\theta = {}\degree$'.format(run_param['theta'][t])
            plt.plot(data['r_vec'],
                     data['amps_II_mean'][t],
                     color=c[t],
                     marker='o',
                     markersize=5,
                     label=label)
            ax.fill_between(data['r_vec'],
                            data['amps_II_mean'][t] - data['amps_II_std'][t],
                            data['amps_II_mean'][t] + data['amps_II_std'][t],
                            color=c[t],
                            alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        # Position the legen on the right side of the plot.
        ax.legend(handles,
                  labels,
                  loc='upper left',
                  bbox_to_anchor=(1.0, 1.04), )
        ax.set_ylabel("Amplitude")
        ax.set_xlabel("Distance from Soma")
        # Save plt.
        path = os.path.join(dir_plot, fname + "." + format)
        plt.savefig(path, format=format, bbox_inches='tight')
        plt.close()

        # New plot.
        fname = 'sym_spike_width_I'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        c = lplot.get_short_color_array(data['widths_I_mean'].shape[0] + 1)
        for t in xrange(data['widths_I_mean'].shape[0]):
            label = r'$\theta = {}\degree$'.format(run_param['theta'][t])
            plt.plot(data['r_vec'],
                     data['widths_I_mean'][t],
                     color=c[t],
                     marker='o',
                     markersize=5,
                     label=label)
            ax.fill_between(data['r_vec'],
                            data['widths_I_mean'][t] - data['widths_I_std'][t],
                            data['widths_I_mean'][t] + data['widths_I_std'][t],
                            color=c[t],
                            alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        # Position the legen on the right side of the plot.
        ax.legend(handles,
                  labels,
                  loc='upper left',
                  bbox_to_anchor=(1.0, 1.04), )
        ax.set_ylabel("Spike Width")
        ax.set_xlabel("Distance from Soma")
        # Save plt.
        path = os.path.join(dir_plot, fname + "." + format)
        plt.savefig(path, format=format, bbox_inches='tight')
        plt.close()

        # New plot.
        fname = 'sym_spike_width_II'
        print "plotting            :", fname
        plt.figure(figsize=lplot.size_common)
        ax = plt.gca()
        lplot.nice_axes(ax)
        # Plot
        c = lplot.get_short_color_array(data['widths_II_mean'].shape[0] + 1)
        for t in xrange(data['widths_II_mean'].shape[0]):
            label = r'$\theta = {}\degree$'.format(run_param['theta'][t])
            plt.plot(data['r_vec'],
                     data['widths_II_mean'][t],
                     color=c[t],
                     marker='o',
                     markersize=5,
                     label=label)
            ax.fill_between(
                data['r_vec'],
                data['widths_II_mean'][t] - data['widths_II_std'][t],
                data['widths_II_mean'][t] + data['widths_II_std'][t],
                color=c[t],
                alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        # Position the legen on the right side of the plot.
        ax.legend(handles,
                  labels,
                  loc='upper left',
                  bbox_to_anchor=(1.0, 1.04), )
        ax.set_ylabel("Spike Width")
        ax.set_xlabel("Distance from Soma")
        # Save plt.
        path = os.path.join(dir_plot, fname + "." + format)
        plt.savefig(path, format=format, bbox_inches='tight')
        plt.close()

        LFPy_util.plot.morphology(data['poly_morph'],
                                  data['poly_morph_axon'],
                                  elec_x=data['elec_x'],
                                  elec_y=data['elec_y'],
                                  fig_size=lplot.size_common,
                                  fname="sym_morph_elec",
                                  plot_save_dir=dir_plot,
                                  show=False)

        if self.plot_detailed:
            # Create the directory if it does not exist.
            sub_dir = os.path.join(dir_plot, "sym_detailed")
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            t = len(run_param['theta'])
            p = run_param['n_phi']
            n = run_param['n']

            cnt = 0
            for i in xrange(t):
                for j in xrange(p):
                    for k in xrange(n):
                        fname = 'sym_elec_t_{}_p_{}_n_{}'.format(
                            run_param['theta'][i], j * 360 / p, k)
                        print "plotting            :", fname
                        c = lplot.get_short_color_array(2 + 1)
                        plt.figure(figsize=lplot.size_common)
                        ax = plt.gca()
                        lplot.nice_axes(ax)
                        # Plot
                        plt.plot(data['spikes_t_vec'],
                                 data['spikes'][cnt],
                                 color=c[0])

                        # Trace I
                        trace_idx = np.where(~np.isnan(data['widths_I_trace'][
                            cnt]))[0]
                        trace_idx = [trace_idx[0], trace_idx[-1]]
                        plt.plot(data['spikes_t_vec'][trace_idx],
                                 data['widths_I_trace'][i][trace_idx],
                                 color=c[1],
                                 marker="|")
                        # Trace II
                        trace_idx = np.where(~np.isnan(data['widths_II_trace'][
                            cnt]))[0]
                        trace_idx = [trace_idx[0], trace_idx[-1]]
                        plt.plot(data['spikes_t_vec'],
                                 data['widths_II_trace'][cnt],
                                 color=c[1])
                        # Save plt.
                        path = os.path.join(sub_dir, fname + "." + format)
                        plt.savefig(path, format=format, bbox_inches='tight')
                        plt.close()
                        cnt += 1
