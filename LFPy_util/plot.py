import numpy as np
import scipy.interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.animation as animation
import LFPy_util.colormaps as cmaps
import os

plot_format = ['pdf']
color_array_long = cmaps._viridis_data

# Some plot sizes that fit well with A4 papers.
size_square_small = [3,3]
size_square = [4,4]
size_common = [8,4]
size_large = [8,8]
size_thin = [8,2]

def set_rc_param():
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.style'] = 'normal'
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{bm,upgreek,textcomp,gensymb}']
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['font.serif'] = ['Times']
    # mpl.rcParams['text.latex.unicode']=False

def get_short_color_array(n) :
    """
    Create a linspace of the colors in the default collor array. Currently
    viridis.

    :param int n: 
        Returned array length.
    :returns: 
        :class:`list` -- (n x 3) color array.
    """
    values = range(n)
    cNorm = colors.Normalize(vmin=0,vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmaps.viridis)
    color_arr = []
    for i in xrange(n):
        colorVal = scalarMap.to_rgba(values[i])
        color_arr.append(colorVal)
    return color_arr

def nice_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid()

def _get_line_segments(t_vec,width_trace_1d):
    # Get linesegments from the trace.
    lines_y = []
    lines_x = []
    prev = False
    tmp_y = [0,0]
    tmp_x = [0,0]
    for i in xrange(len(width_trace_1d)):
        if np.isnan(width_trace_1d[i]):
            cur = False
        else:
            cur = True
        if prev == False and cur == True:
            tmp_y[0] = width_trace_1d[i]
            tmp_x[0] = t_vec[i]
        if prev == True and cur == False:
            tmp_y[1] = width_trace_1d[i-1]
            tmp_x[1] = t_vec[i-1]
            lines_y.append(tmp_y[:])
            lines_x.append(tmp_x[:])
        prev = cur
    if cur == True:
        tmp_y[1] = width_trace_1d[len(width_trace_1d)-1]
        tmp_x[1] = t_vec[len(width_trace_1d)-1]
        lines_y.append(tmp_y)
        lines_x.append(tmp_x)
    lines_y = np.array(lines_y)
    lines_x = np.array(lines_x)
    return lines_x,lines_y


def soma(t_vec, signal,fname=None, plot_save_dir='.', show=True, 
        width_trace=None) :
    """
    Plots membrane potential. 

    :param `~numpy.ndarray` tvec: 
        Time vector with length equal **signal**. 
    :param `~numpy.ndarray` signal: 
        Signal to plot.
    :param string fname: 
        File name to save plot as. Format is specified by the module. 
    :param string plot_save_dir: 
        Output directiory.
    :param bool show: 
        Show plot. 
    """

    print "plotting            :", fname

    signal = np.array(signal)
    t_vec = np.array(t_vec)

    set_rc_param()
    fig = plt.figure(figsize=size_common)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid()
    ax.set_ylabel(r'Membrane Potential \textbf[$\mathbf{mV}$\textbf]')
    ax.set_xlabel(r'Time \textbf[$\mathbf{ms}$\textbf]')

    plt.plot(t_vec,signal, color=color_array_long[0])
    if width_trace is not None:
        # Get linesegments from the trace.
        lines_x,lines_y = _get_line_segments(t_vec, width_trace)

        color_idx = int(len(color_array_long)*0.7)
        color = color_array_long[color_idx]
        plt.hold('on')
        for i in xrange(lines_y.shape[0]):
            text = r"{} ms".format((lines_x[i,1]-lines_x[i,0]))
            ax.annotate(
                text,
                xy=(lines_x[i,1],lines_y[i,1]), 
                xycoords='data',
                xytext=(20, 20), 
                textcoords='offset points',
                va="center", 
                ha="left",
                bbox=dict(boxstyle="round4", fc="w"),
                arrowprops=dict(arrowstyle="-|>",
                              connectionstyle="arc3,rad=-0.2",
                              fc="w"), 
                )
            plt.plot(lines_x[i],lines_y[i],
                linestyle='-',
                linewidth=2,
                marker='|', 
                markersize=4,
                color=color,
                solid_capstyle='butt'
            )
        plt.hold('off')

    if (fname is not None):
        # Create the directory if it does not exist.
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        # Create different versions of the file. 
        os.chdir(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            plt.savefig(name,format=format_str,bbox_inches='tight')
    if show :
        plt.show()
    print 'finished            :', fname

def frequency_amp(firing_rate, amp, relative=False,
        fname=None, show=True, plot_save_dir='.'):
    """
    """
    print "plotting            :", fname

    set_rc_param()
    fig = plt.figure(figsize=size_common)

    if relative:
        # Find the threshold.
        if firing_rate[0] != 0:
            print 'Warning: Cannot find threshold. Continuing without.'
        else:
            for i in xrange(len(amp)):
                if firing_rate[i] != 0:
                    threshold = amp[i-1]
                    break
            amp = amp/threshold
    plt.plot(amp,firing_rate, color=color_array_long[0])

    if (fname is not None):
        # Create the directory if it does not exist.
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        # Create different versions of the file. 
        os.chdir(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            plt.savefig(
                    name,
                    format=format_str,
                    transparent=False, 
                    bbox_inches='tight',
                    pad_inches=0.1
            )
    if show :
        plt.show()
    plt.close()
    print 'finished            :', fname

def i_mem_v_mem(v_vec, i_vec, t_vec,
        fname=None, show=True, plot_save_dir='.'):
    """
    """
    print "plotting            :", fname

    set_rc_param()
    fig = plt.figure(figsize=size_common)
    ax = plt.subplot(2,1,1)
    ax.grid()
    plt.plot(t_vec,v_vec, color=color_array_long[0])

    ax = plt.subplot(2,1,2)
    ax.grid()
    plt.plot(t_vec,i_vec, color=color_array_long[0])

    if (fname is not None):
        # Create the directory if it does not exist.
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        # Create different versions of the file. 
        os.chdir(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            plt.savefig(
                    name,
                    format=format_str,
                    transparent=False, 
                    bbox_inches='tight',
                    pad_inches=0.1
            )
    if show :
        plt.show()
    plt.close()
    print 'finished            :', fname

def morphology(poly_morph, poly_morph_axon=None, elec_x=None, elec_y=None, mirror=False, 
        fig_size='common',x_label='x', y_label='y',
        fname=None, show=True, plot_save_dir=None):
    """
    """
    print "plotting            :", fname
    if elec_x is not None:
        elec_x = np.array(elec_x)
    if elec_y is not None:
        elec_y = np.array(elec_y)

    set_rc_param()
    if fig_size == 'square_small':
        figsize = size_square_small
    elif fig_size == 'square':
        figsize = size_square
    else:
        figsize = size_common
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0

    colors = get_short_color_array(3)
    # Plot morphology.
    zips = []
    for a,b in poly_morph:
        if mirror:
            tmp = b
            b = a
            a = tmp
        xmin = min(xmin,min(a))
        xmax = max(xmax,max(a))
        ymin = min(ymin,min(b))
        ymax = max(ymax,max(b))
        zips.append(zip(a,b))
    polycol = mpl.collections.PolyCollection(
            zips,
            edgecolors='none',
            facecolors=colors[0]
    )
    ax.add_collection(polycol,)
    
    # Plot second morpholgy in different color.
    if poly_morph_axon is not None:
        zips = []
        for a,b in poly_morph_axon:
            if mirror:
                tmp = b
                b = a
                a = tmp
            zips.append(zip(a,b))
        polycol_a = mpl.collections.PolyCollection(
                zips,
                edgecolors='none',
                facecolors=colors[1]
        )
        ax.add_collection(polycol_a,)

    plt.axis('equal')
    ax.grid()

    if elec_x is not None and elec_y is not None:
        # Change the electrode positions if mirrored.
        if mirror:
            tmp = elec_y
            elec_y = elec_x
            elec_x = tmp
        # Plot small x markers for each electrode.
        # Set axis limits.
        # The morphology is often much bigger than the position of the electrodes.
        plt.scatter(elec_x, elec_y, marker='x',color='black',linewidth=0.2)
        plt.axis([elec_x.min(),elec_x.max(),elec_y.min(),elec_y.max()])
        # ax.set_ylim([elec_y.min(),elec_y.max()])
        # ax.set_xlim([elec_x.min(),elec_x.max()])


    # Add axis label.
    ax.set_xlabel(x_label + '\n' + 
            r'Distance from Soma \textbf[$\mathbf{\bm\upmu m}$\textbf]')
    ax.set_ylabel(y_label)

    if (fname is not None):
        # Create the directory if it does not exist.
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        # Create different versions of the file. 
        os.chdir(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            plt.savefig(
                    name,
                    format=format_str,
                    transparent=False, 
                    bbox_inches='tight',
                    pad_inches=0.1
            )
    if show :
        plt.show()
    plt.close()
    print 'finished            :', fname

def scattered_i_mem_v_mem(v_vec_list, i_vec_list, t_vec, rec_x, rec_y, 
        poly_morph,
        fname=None, show=True, plot_save_dir=None):
    """
    Plot i_mem and v_mem at scattered points along a neuron. 
    
    :param `~numpy.ndarray` t_vec: 
        Time vector with length equal **signal**. 
    :param `~numpy.ndarray` LFP: 
        (n_electrodes x time). Output from LFPy. 
    :param string mode: 
        'all'. Only one option available.
    :param `~numpy.ndarray` elec_pos: 
        Optional list of numbers of that will be the label of each graph. 
        Used for showing radial distance dependencies of LFP. Length must equal
        **t_vec**.
    :param string fname: 
        File name to save plot as. Format is specified by the module. 
    :param string plot_save_dir: 
        Output directiory.
    :param bool show: 
        Show plot. 

    Example:
        .. code-block:: python

            # Create a vector of the radial position of the electrodes.
            elec_pos = np.linspace(r_0,r,n)
            # Get the signals from the first electrodes radilly away from x_0
            # and plot them.
            LFPy_util.plot.electrodeSignals(
                    t_vec,
                    LFP,
                    elec_pos=elec_pos,
            )
    """
    print "plotting            :", fname

    mirror = True

    # Remove data points before 0. 
    idx = (np.abs(t_vec-0)).argmin() + 1
    t_vec = np.delete(t_vec,range(idx))
    v_vec_list = np.delete(v_vec_list, range(idx), 1)
    i_vec_list = np.delete(i_vec_list, range(idx), 1)

    set_rc_param()
    fig = plt.figure(figsize=size_common)
    ax = plt.subplot(2,1,1)
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    # Plot morphology.
    zips = []
    for a,b in poly_morph:
        if mirror:
            tmp = b
            b = a
            a = tmp
        xmin = min(xmin,min(a))
        xmax = max(xmax,max(a))
        ymin = min(ymin,min(b))
        ymax = max(ymax,max(b))
        zips.append(zip(a,b))
    polycol = mpl.collections.PolyCollection(zips,edgecolors='none',facecolors='black')
    ax.add_collection(polycol,)
    colors = get_short_color_array(len(rec_x))
    if mirror:
        tmp = rec_y
        rec_y = rec_x
        rec_x = tmp
    plt.scatter(rec_x, rec_y, marker='o',color=colors,linewidth=0.2)
    ax.set_ylim([ymin,ymax])
    ax.set_xlim([xmin,xmax])

    ax = plt.subplot(2,2,3)
    for i in xrange(len(v_vec_list)):
        plt.plot(t_vec,v_vec_list[i],color=colors[i])

    ax = plt.subplot(2,2,4)
    for i in xrange(len(i_vec_list)):
        plt.plot(t_vec,i_vec_list[i],color=colors[i])

    if (fname is not None):
        # Create the directory if it does not exist.
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        # Create different versions of the file. 
        os.chdir(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            plt.savefig(
                    name,
                    format=format_str,
                    transparent=False, 
                    bbox_inches='tight',
                    pad_inches=0.1
            )
    if show :
        plt.show()
    plt.close()
    print 'finished            :', fname

def hide_spines():
    """Hides the top and rightmost axis spines from view for all active
    figures and their respective axes."""

    # Retrieve a list of all current figures.
    figures = [x for x in mpl._pylab_helpers.Gcf.get_all_fig_managers()]
    for figure in figures:
        # Get all Axis instances related to the figure.
        for ax in figure.canvas.figure.get_axes():
            # Disable spines.
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            # Disable ticks.
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

def fourierSpecter(freqs, amps, f_end=None,
        fname=None, plot_save_dir='.', show=True) :
    """
    Plot fourier frequencies with their amplitudes. 

    :param `~numpy.ndarray` freqs: 
        Frequencies, length equal **amps**. 
    :param `~numpy.ndarray` amps: 
        Amplitudes. 
    :param float f_end: 
        Frequency to end plot at.
    :param string fname: 
        File name to save plot as. Format is specified by the module. 
    :param string plot_save_dir: 
        Output directiory.
    :param bool show: 
        Show plot. 
    """

    print "plotting            :", fname

    freqs = np.asarray(freqs)

    set_rc_param()
    fig = plt.figure(figsize=size_common)
    ax = plt.gca()
    ax.grid()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylabel(r'Amplitude \textbf[$\mathbf{mV}$\textbf]')
    ax.set_xlabel(r'Frequency \textbf[$\mathbf{kHz}$\textbf]')

    if f_end is not None:
        idx = min(range(len(freqs)), key=lambda i: abs(freqs[i]-f_end))
        freqs = freqs[0:idx]
        amps = amps[0:idx]

    plt.plot(freqs,amps, color=color_array_long[0])
    ax.set_xlim([-0.05,freqs[-1]])

    if (plot_save_dir is not None):
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            path = os.path.join(plot_save_dir,name)
            plt.savefig(path,format=format_str,bbox_inches='tight')
    if show :
        plt.show()
    print 'finished            :', fname

def signal_sub_plot_3(t_vec, v_mat, width_trace=None,
        fname=None, plot_save_dir='.', show=True):
    t_vec = np.array(t_vec)
    v_mat = np.array(v_mat)
    if width_trace is not None:
        width_trace = np.array(width_trace)
    if v_mat.shape[0] != 3:
        print 'Warning: Three signals should be supplied. Returning.'
        return

    print "plotting            :", fname

    # Use microvolt if the values are small.
    mili_volt = True
    if np.fabs(v_mat).max() < 1:
        mili_volt = False
        v_mat *= 1000

    # Set the axis labels.
    ylabel = r'Potential '
    if mili_volt :
        ylabel += r'\textbf[$\mathbf{mV}$\textbf]'
    else : 
        ylabel += r'\textbf[$\mathbf{\bm\upmu V}$\textbf]'
    
    # Add packages for latex commands.
    set_rc_param()
    fig = plt.figure(figsize=size_thin)

    # Get the viridis color.
    for i in xrange(v_mat.shape[0]):
        # Configure the axes.
        ax = plt.subplot(1,3,i+1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.grid()
        # Set the axis labels.
        if i == 0:
            ax.set_ylabel(ylabel)
        if i == 1:
            ax.set_xlabel(r'Time \textbf[$\mathbf{ms}$\textbf]')

        signal = v_mat[i,:]
        ax.plot(t_vec,signal,color=color_array_long[0])
        if width_trace is not None:
            # Get linesegments from the trace.
            lines_x,lines_y = _get_line_segments(t_vec,width_trace[i])

            if not mili_volt:
                lines_y *= 1000

            color_idx = int(len(color_array_long)*0.5)
            color = color_array_long[color_idx]
            for j in xrange(lines_y.shape[0]):
                plt.plot(lines_x[j],lines_y[j],
                    linestyle='-',
                    linewidth=2,
                    marker='|', 
                    markersize=4,
                    color=color,
                    solid_capstyle='butt'
                )
            # Only annotate the last bar.
            width_length = np.diff(lines_x).sum()
            text = r"{0:.3f} ms".format(width_length)
            ax.annotate(
                text,
                xy=(lines_x[j,-1],lines_y[j,-1]), 
                xycoords='data',
                xytext=(20, 20), 
                textcoords='offset points',
                va="center", 
                ha="left",
                bbox=dict(boxstyle="round4", fc="w"),
                arrowprops=dict(arrowstyle="-|>",
                              connectionstyle="arc3,rad=-0.2",
                              fc="w"), 
                )

    # plt.tight_layout()
    if (fname is not None):
        # Create the directory if it does not exist.
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        # Create different versions of the file. 
        os.chdir(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            plt.savefig(
                    name,
                    format=format_str,
                    transparent=False, 
                    bbox_inches='tight',
                    pad_inches=0.1
            )
    if show :
        plt.show()
    plt.close()
    print 'finished            :', fname

def electrodeSignals(t_vec, LFP, mode='all', elec_pos=[],
        fname=None, plot_save_dir='.', show=True):
    """
    Plot a matrix of signals in the same plot. 
    
    Shows one graph for each signal.

    :param `~numpy.ndarray` t_vec: 
        Time vector with length equal **signal**. 
    :param `~numpy.ndarray` LFP: 
        (n_electrodes x time). Output from LFPy. 
    :param string mode: 
        'all'. Only one option available.
    :param `~numpy.ndarray` elec_pos: 
        Optional list of numbers of that will be the label of each graph. 
        Used for showing radial distance dependencies of LFP. Length must equal
        **t_vec**.
    :param string fname: 
        File name to save plot as. Format is specified by the module. 
    :param string plot_save_dir: 
        Output directiory.
    :param bool show: 
        Show plot. 

    Example:
        .. code-block:: python

            # Create a vector of the radial position of the electrodes.
            elec_pos = np.linspace(r_0,r,n)
            # Get the signals from the first electrodes radilly away from x_0
            # and plot them.
            LFPy_util.plot.electrodeSignals(
                    t_vec,
                    LFP,
                    elec_pos=elec_pos,
            )
    """
    t_vec = np.array(t_vec)
    LFP = np.array(LFP)
    elec_pos = np.array(elec_pos)

    print "plotting            :", fname

    # Use microvolt if the values are small.
    mili_volt = True
    if np.fabs(LFP).max() < 1:
        mili_volt = False
        LFP *= 1000
    
    # Add packages for latex commands.
    set_rc_param()
    fig = plt.figure(figsize=size_common)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid()

    # Set the axis labels.
    ylabel = r'Potential '
    if mili_volt :
        ylabel += r'\textbf[$\mathbf{mV}$\textbf]'
    else : 
        ylabel += r'\textbf[$\mathbf{\bm\upmu V}$\textbf]'
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r'Time \textbf[$\mathbf{ms}$\textbf]')

    # Plot all rows in the same plot.
    if mode == 'all':
        # Get the viridis color map.
        color_array = get_short_color_array(LFP.shape[0])
        for i in xrange(LFP.shape[0]):
            signal = LFP[i,:]
            # Add info about the position about the electrodes
            label = ''
            if len(elec_pos) != 0:
                label = r'r = %.2f $\mathbf{\bm\upmu m}$' %(elec_pos[i])
            ax.plot(t_vec,signal,color=color_array[i],label=label)
        # Get the labels and plot the legend.
        if len(elec_pos) != 0 :
            handles,labels = ax.get_legend_handles_labels()
            # Position the legen on the right side of the plot.
            ncol = elec_pos.shape[0]/17 + 1
            lgd = ax.legend(
                    handles,
                    labels,
                    # loc='center left',
                    loc='upper left',
                    bbox_to_anchor=(1.0,1.0),
                    ncol=ncol,
            )

    # plt.tight_layout()
    if (fname is not None):
        # Create the directory if it does not exist.
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        # Create different versions of the file. 
        os.chdir(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            plt.savefig(
                    name,
                    format=format_str,
                    transparent=False, 
                    bbox_inches='tight',
                    pad_inches=0.1
            )
    if show :
        plt.show()
    plt.close()
    print 'finished            :', fname

def circularLfp(LFP, x, y, neuron_x=[], neuron_y=[], poly_morph=[], res_x=100, res_y=100, 
        fname=None, plot_save_dir='.', show=True) :
    """
    Plot a 2d intensity image of scattered electrodes. 

    Output is a square image where area between electrodes have been 
    interpolated. Can also show the neuron morphology on top of the image. 
    
    Either shows one graph for 
    each signal or the mean +/- the standard deviation. 

    :param `~numpy.ndarray` LFP: 
        (n_electrodes x time). Output from LFPy. 
    :param `~numpy.ndarray` x: 
        x position of electrodes. 
    :param `~numpy.ndarray` y: 
        y position of electrodes. 
    :param `~numpy.ndarray` neuron_x: 
        List of neuron positions. Neuron is displayed using thin lines. 
    :param `~numpy.ndarray` neuron_y: 
        List of neuron positions. Neuron is displayed using thin lines. 
    :param list poly_morph: 
        Draw the neuron using polygons.
    :param int res_x: 
        Resolution of output image.
    :param int res_y: 
        Resolution of output image.
    :param string fname: 
        File name to save plot as. Format is specified by the module. 
    :param string plot_save_dir: 
        Output directiory.
    :param bool show: 
        Show plot. 

    Example:
        .. code-block:: python

            # Create circular electrodes.
            electrode_dict = LFPy_util.electrodes.circularElectrodesXZ(
                    r = r,
                    n = n,
                    n_theta = n_theta,
                    r_0 = r_0,
                    x_0=[0,0,0]
            ) 
            electrode_dict['sigma'] = 0.3
            # Record the LFP of the electrodes. 
            electrode = LFPy.RecExtElectrode(cell, **electrode_dict)
            electrode.calc_lfp()

            LFP = electrode.LFP
            x = electrode_dict['x']
            z = electrode_dict['z']
            neuron_x = cell.x3d
            neuron_z = cell.z3d
            poly_morph = cell.get_idx_polygons(('x','z'))

            LFPy_util.plot.circularLfp(LFP,x,z,
                    poly_morph=poly_morph,
                    fname='Plot1'
                    show=False,
            )
            LFPy_util.plot.circularLfp(LFP,x,z,
                    neuron_x=neuron_x,
                    neuron_y=neuron_z,
                    fname='Plot2'
                    show=False,
            )

    """
    LFP = np.array(LFP)
    x = np.array(x)
    y = np.array(y)

    print "plotting            :", fname

    # Use microvolt if the values are small.
    mili_volt = True
    if np.fabs(LFP).max() < 1:
        mili_volt = False
        LFP *= 1000

    # Get the maximum value along the time axis. 
    z = np.amax(np.fabs(LFP),axis=1)
    z = np.ma.log10(z)

    xi = np.linspace(x.min(),x.max(),res_x)
    yi = np.linspace(y.min(),y.max(),res_y)
    xi,yi = np.meshgrid(xi,yi)

    # Interpolate to create a image. 
    rbf = scipy.interpolate.Rbf(x,y,z, function='linear')
    zi = rbf(xi,yi)
    # xy = np.column_stack((x,y))
    # zi = scipy.interpolate.griddata(xy,z,(xi,yi),method='cubic')

    fig = plt.figure()
    ax = plt.gca()
    # Add packages for latex commands.
    set_rc_param()

    # Plot the neuron morphology. Can either supply lines or the polygon.
    for i in xrange(len(neuron_x)):
        ax.plot(neuron_x[i],neuron_y[i],color='black',linewidth=0.2)
    # Use the polygons.
    if len(poly_morph) > 0:
        zips = []
        for a,b in poly_morph:
            zips.append(zip(a,b))
        polycol = mpl.collections.PolyCollection(zips,edgecolors='none',facecolors='black')
        ax.add_collection(polycol,)

    # Plot the interpolated image.
    plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
            extent=[x.min(), x.max(), y.min(), y.max()],cmap=cmaps.viridis)

    # Create a colorbar.
    cb = plt.colorbar()
    cb.locator = mpl.ticker.LinearLocator(numticks=5)
    cb.update_ticks()
    # Create the label for the colorbar.
    cb_label = r'Max. Potential '
    if mili_volt:
        cb_label += r'\textbf[$\mathbf{mV}$\textbf]'
    else:
        cb_label += r'\textbf[$\mathbf{\bm\upmu V}$\textbf]'
    cb.set_label(cb_label,
        rotation=0, y=1.07, labelpad=-50)

    # Plot small x markers for each electrode.
    plt.scatter(x, y, marker='x',color='black',linewidth=0.2)

    # Set axis limits.
    # The morphology is often much bigger than the position of the electrodes.
    ax.set_ylim([y.min(),y.max()])
    ax.set_xlim([x.min(),x.max()])

    # Add axis label.
    ax.set_xlabel(r'Distance from Soma \textbf[$\mathbf{\bm\upmu m}$\textbf]')

    if (fname is not None):
        # Create the directory if it does not exist.
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        # Create different versions of the file. 
        os.chdir(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            plt.savefig(name,format=format_str,bbox_inches='tight')
    if show :
        plt.show()
    plt.close()
    print 'finished            :', fname

    

def gradient2D(LFP, x, y, neuron_x = [], neuron_y = [], soma_diam=0, 
        exclude_soma=True, res_x=100, res_y=100, animate=False,
        fname=None, show=True, plot_save_dir=None):
    """
    Plot a 2d intensity image of scattered electrodes. 

    fname must be set if **animation** is selected. Animation uses HTML writer
    from JSAnimation.

    :param `~numpy.ndarray` LFP: 
        (n_electrodes x time). Output from LFPy. 
    :param `~numpy.ndarray` x: 
        x position of electrodes. 
    :param `~numpy.ndarray` y: 
        y position of electrodes. 
    :param `~numpy.ndarray` neuron_x: 
        List of neuron positions. Neuron is displayed using thin lines. 
    :param `~numpy.ndarray` neuron_y: 
        List of neuron positions. Neuron is displayed using thin lines. 
    :param int res_x: 
        Resolution of output image.
    :param int res_y: 
        Resolution of output image.
    :param bool animate: 
        If True, will make an image for each timestep.
    :param string fname: 
        File name to save plot as. Format is specified by the module. 
    :param string plot_save_dir: 
        Output directiory.
    :param bool show: 
        Show plot. 

    """

    print "plotting            :", fname

    LFP = np.asarray(LFP)
    x = np.asarray(x)
    y = np.asarray(y)

    nx = len(x)
    ny = len(y)
    n = nx*ny
    X,Y = np.meshgrid(x,y)
    X = X.flatten()
    Y = Y.flatten()

    if exclude_soma == True and soma_diam != 0:
        for i in xrange(n):
            i = n - 1 - i
            a = X[i];
            b = Y[i];
            c = np.sqrt(a*a+b*b)
            if c < soma_diam :
                LFP = np.delete(LFP,i,axis=0)
                X = np.delete(X,i)
                Y = np.delete(Y,i)

    fig = plt.figure()
    ax = plt.gca()
    # Add packages for latex commands.
    set_rc_param()

    # Make data points to calculate the interpolation on. 
    a = np.linspace(x.min(),x.max(),res_x)
    b = np.linspace(y.min(),y.max(),res_y)
    xi,yi = np.meshgrid(a,b)

    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()

    # Plot the neuron morphology. 
    for i in xrange(len(neuron_x)):
        ax.plot(neuron_x[i],neuron_y[i],color='black',linewidth=0.2)

    # Plot a circle around the soma.
    if soma_diam != 0 :
        circle = plt.Circle((0,0),soma_diam,color='black',zorder=10)
        ax.add_artist(circle)

    # Set axis limits.
    # The morphology is often much bigger than the position of the electrodes.
    ax.set_ylim([y_min,y_max])
    ax.set_xlim([x_min,x_max])
    ax.set_xlabel(r'Distance from Soma \textbf[$\mathbf{\bm\upmu m}$\textbf]')
    if animate:
        from JSAnimation import HTMLWriter
        ims = []
        LFP = np.fabs(LFP)
        LFP = np.ma.log10(LFP)
        LFP_min = np.min(LFP)
        LFP_max = np.max(LFP)
        print LFP_min
        print LFP_max

        XY = np.column_stack((X,Y))
        for i in xrange(LFP.shape[1]) :
            Z = LFP[:,i]

            # # Create an interpolation function and calculate the image zi. 
            # rbf = scipy.interpolate.Rbf(X,Y,Z, function='linear')
            # zi = rbf(xi,yi)

            zi = scipy.interpolate.griddata(XY,Z,(xi,yi),method='cubic')

            # im = plt.imshow(zi,origin='lower',
            im = plt.imshow(zi, vmin=LFP_min, vmax=LFP_max, origin='lower',
                    extent=[x_min, x_max, y_min, y_max],
                    cmap=cmaps.viridis,zorder=-1)
            ims.append([im])

        # Add colorbar.
        cb = plt.colorbar()
        cb.set_label(r'Log$_{10}$ Potential \textbf[$\mathbf{mV}$\textbf]',
            rotation=0, y=1.07, labelpad=-50)
        cb.locator = mpl.ticker.LinearLocator(numticks=5)
        cb.update_ticks()

        # Show electrode positions.
        plt.scatter(X, Y, marker='x',color='black',linewidth=0.2,zorder=10)

        print 'Making animation'
        ani = animation.ArtistAnimation(fig, ims, interval=40, blit=True,
                repeat_delay=500)

        plt.tight_layout()
        if (fname is not None):
            # Create the directory if it does not exist.
            if not os.path.exists(plot_save_dir):
                os.makedirs(plot_save_dir)
            name = fname+'.html'
            os.chdir(plot_save_dir)
            
            ani.save(name,
                    writer=HTMLWriter(embed_frames=False),
            )

            name = fname+'.mp4'
            ani.save(name,
                    writer='mencoder',
            )
    else :
        Z = np.amax(np.absolute(LFP),axis=1)
        Z = np.ma.log10(Z)

        rbf = scipy.interpolate.Rbf(X,Y,Z, function='linear')
        zi = rbf(xi,yi)

        plt.imshow(zi, origin='lower',
                extent=[x.min(), x.max(), y.min(), y.max()],
                cmap=cmaps.viridis,zorder=-1)

        # Add colorbar.
        cb = plt.colorbar()
        cb.set_label(r'Log$_{10}$ Amplitude \textbf[$\mathbf{mV}$\textbf]',
            rotation=0, y=1.07, labelpad=-40)
        cb.locator = mpl.ticker.LinearLocator(numticks=5)
        cb.update_ticks()

        # Show electrode positions.
        plt.scatter(X, Y, marker='x',color='black',linewidth=0.2,zorder=10)

        plt.tight_layout()
        if (fname is not None):
            # Create the directory if it does not exist.
            if not os.path.exists(plot_save_dir):
                os.makedirs(plot_save_dir)
            # Create different versions of the file. 
            os.chdir(plot_save_dir)
            for format_str in plot_format:
                name = fname+'.'+format_str
                plt.savefig(
                        name,
                        format=format_str,
                        transparent=False, 
                        bbox_inches='tight',
                        pad_inches=0
                )
    if show :
        plt.show()
    plt.close()
    print 'finished            :', fname

def signals2D(LFP, x, y, neuron_x = [], neuron_y = [], soma_diam=0, 
        exclude_soma=True, normalization=True, poly_morph=[], 
        amp_option='both', figsize=size_common,
        fname=None, show=True, plot_save_dir=None ):

    print "plotting            :", fname

    LFP = np.asarray(LFP)
    x = np.asarray(x)
    y = np.asarray(y)

    nx = len(x)
    ny = len(y)
    if nx == 0 or ny == 0:
        raise ValueError('Array empty.')

    n = nx*ny
    xi,yi = np.meshgrid(x,y)
    xi = xi.flatten()
    yi = yi.flatten()

    if exclude_soma == True and soma_diam != 0:
        for i in xrange(n):
            i = n - 1 - i
            a = xi[i];
            b = yi[i];
            c = np.sqrt(a*a+b*b)
            if c < soma_diam :
                LFP = np.delete(LFP,i,axis=0)
                xi = np.delete(xi,i)
                yi = np.delete(yi,i)

    if amp_option == 'neg' :
        amp = np.absolute(np.amin(LFP,axis=1))
    elif amp_option == 'pos' :
        amp = np.amax(LFP,axis=1)
    elif amp_option == 'both' :
        amp = np.amax(np.absolute(LFP),axis=1)
    if len(amp) != nx*ny:
        raise ValueError('Array mismatch.')


    amp_max = amp.max()

    # Create a function that maps an amplitude between min and max to a color
    # using scalarMap.to_rgba(val).
    cmap = cmaps.viridis
    cNorm = colors.Normalize(vmin=amp.min(),vmax=amp.max())
    scalarMap = cmx.ScalarMappable(norm=cNorm,cmap=cmap)

    # Use plt.scatter only to create a colorbar
    fig = plt.figure(figsize=figsize)
    sc = plt.scatter(xi,yi,c=amp,vmin=amp.min(),vmax=amp.max(), cmap=cmap)
    plt.clf()

    # Add packages for latex commands.
    set_rc_param()

    # Create colorbar from the scatterplot that was removed.
    cb = plt.colorbar(sc)
    cb.set_label(r'Abs. Amplitude \textbf[$\mathbf{mV}$\textbf]',
            rotation=0, y=1.07, labelpad=-40)
    cb.locator = mpl.ticker.LinearLocator(numticks=5)
    cb.update_ticks()

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlabel(r'Distance from Soma \textbf[$\mathbf{\bm\upmu m}$\textbf]')


    # Calculate a width and height of the signals in the plot.
    signal_size_x = (x.max()-x.min())/nx * 0.75
    signal_size_y = (y.max()-y.min())/ny * 0.75/2

    # Calculate the x-points of the signals. Resized and moved so the midpoint
    # is at the electrode positions.
    time_a = np.linspace(0,1,LFP.shape[1]) 
    time_a = time_a * signal_size_x
    time_a = time_a - signal_size_x/2

    # Calculate new dimensions of the image. The signals will go the left and 
    # right side of each electrode.
    xmin = x.min()-signal_size_x
    xmax = x.max()+signal_size_x
    ymin = y.min()-signal_size_y
    ymax = y.max()+signal_size_y
    ax.set_ylim([ymin,ymax])
    ax.set_xlim([xmin,xmax])

    # Plot the neuron morphology. 
    if len(neuron_x) > 0 and len(neuron_y) > 0 :
        for i in xrange(len(neuron_x)):
            ax.plot(neuron_x[i],neuron_y[i],color='black',linewidth=0.2)
    # Plot morphology.
    if len(poly_morph) > 0 :
        zips = []
        for a,b in poly_morph:
            zips.append(zip(a,b))
        polycol = mpl.collections.PolyCollection(zips,edgecolors='none',facecolors='black')
        ax.add_collection(polycol,)
    

    # Plot a circle around the soma.
    if soma_diam != 0 :
        circle = plt.Circle((0,0),soma_diam,color='black')
        ax.add_artist(circle)

    # Plot the all the signals.
    for i in xrange(len(amp)):
        signal = LFP[i,:]
        signal = signal - signal.min()
        if normalization :
            signal = signal/signal.max() * signal_size_y
        else :
            signal = signal/amp_max * signal_size_y
        signal = signal - signal[0] + yi[i]
        time_b = time_a + xi[i]
        ax.plot(time_b,signal,c=scalarMap.to_rgba(amp[i]))


    # Set new ticks for the axes.
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator())
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator())
            
    plt.tight_layout()
    if (fname is not None):
        # Create the directory if it does not exist.
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        # Create different versions of the file. 
        os.chdir(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            plt.savefig(
                    name,
                    format=format_str,
                    transparent=False, 
                    bbox_inches='tight',
                    pad_inches=0
            )
    if show :
        plt.show()
    plt.close()
    print 'finished            :', fname

def spikeAmplitudes(amps, dr=1, r_0=0, show=True, scale='linear',
        show_points=False,
        fname=None, plot_save_dir=None, mode='all'):

    print "plotting            :", fname

    amps = np.asarray(amps)

    # Use microvolt if the values are small.
    mili_volt = True
    if amps.max() < 1:
        mili_volt = False
        amps *= 1000

    amps_std = np.sqrt(np.var(amps,axis=0))
    amps_mean = np.mean(amps,axis=0)

    x = np.arange(amps.shape[1])
    x = x*dr
    x = x + r_0

    # Initial plot parameters.
    set_rc_param()
    fig = plt.figure(figsize=size_common)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid()
    ax.set_xlim([x.min(),x.max()])

    # Plot the data
    if mode == 'all' :
        # Scatter plot has a tendency to extend the limits.
        # Set the limits to follow the plotting  function.
        y_min = amps.min()
        y_max = amps.max()
        ax.set_ylim([y_min,y_max])
        # Get the colors of the viridis color map.
        color_array = get_short_color_array(amps.shape[0])
        lines = []
        for idx in range(amps.shape[0]):
            y = amps[idx,:]
            label = r'$\theta_n = %d $ ' %idx
            ret_line = ax.plot(x,y,color=color_array[idx],label=label)
            # Show the data points.
            if show_points:
                plt.scatter(x,y, 
                        marker='o',
                        color=color_array[idx],
                        linewidth=0.2
                )
            lines.append(ret_line)
        handles,labels = ax.get_legend_handles_labels()
        lgd = ax.legend(
                handles,
                labels,
                loc='center left',
                bbox_to_anchor=(1,0.5),
                ncol=amps.shape[1]/17 + 1
        )
    elif mode == 'std' :
        ax.plot(x,amps_mean,color=color_array_long[0])
        ax.fill_between(x,amps_mean-amps_std,amps_mean+amps_std,
                color=color_array_long[0],alpha=0.2)
        y_min,y_max = ax.get_ylim();
        ax.set_ylim([y_min,y_max])
        if show_points:
            plt.scatter(x,amps_mean, 
                    marker='o',
                    color=color_array_long[0],
                    linewidth=0.2
            )
        # Ugly way to put in some graphs for power laws.
        if scale == 'log' :
            # Left side.
            x0 = x[0]
            x1 = x[1]
            y0 = amps_mean[0]
            for p in [1.0,2.0,3.0]:
                y1 = np.power(1.0/x[1],p)*amps_mean[0]/np.power(1.0/x[0],p)
                ax.plot([x0,x1],[y0,y1],color='black')
                # y = np.power(1.0/x,p)*amps_mean[0]/np.power(1.0/x[0],p)
                # ax.plot(x,y,color='black')
            # Right side.
            x0 = x[-5]
            x1 = x[-1]
            y1 = amps_mean[-1]
            for p in [1.0,2.0,3.0]:
                y0 = np.power(1.0/x[-5],p)*amps_mean[-1]/np.power(1.0/x[-1],p)
                ax.plot([x0,x1],[y0,y1],color='black')
                # y = np.power(1.0/x,p)*amps_mean[-1]/np.power(1.0/x[-1],p)
                # ax.plot(x,y,color='blue')

    # Change the axis titles and the scale if log plot is selected.
    xlabel = r'Distance from Soma '
    ylabel = r'Amplitude '

    xlabel += r'\textbf[$\mathbf{\bm\upmu m}$\textbf]'
    # Use microvolt if not milivolt
    if mili_volt:
        ylabel += r'\textbf[$\mathbf{mV}$\textbf]'
    else:
        ylabel += r'\textbf[$\mathbf{\bm\upmu V}$\textbf]'

    if scale == 'log' : 
        # Set the scale for log plot.
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim([x.min(),x.max()])
        if mode == 'std':
            a_min = amps_mean.min()-amps_std.max()
            a_max = amps_mean.max()+amps_std.max()
            ax.set_ylim([a_min,a_max])
        elif mode == 'all':
            ax.set_ylim([amps.min(),amps.max()])

        ticker = mpl.ticker.MaxNLocator(nbins=7)
        # ticker = mpl.ticker.AutoLocator()
        ax.xaxis.set_major_locator(ticker)
        ax.xaxis.get_major_formatter().labelOnlyBase = False

        ticker = mpl.ticker.MaxNLocator(nbins=7)
        # ticker = mpl.ticker.AutoLocator()
        ax.yaxis.set_major_locator(ticker)
        ax.yaxis.get_major_formatter().labelOnlyBase = False

        # Set a label formatter to use normal numbers.
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        
    # Set the label which was created earlier.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    if (fname is not None):
        # Create the directory if it does not exist.
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        # Create different versions of the file. 
        os.chdir(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            plt.savefig(
                    name,
                    format=format_str,
                    transparent=False, 
                    bbox_inches='tight',
                    pad_inches=0
            )
    if show :
        plt.show()
    plt.close()
    print 'finished            :', fname

def spike_widths_and_amp_grouped(grouped_widths,
        grouped_amps, grouped_elec_pos,
        group_labels=None, neuron_labels=None, micro_volt=True,
        show=True, fname=None, plot_save_dir=None):
    """
    See :meth:`~LFPy_util.electrodes.circularElectrodes`

    :param LFP: Matrix of electrode signals at each row.
    :type LFP: :class:`~numpy.ndarray` shape = (nElectrodes,frames)
    :param int n: Number of electrodes between **r_0** and **r**.
    :param int nTheta: Number of directions from :math:`0` to :math:`2\pi`.
    :param float dr: Distance between two electrodes.
    :param int r_min: Distance to first placed electrode.
    """

    print "plotting            :", fname

    # Inital plot paramters.
    set_rc_param()
    fig = plt.figure(figsize=size_common)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid()

    if len(grouped_amps) != len(grouped_elec_pos):
        raise ValueError("grouped_amps and grouped_elec_pos are of unequal lengths.")
    if len(grouped_widths) != len(grouped_elec_pos):
        raise ValueError("grouped_widths and grouped_elec_pos are of unequal lengths.")

    # All data must be of equal lengths. Get the length of data of
    # the first neuron of the first group.
    elec_pos = np.array(grouped_elec_pos[0][0])

    # Get an array of colors for each line that will be drawn. +1 because
    # the last color is very light and cannot be seen very well.
    color_array = get_short_color_array(len(grouped_elec_pos)+1)
    # For each group of neurons.
    for i in xrange(len(grouped_elec_pos)):
        widths_list = grouped_widths[i]
        amps_list = grouped_amps[i]
        label = None
        if group_labels is not None:
            label = group_labels[i]

        # Width part.
        # Each row of width measurments is one direction with electrodes.
        # Gather all rows from each neuron into a big array. One electrode
        # gives one value for the whole simulation, spike width or amplitude
        # in this case.
        rows = 0
        for widths in widths_list:
            widths = np.array(widths)
            rows += widths.shape[0]
            if widths.shape[1] != len(elec_pos):
                raise ValueError('Data has unequal lengths.')
        widths_all = np.empty([rows,len(elec_pos)])
        # For each neuron there is multiple width measurements.  
        row = 0
        for widths in widths_list:
            widths = np.array(widths)
            # These are widths from one neuron.
            widths_all[row:row+widths.shape[0]] = widths
            row += widths.shape[0]
        # Calculate the mean and std for a group of neurons.
        widths_mean = np.mean(widths_all,axis=0)
        widths_std = np.sqrt(np.var(widths_all,axis=0))

        # Amp part.
        # Same as above, but with different data.
        rows = 0
        for amps in amps_list:
            amps = np.array(amps)
            rows += amps.shape[0]
            if amps.shape[1] != len(elec_pos):
                raise ValueError('Data has unequal lengths.')
        amps_all = np.empty([rows,len(elec_pos)])
        # For each neuron there is multiple amp measurements.  
        row = 0
        for amps in amps_list:
            amps = np.array(amps)
            amps_all[row:row+amps.shape[0]] = amps
            row += amps.shape[0]
        if micro_volt:
            amps_all *= 1000;
        # Calculate the mean and std for a group of neurons.
        amps_mean = np.mean(amps_all,axis=0)
        amps_std = np.sqrt(np.var(amps_all,axis=0))

        # Plot.
        ax.plot(widths_mean,
                amps_mean,
                color=color_array[i],
                label=label,
                marker='o',
                markersize=5,
        )
        upper_y = amps_mean + amps_std
        upper_x = widths_mean + widths_std
        lower_y = amps_mean - amps_std
        lower_x = widths_mean - widths_std
        # Fill between does not works for specials shapes like these functions.
        # Using a polygon instead. [::-1] reverses the array.
        y = np.hstack((upper_y,lower_y[::-1]))
        x = np.hstack((upper_x,lower_x[::-1]))
        points = np.zeros([len(elec_pos)*2,2])
        points[:,0] = x
        points[:,1] = y
        points = points.tolist()
        patch = plt.Polygon(
                points,
                color=color_array[i],
                fill=True,
                edgecolor=None,
                alpha=0.2,
        )
        ax.add_patch(patch)
    if group_labels is not None:
        handles,labels = ax.get_legend_handles_labels()
        lgd = ax.legend(
                handles,
                labels,
                loc='center left',
                bbox_to_anchor=(1,0.5),
                ncol=widths.shape[1]/17 + 1
        )

    ax.set_xlabel(r'Spike width \textbf[$\mathbf{ms}$\textbf]')
    if micro_volt:
        ax.set_ylabel(r'Amplitude \textbf[$\mathbf{\bm\upmu V}$\textbf]')
    else:
        ax.set_ylabel(r'Amplitude \textbf[$\mathbf{mV}$\textbf]')

    plt.tight_layout()
    if (fname is not None):
        # Create the directory if it does not exist.
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        # Create different versions of the file. 
        os.chdir(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            plt.savefig(
                    name,
                    format=format_str,
                    transparent=False, 
                    bbox_inches='tight',
                    pad_inches=0
            )
    if show :
        plt.show()
    plt.close()
    print 'finished            :', fname

def spike_amps_grouped(grouped_amps, grouped_elec_pos,
        group_labels=None, neuron_labels=None, micro_volt=True,
        scale='linear', mode='all',
        show=True, fname=None, plot_save_dir=None):
    """
    """

    print "plotting            :", fname

    # Inital plot paramters.
    set_rc_param()
    fig = plt.figure(figsize=size_common)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid()

    if len(grouped_amps) != len(grouped_elec_pos):
        raise ValueError("grouped_amps and grouped_elec_pos are of unequal lengths.")

    # All data must be of equal lengths. Get the length of data of
    # the first neuron of the first group.
    elec_pos = np.array(grouped_elec_pos[0][0])

    # Plot data.
    if mode == 'std' :
        # For each group of neurons.
        color_array = get_short_color_array(len(grouped_elec_pos)+1)
        # For each group of neurons.
        for i in xrange(len(grouped_elec_pos)):
            amps_list = grouped_amps[i]
            elec_pos_list = grouped_elec_pos[i]
            label = None
            if group_labels is not None:
                label = group_labels[i]

            rows = 0
            for amps in amps_list:
                amps = np.array(amps)
                rows += amps.shape[0]
                if amps.shape[1] != len(elec_pos):
                    raise ValueError('Data has unequal lengths.')
            amps_all = np.empty([rows,len(elec_pos)])
            # For each neuron there is multiple width measurements.  
            row = 0
            for j, amps in enumerate(amps_list):
                # These are amps from one neuron.
                amps = np.asarray(amps)
                amps_all[row:row+amps.shape[0]] = amps
                row += amps.shape[0]

            # Calculate the mean and std for a group of neurons.
            if micro_volt:
                amps_all *= 1000
            amps_mean = np.mean(amps_all,axis=0)
            amps_std = np.sqrt(np.var(amps_all,axis=0))

            line = ax.plot(
                    elec_pos,
                    amps_mean,
                    color=color_array[i],
                    label=label,
                    marker='o',
                    markersize=5,)
            ax.fill_between(
                    elec_pos,
                    amps_mean-amps_std,
                    amps_mean+amps_std,
                    color=color_array[i],
                    alpha=0.2
            )
        if group_labels is not None:
            handles,labels = ax.get_legend_handles_labels()
            lgd = ax.legend(
                    handles,
                    labels,
                    loc='center left',
                    bbox_to_anchor=(1,0.5),
                    ncol=amps.shape[1]/17 + 1
            )

    # Change the axis titles and the scale if log plot is selected.
    if scale == 'linear':
        # Use microvolt if not milivolt
        if micro_volt:
            ylabel = r'\textbf[$\mathbf{\bm\upmu V}$\textbf]'
        else:
            ylabel = r'\textbf[$\mathbf{mV}$\textbf]'
        ax.set_ylabel(ylabel)
        ax.set_xlabel(
                r'Distance from Soma \textbf[$\mathbf{\bm\upmu m}$\textbf]')

    plt.tight_layout()
    if (fname is not None):
        # Create the directory if it does not exist.
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        # Create different versions of the file. 
        os.chdir(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            plt.savefig(
                    name,
                    format=format_str,
                    transparent=False, 
                    bbox_inches='tight',
                    pad_inches=0
            )
    if show :
        plt.show()
    plt.close()
    print 'finished            :', fname

def spike_widths_grouped(grouped_widths, grouped_elec_pos,
        group_labels=None, neuron_labels=None,
        scale='linear', mode='all',
        show=True, fname=None, plot_save_dir=None):
    """
    """

    print "plotting            :", fname

    # Inital plot paramters.
    set_rc_param()
    fig = plt.figure(figsize=size_common)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid()

    if len(grouped_widths) != len(grouped_elec_pos):
        raise ValueError("grouped_widths and grouped_elec_pos are of unequal lengths.")

    # All data must be of equal lengths. Get the length of data of
    # the first neuron of the first group.
    elec_pos = np.array(grouped_elec_pos[0][0])

    # Plot data.
    if mode == 'std' :
        # For each group of neurons.
        color_array = get_short_color_array(len(grouped_elec_pos)+1)
        # For each group of neurons.
        for i in xrange(len(grouped_elec_pos)):
            widths_list = grouped_widths[i]
            elec_pos_list = grouped_elec_pos[i]
            label = None
            if group_labels is not None:
                label = group_labels[i]

            rows = 0
            for widths in widths_list:
                widths = np.array(widths)
                rows += widths.shape[0]
                if widths.shape[1] != len(elec_pos):
                    raise ValueError('Data has unequal lengths.')
            widths_all = np.empty([rows,len(elec_pos)])
            # For each neuron there is multiple width measurements.  
            row = 0
            for j, widths in enumerate(widths_list):
                # These are widths from one neuron.
                widths = np.asarray(widths)
                widths_all[row:row+widths.shape[0]] = widths
                row += widths.shape[0]

            # Calculate the mean and std for a group of neurons.
            widths_mean = np.mean(widths_all,axis=0)
            widths_std = np.sqrt(np.var(widths_all,axis=0))

            line = ax.plot(
                    elec_pos,
                    widths_mean,
                    color=color_array[i],
                    label=label,
                    marker='o',
                    markersize=5,)
            ax.fill_between(
                    elec_pos,
                    widths_mean-widths_std,
                    widths_mean+widths_std,
                    color=color_array[i],
                    alpha=0.2
            )
        if group_labels is not None:
            handles,labels = ax.get_legend_handles_labels()
            lgd = ax.legend(
                    handles,
                    labels,
                    loc='center left',
                    bbox_to_anchor=(1,0.5),
                    ncol=widths.shape[1]/17 + 1
            )

    # Change the axis titles and the scale if log plot is selected.
    if scale == 'linear':
        ax.set_ylabel(r'Signal Width \textbf[$\mathbf{ms}$\textbf]')
        ax.set_xlabel(
                r'Distance from Soma \textbf[$\mathbf{\bm\upmu m}$\textbf]')

    plt.tight_layout()
    if (fname is not None):
        # Create the directory if it does not exist.
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        # Create different versions of the file. 
        os.chdir(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            plt.savefig(
                    name,
                    format=format_str,
                    transparent=False, 
                    bbox_inches='tight',
                    pad_inches=0
            )
    if show :
        plt.show()
    plt.close()
    print 'finished            :', fname

def spikeWidths(widths, dr=1, r_0=0, threshold=None,
        scale='linear', mode='all', show_points=False,
        show=True, fname=None, plot_save_dir=None):
    """
    Plot the mean spike width radially away from a point r_0 and 
    also show the variance. 

    See :meth:`~LFPy_util.electrodes.circularElectrodes`

    :param LFP: Matrix of electrode signals at each row.
    :type LFP: :class:`~numpy.ndarray` shape = (nElectrodes,frames)
    :param int n: Number of electrodes between **r_0** and **r**.
    :param int nTheta: Number of directions from :math:`0` to :math:`2\pi`.
    :param float dr: Distance between two electrodes.
    :param int r_min: Distance to first placed electrode.
    """

    print "plotting            :", fname

    widths = np.asarray(widths)

    widths_std = np.sqrt(np.var(widths,axis=0))
    widths_mean = np.mean(widths,axis=0)

    # x is the vector of electrodes
    x = np.arange(widths.shape[1])
    x = x*dr
    x = x + r_0


    # Inital plot paramters.
    set_rc_param()
    fig = plt.figure(figsize=size_common)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.grid()
    ax.set_xlim([x.min(),x.max()])

    # Plot data.
    if mode == 'all' :
        # Scatter plot has a tendency to extend the limits.
        # Set the limits to follow the plotting  function.
        y_min = widths.min()
        y_max = widths.max()
        ax.set_ylim([y_min,y_max])
        color_array = get_short_color_array(widths.shape[0])
        lines = []
        for idx in range(widths.shape[0]):
            y = widths[idx,:]
            label = r'$\theta_n = %d $ ' %idx
            ret_line = ax.plot(x,y,color=color_array[idx],label=label)
            # Append the lines so we can make a legend. 
            lines.append(ret_line)
            # Show the data points.
            if show_points:
                plt.scatter(x,y, 
                        marker='o',
                        color=color_array[idx],
                        linewidth=0.2
                )
        handles,labels = ax.get_legend_handles_labels()
        lgd = ax.legend(
                handles,
                labels,
                loc='center left',
                bbox_to_anchor=(1,0.5),
                ncol=widths.shape[1]/17 + 1
        )
    elif mode == 'std' :
        # w_min = widths_mean.min() - widths_std.min();
        # w_max = widths_mean.max() + widths_std.max();
        # ax.set_ylim([w_min,w_max])
        ax.plot(x,widths_mean,color=color_array_long[0])
        ax.fill_between(x,widths_mean-widths_std,widths_mean+widths_std,
                color=color_array_long[0],alpha=0.2)
        y_min,y_max = ax.get_ylim();
        ax.set_ylim([y_min,y_max])
        if show_points:
            plt.scatter(x,widths_mean, 
                    marker='o',
                    color=color_array_long[0],
                    linewidth=0.2
            )

    # Change the axis titles and the scale if log plot is selected.
    if scale == 'linear':
        if threshold != None : 
            thresh_str = str(int(threshold*100)) + ' \% '
            ax.set_ylabel(thresh_str
                    + r'Signal Width \textbf[$\mathbf{ms}$\textbf]')
        else :
            ax.set_ylabel(r'Signal Width \textbf[$\mathbf{ms}$\textbf]')
        ax.set_xlabel(
                r'Distance from Soma \textbf[$\mathbf{\bm\upmu m}$\textbf]')
    elif scale == 'log' : 
        if threshold != None : 
            thresh_str = str(int(threshold*100)) + ' \% '
            ax.set_ylabel(thresh_str
                    + r'Signal Width \textbf[$\mathbf{ms}$\textbf]')
        else :
            ax.set_ylabel(r'Signal Width \textbf[$\mathbf{ms}$\textbf]')
        ax.set_xlabel(
                r'Distance from Soma \textbf[$\mathbf{\bm\upmu m}$\textbf]')

        # Set the scale for log plot.
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim([x.min(),x.max()])
        if mode == 'std':
            w_min = widths_mean.min()-widths_std.max()
            w_max = widths_mean.max()+widths_std.max()
            ax.set_ylim([w_min,w_max])
        elif mode == 'all':
            ax.set_ylim([widths.min(),widths.max()])

        # ticker = mpl.ticker.MaxNLocator(nbins=7)
        ticker = mpl.ticker.AutoLocator()
        ax.xaxis.set_major_locator(ticker)
        ax.xaxis.get_major_formatter().labelOnlyBase = False

        # ticker = mpl.ticker.MaxNLocator(nbins=7)
        ticker = mpl.ticker.AutoLocator()
        ax.yaxis.set_major_locator(ticker)
        ax.yaxis.get_major_formatter().labelOnlyBase = False

        # Set a label formatter to use normal numbers.
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    plt.tight_layout()
    if (fname is not None):
        # Create the directory if it does not exist.
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        # Create different versions of the file. 
        os.chdir(plot_save_dir)
        for format_str in plot_format:
            name = fname+'.'+format_str
            plt.savefig(
                    name,
                    format=format_str,
                    transparent=False, 
                    bbox_inches='tight',
                    pad_inches=0
            )
    if show :
        plt.show()
    plt.close()
    print 'finished            :', fname
    
