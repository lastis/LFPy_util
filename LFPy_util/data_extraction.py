"""
Data extraction module
"""
# pylint: disable=ungrouped-imports, no-member
import numpy as np
import scipy.fftpack as ff
import warnings
import quantities as pq
from sklearn.decomposition import PCA
from neuron import h
from scipy.signal import argrelextrema
from scipy.stats.mstats import zscore

def tetrode_spikes_mean(signal, amp_option='pos'):
    """
    Input is a 3d array, (spikes x electrodes x time)
    Takes the strongest signal from each electrode and means them.

    Each electrode records the same spike.
    """
    if amp_option == 'pos':
        pass
    elif amp_option == 'neg':
        signal = -signal
    elif amp_option == 'both':
        signal = np.fabs(signal)
    # Take the maximum of all signals
    signal_max = np.amax(signal, axis=2)
    # Make a list with the index of the electrode that has the highest value
    # for each spike.
    signal_max_idx = np.argmax(signal_max, axis=1)
    cols = signal.shape[0]
    signal_out = np.mean(signal[np.arange(cols),signal_max_idx], axis=0)
    return signal_out

def combined_mean_std(mean, std, axis=0):
    mean = np.array(mean)
    std = np.array(std)

    if mean.shape != std.shape:
        raise ValueError("mean and std must have equal shape.")

    var = np.power(std,2)
    mean_tot, var_tot = combined_mean_var(mean, var, axis)
    return mean_tot, np.sqrt(var_tot)

def combined_mean_var(mean_array, var_array, axis=0):
    mean_array = np.array(mean_array)
    var_array = np.array(var_array)

    if mean_array.shape != var_array.shape:
        raise ValueError("mean_array and var must have equal shape.")

    mean_array_new = np.mean(mean_array, axis=axis)
    var_array_new = np.mean(var_array + mean_array*mean_array, axis=axis) \
        - mean_array_new*mean_array_new

    return mean_array_new.squeeze(), var_array_new.squeeze()

def maxabs(a, axis=None):
    """Return slice of a, keeping only those values that are furthest away
    from 0 along axis"""
    maxa = a.max(axis=axis)
    mina = a.min(axis=axis)
    p = abs(maxa) > abs(mina) # bool, or indices where +ve values win
    n = abs(mina) > abs(maxa) # bool, or indices where -ve values win
    if axis == None:
        if p: return maxa
        else: return mina
    shape = list(a.shape)
    shape.pop(axis)
    out = np.zeros(shape, dtype=a.dtype)
    out[p] = maxa[p]
    out[n] = mina[n]
    return out

def find_spikes(t_vec, v_vec, threshold=4, pre_dur=0, post_dur=0, amp_option='both',
        threshold_abs=None):
    """
    Get the indices of the spikes. 
    pre_dur and post_dur are used to ignore spikes at the end and start of signal.
    """
    # pylint: disable=no-member
    v_vec_original = v_vec
    v_vec = zscore(v_vec)
    threshold = np.fabs(threshold)
    dt = t_vec[1] - t_vec[0]

    pre_idx = int(pre_dur / float(dt))
    post_idx = len(t_vec) - int(post_dur / float(dt))
    if pre_idx > post_idx :
        raise ValueError("pre_dur + post_dur are longer than the time vector.")

    if amp_option == 'pos':
        pass
    elif amp_option == 'neg':
        v_vec = - v_vec
    elif amp_option == 'both':
        if -v_vec.min() > v_vec.max():
            v_vec = -v_vec 
            v_vec_original = -v_vec_original 

    # Find local maxima.
    max_idx = argrelextrema(v_vec, np.greater)[0]
    max_idx = np.array(max_idx,dtype='int')
    v_max = v_vec[max_idx]
    max_idx = max_idx[v_max > threshold]
    max_idx = max_idx[max_idx > pre_idx]
    max_idx = max_idx[max_idx < post_idx]
    if threshold_abs is not None:
        v_max_original = v_vec_original[max_idx]
        max_idx = max_idx[v_max_original > threshold_abs]

    return max_idx


def get_polygons_axon(cell, projection=('x', 'y')):
    """
    Get polygons of the axon.
    """
    axon_exclude = lambda s: True if 'axon' in s else False
    return get_polygons(cell, projection, axon_exclude)


def get_polygons_no_axon(cell, projection=('x', 'y')):
    """
    Get polygons excluding the axon.
    """
    axon = lambda s: False if 'axon' in s else True
    return get_polygons(cell, projection, axon)


def get_polygons(cell, projection=('x', 'y'), comp_func=None):
    """
    If comp_func(str) is False, the section will be skipped.
    """
    # pylint: disable=protected-access
    polygons = []
    cnt = -1
    for sec in cell.allseclist:
        for _ in xrange(sec.nseg):
            cnt += 1
            if comp_func is not None and not comp_func(sec.name()):
                continue
            polygons.append(cell._create_segment_polygon(cnt, projection))
    return polygons


def extract_spikes(t_vec,
                   v_vec,
                   pre_dur=16.7*0.5,
                   post_dur=16.7*0.5,
                   threshold=4,
                   amp_option='both',
                   threshold_abs=None):
    """
    Get a new matrix of all spikes of the input v_vec.
    """
    # pylint: disable=invalid-name, too-many-branches, too-many-arguments,too-many-locals
    t_vec = np.array(t_vec)
    v_vec = np.array(v_vec)
    if len(t_vec) != len(v_vec):
        raise ValueError("t_vec and v_vec have unequal lengths.")

    threshold = np.fabs(threshold)
    dt = t_vec[1] - t_vec[0]
    pre_idx = int(pre_dur / float(dt))
    post_idx = int(post_dur / float(dt))
    if pre_idx + post_idx > len(t_vec):
        # The desired durations before and after spike are too long.
        raise ValueError("pre_dur + post_dur are longer than the time vector.")
    if pre_idx == 0 and post_idx == 0:
        raise ValueError("pre_dur and post_dur cannot both be 0.")

    v_vec_unmod = v_vec
    v_vec = zscore(v_vec)
    if amp_option == 'pos':
        pass
    elif amp_option == 'neg':
        v_vec = - v_vec
    elif amp_option == 'both':
        if -v_vec.min() > v_vec.max():
            v_vec = -v_vec 
            v_vec_unmod = -v_vec_unmod 

    # Find local maxima (or minima).
    max_idx = argrelextrema(v_vec, np.greater)[0]

    # Return if no spikes were found.
    if len(max_idx) == 0:
        warnings.warn("No local maxima found.", RuntimeWarning)
        return np.array([]), np.array([]), np.array([])
    # Only count local maxima over threshold as spikes.
    v_max = v_vec[max_idx]
    v_max_unmod = v_vec_unmod[max_idx]
    length = len(v_max) - 1
    for i in xrange(length, -1, -1):
        # Remove local maxima that is not above threshold or if the spike
        # shape cannot fit inside pre_dur and post_dur
        check_1 = v_max[i] < threshold
        check_2 = max_idx[i] < pre_idx
        check_3 = max_idx[i] + post_idx > len(t_vec)
        check_4 = v_max_unmod[i] < threshold_abs
        if (check_1 or check_2 or check_3 or check_4):
            v_max = np.delete(v_max, i)
            max_idx = np.delete(max_idx, i)

    spike_cnt = len(max_idx)
    # Return if no spikes were found.
    if spike_cnt == 0:
        warnings.warn("No maxima above threshold.", RuntimeWarning)
        return np.array([]), np.array([]), np.array([])
    n = pre_idx + post_idx

    spikes = np.zeros([spike_cnt, n])
    I = np.zeros([spike_cnt, 2], dtype=np.int)

    for i in xrange(spike_cnt):
        start_idx = max_idx[i] - pre_idx
        end_idx = max_idx[i] + post_idx
        I[i, 0] = start_idx
        I[i, 1] = end_idx
        spikes[i, :] = v_vec_unmod[start_idx:end_idx]
    t_vec_new = np.arange(spikes.shape[1]) * dt
    return spikes, t_vec_new, I


def find_freq_and_fft(timestep, signal, length=None, axis=-1, f_cut=None):
    """
    Calculate the magnitude and phase of a signal using fft.

    The input must be a real sequence. 
    

    :param `~numpy.ndarray` tvec: 
        Time vector with length equal **sig**. 
    :param `~numpy.ndarray` sig: 
        Signal to analyze.
    :returns: 
        *  
         freq (:class:`~numpy.ndarray`) 
        * 
         amp (:class:`~numpy.ndarray`)
        * 
         phase (:class:`~numpy.ndarray`)

    Example:
        .. code-block:: python

            freq, amp, phase = fndFreqAndFft(tvec,sig)

    """
    signal = np.array(signal)
    N = signal.shape[axis]
    if length is None:
        length = signal.shape[axis]
    freqs = ff.fftfreq(length, d=timestep)
    # freqs = freqs[:N/2]
    freqs = np.array_split(freqs, 2)[0]
    ft = ff.fft(signal, n=length, axis=axis) / N
    # Multiply by two when removing half the specter to keep 
    # energy conserved.
    # ft = np.take(ft[:N/2], axis=axis) * 2
    ft = np.array_split(ft, 2, axis=axis)[0] * 2
    amplitude = np.abs(ft)
    phase = np.angle(ft, deg=0)
    if f_cut is not None:
        idx = min(
            range(len(freqs)), 
            key=lambda i: abs(freqs[i] - f_cut)
            )
        freqs = freqs[0:idx]
        amplitude = amplitude[0:idx]
        phase = phase[0:idx]
    return freqs, amplitude, phase


def _from_to_distance(origin_segment, to_segment):
    h.distance(0, origin_segment.x, sec=origin_segment.sec)
    return h.distance(to_segment.x, sec=to_segment.sec)


def _longest_path(sec_ref, path=[]):
    length = 0
    ret_path = []
    if len(sec_ref.child) == 0:
        length = h.distance(1, sec=sec_ref.sec)
        ret_path = path
    else:
        for idx, sec in enumerate(sec_ref.child):
            child_path = path[:]
            child_path.append(idx)
            sr = h.SectionRef(sec=sec)
            p, l = _longest_path(sr, child_path)
            if l > length:
                length = l
                ret_path = p
    return ret_path, length


def _walk_path(sec_ref, path):
    rec_pos = np.zeros([len(path), 3])
    v_vec_list = []
    i_vec_list = []
    t_vec = h.Vector()
    t_vec.record(h._ref_t)
    for cnt, idx in enumerate(path):
        v_vec = h.Vector()
        i_vec = h.Vector()
        sec = sec_ref.child[idx]
        sec_pos = 0.5
        # Find closest segment index to sec_pos.
        idx_seg = int((h.n3d(sec=sec) - 1) * sec_pos)
        rec_pos[cnt, 0] = h.x3d(idx_seg, sec=sec)
        rec_pos[cnt, 1] = h.y3d(idx_seg, sec=sec)
        rec_pos[cnt, 2] = h.z3d(idx_seg, sec=sec)
        v_vec.record(sec(sec_pos)._ref_v)
        i_vec.record(sec(sec_pos)._ref_i_membrane)
        v_vec_list.append(v_vec)
        i_vec_list.append(i_vec)
        sec_ref = h.SectionRef(sec=sec)
    return v_vec_list, i_vec_list, t_vec, rec_pos


def rec_along_longest_branch():
    """
    Record membrane current and potential along the longest branch. 

    The vectors will be filled only after the neuron simulation is run as the
    vectors are :class:`Vector`. 

    Data is gathered directly from neuron and not LFPy. Treversal of the
    longest branch is done usingn :class:`neuron:SectionRef` and 
    :class:`neuron:SectionRef.child()`

    :returns: 
        *  
         :class:`neuron:Vector` -- List of membrane potentials.
        * 
         :class:`neuron:Vector` -- List of membrane currents.
        * 
         :class:`neuron:Vector` -- Time vector.
        *
         :class:`~numpy.ndarray` -- Position of recording sites. 

    Example:
        .. code-block:: python

            v_vec_list, i_vec_list, t_vec, rec_pos = rec_along_longest_branch()
            h.run()

    """
    sr = h.SectionRef()
    # Start the distance calculation from the root section.
    h.distance(0, 0.5, sec=sr.root)
    path, dist = _longest_path(sr)
    return _walk_path(sr, path)


def get_pos_data():
    """
    Get positions x, y, z for all segments and their diameter. 

    :returns: 
        4 lists: x,y,z,d. One element per section where each element is
        a :class:`~numpy.ndarray`.

    Example:
        .. code-block:: python

            x,y,z,d = get_pos_data()
            for sec in xrange(len(x)):
                for seg in xrange(len(x[sec]):
                    print x[sec][seg], y[sec][seg], z[sec][seg]
    """
    x = []
    y = []
    z = []
    d = []
    for sec in h.allsec():
        n3d = int(h.n3d())
        x_i, y_i, z_i = np.zeros(n3d), np.zeros(n3d), np.zeros(n3d),
        d_i = np.zeros(n3d)
        for i in xrange(n3d):
            x_i[i] = h.x3d(i)
            y_i[i] = h.y3d(i)
            z_i[i] = h.z3d(i)
            d_i[i] = h.diam3d(i)
        x.append(x_i)
        y.append(y_i)
        z.append(z_i)
        d.append(d_i)
    return x, y, z, d


def get_pos_data_short():
    """
    Get positions of all segments currently loaded in Neuron in a simple matrix.
    Section position information is not available.

    :returns: 
        Matrix (3 x nSegments) With x,y,z positions. 
    :rtype: :class:`~numpy.ndarray`

    Example:
        .. code-block:: python

            data = get_pos_data_short()
    """
    n = 0
    for sec in h.allsec():
        n += int(h.n3d())
    data = np.zeros([4, n])
    cnt = 0
    for sec in h.allsec():
        for i in xrange(int(h.n3d())):
            data[0, cnt] = h.x3d(i)
            data[1, cnt] = h.y3d(i)
            data[2, cnt] = h.z3d(i)
            data[3, cnt] = h.diam3d(i)
            cnt += 1
    return data

def find_major_axes():
    """
    Find the principal geometrical components of the neuron currently loaded 
    with Neuron. Uses :class:`sklearn.decomposition.PCA`. 

    If used with :class:`LFPy.Cell`, the parameter **pt3d** must be set to True.

    :returns: 
        Matrix (3 x 3) where each row is a principal component.
    :rtype: :class:`~numpy.ndarray`

    Example:
        .. code-block:: python

            # Find the principal component axes and rotate cell.
            axes = LFPy_util.data_extraction.findMajorAxes()
            LFPy_util.rotation.alignCellToAxes(cell,axes[0],axes[1])
    """
    points = get_pos_data_short()
    pca = PCA(n_components=3)
    pca.fit(points[:3].T)
    return pca.components_


def find_wave_width_type_II(matrix, threshold=0.5, dt=1, amp_option='both'):
    """
    Compute wave width at some fraction of max amplitude. Counts the number
    of indices above threshold and multiplies by **dt**. 

    :param `~numpy.ndarray` matrix: 
        Matrix (nSignals x frames) of 1D signals at each row. . 
        What happens now. Is this to long and will go over two lines
        or does it linebreak.
    :param float threshold: 
        Between 0 and 1.
    :param float dt: 
        Time per frame.
    :param string amp_option: 
        Calculate width from the negative side or the positive side.
        Can either 'both', 'neg' or 'pos'.
    :returns: 
        Array (nSignals) with widths.
    :rtype: :class:`~numpy.ndarray`

    Example:
        .. code-block:: python

            widths, trace = LFPy_util.data_extraction.findWaveWidthsSimple(LFP)
    """
    matrix = np.array(matrix)
    if len(matrix.shape) == 1:
        matrix = np.reshape(matrix, (1, -1))
    widths = np.zeros(matrix.shape[0])
    trace = np.empty(matrix.shape)
    trace[:] = np.NAN
    for row in xrange(matrix.shape[0]):
        signal = matrix[row].copy()
        signal -= signal[0]
        # Flip the signal if the negative side should be used.
        if amp_option == 'neg':
            signal = -signal
        elif amp_option == 'both' and signal.max() < -signal.min():
            signal = -signal
        thresh_abs = signal.max() * np.fabs(threshold)
        signal_bool = signal > thresh_abs
        signal_index = np.where(signal_bool)[0]
        widths[row] = np.sum(signal_bool)
        trace[row, signal_index] = matrix[row, signal_index[-1]]
    return widths * dt, trace


def find_wave_width_type_I(matrix, dt=1):
    """
    Wave width defined as time from minimum to maximum.
    """

    matrix = np.array(matrix)
    if len(matrix.shape) == 1:
        matrix = np.reshape(matrix, (1, -1))
    widths = np.zeros(matrix.shape[0])
    trace = np.empty(matrix.shape)
    trace[:] = np.NAN
    for row in xrange(matrix.shape[0]):
        signal = matrix[row].copy()
        offset = signal[0]
        signal -= offset
        # Assume that maximum abs. value is the "spiking" direction.
        if signal.max() < -signal.min():
            signal = -signal
        idx_1 = np.argmax(signal)
        idx_2 = idx_1 + np.argmin(signal[idx_1:])
        widths[row] = idx_2 - idx_1
        trace[row, idx_1:idx_2] = matrix[row, idx_1:].max() * 1.05
    return widths * dt, trace


def find_amplitude_type_I(matrix, amp_option='both'):
    """
    Finds the amplitude of signals. Simply takes the maximum absolute value.

    :param `~numpy.ndarray` matrix:
        Matrix (nSignals x frames) of 1D signals at each row. Can
        also be a single vector.
    :param string amp_option:
        Calculate amplitude from the negative side or the positive side.
        Can either 'both', 'neg' or 'pos'.
    :returns:
        *
         :class:`~numpy.ndarray` --
         Array (nSignals) of amplitudes.

    Example:
        .. code-block:: python

            amps = LFPy_util.data_extraction.findAmplitudeSimple(LFP)
    """
    matrix = np.array(matrix)
    if matrix.ndim == 1:
        matrix = matrix.reshape([1, -1])
    amp_low = np.zeros(matrix.shape[0])
    if amp_option == 'both' or amp_option == 'neg':
        for row in xrange(matrix.shape[0]):
            amp_low[row] = np.abs(np.min(matrix[row] - matrix[row,0]))
    amp_high = np.zeros(matrix.shape[0])
    if amp_option == 'both' or amp_option == 'pos':
        for row in xrange(matrix.shape[0]):
            amp_high[row] = np.abs(np.max(matrix[row] - matrix[row,0]))
    amp = np.maximum(amp_low, amp_high)
    return amp


def find_amplitude_type_II(matrix):
    """
    Finds the amplitude of signals from minimum to maximum.

    :param `~numpy.ndarray` matrix:
        Matrix (nSignals x frames) of 1D signals at each row. Can
        also be a single vector.
    :returns:
        *
         :class:`~numpy.ndarray` --
         Array (nSignals) of amplitudes.

    Example:
        .. code-block:: python

            amps = LFPy_util.data_extraction.findAmplitudeSimple(LFP)
    """
    matrix = np.array(matrix)
    if matrix.ndim == 1:
        matrix = matrix.reshape([1, -1])
    amps = np.zeros(matrix.shape[0])
    for row in xrange(matrix.shape[0]):
        signal = matrix[row]
        offset = signal[0]
        signal -= offset

        amp_1 = signal.max()
        amp_2 = signal.min()
        amps[row] = np.fabs(amp_2 - amp_1)
    return amps
