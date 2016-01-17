import numpy as np
from sklearn.decomposition import PCA
import scipy.fftpack as ff
from neuron import h
from scipy.signal import argrelextrema
from scipy.stats.mstats import zscore

def find_spikes(v_vec, threshold=1):
    v_vec = zscore(v_vec)
    max_idx = argrelextrema(v_vec,np.greater)[0]
    v_max = v_vec[max_idx]
    length = len(v_max)-1
    for i in xrange(length,-1,-1):
        # Remove local maxima that is not above threshold or if the spike
        # shape cannot fit inside pre_dur and post_dur
        if (v_max[i] < threshold):
            # v_max = np.delete(v_max,i)
            max_idx = np.delete(max_idx,i)
    return max_idx

def extract_spikes(t_vec, v_vec, pre_dur=0, post_dur=0,threshold=3,
        amp_option='pos'):
    t_vec = np.array(t_vec)
    v_vec = np.array(v_vec)
    if len(t_vec) != len(v_vec):
        raise ValueError("t_vec and v_vec have unequal lengths.")

    threshold = np.fabs(threshold)
    dt = t_vec[1] - t_vec[0]
    pre_idx = int(pre_dur/float(dt));
    post_idx = int(post_dur/float(dt));
    if pre_idx + post_idx > len(t_vec):
        # The desired durations before and after spike are too long.
        raise ValueError("pre_dur + post_dur are longer than the time vector.")

    v_vec_unmod = v_vec
    v_vec = zscore(v_vec)
    if amp_option == 'neg':
        v_vec = -v_vec

    # Find local maxima (or minima).
    max_idx = argrelextrema(v_vec,np.greater)[0]

    # Return if no spikes were found.
    if len(max_idx) == 0:
        # print "Warning (extract_spikes): No local maxima found!"
        return [], [], []

    # Only count local maxima over threshold as spikes.
    v_max = v_vec[max_idx]
    length = len(v_max)-1
    for i in xrange(length,-1,-1):
        # Remove local maxima that is not above threshold or if the spike
        # shape cannot fit inside pre_dur and post_dur
        # print v_max[i] < threshold 
        # print max_idx[i] < pre_idx 
        # print max_idx[i] + post_idx > len(t_vec)
        if (v_max[i] < threshold or
                max_idx[i] < pre_idx or
                max_idx[i] + post_idx > len(t_vec)):
            v_max = np.delete(v_max,i)
            max_idx = np.delete(max_idx,i)

    # Return if no spikes were found.
    if len(max_idx) == 0:
        print "Warning (extract_spikes): No maxima above threshold."
        return [], [], []

    spike_cnt = len(max_idx)
    n = pre_idx + post_idx
    if n == 0:
        n = len(t_vec)

    spikes = np.zeros([spike_cnt,n])
    I = np.zeros([spike_cnt,2],dtype=np.int)

    for i in xrange(spike_cnt):
        if n == len(t_vec):
            start_idx = 0
            end_idx = len(t_vec)
        else:
            start_idx = max_idx[i] - pre_idx
            end_idx = max_idx[i] + post_idx
        I[i,0] = start_idx
        I[i,1] = end_idx
        spikes[i,:] = v_vec_unmod[start_idx:end_idx]
    t_vec_new = np.arange(spikes.shape[1])*dt
    return spikes, t_vec_new, I


def findFreqAndFft(tvec, sig):
    """
    Amplitude and frequency of the input signal using fourier analysis.

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
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")

    timestep = (tvec[1] - tvec[0])/1. if type(tvec) in [list, np.ndarray] else tvec
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]
    Y = ff.fft(sig, axis=1)[:, pidxs[0]]
    amplitude = np.abs(Y)/Y.shape[1]
    phase = np.angle(Y, deg=0)
    #power = np.abs(Y)**2/Y.shape[1]
    return freqs, [amplitude], [phase]

def _from_to_distance(origin_segment, to_segment):
    h.distance(0, origin_segment.x, sec=origin_segment.sec)
    return h.distance(to_segment.x, sec=to_segment.sec)

def _longest_path(sec_ref,path=[]):
    length = 0
    ret_path = []
    if len(sec_ref.child) == 0:
        length = h.distance(1,sec=sec_ref.sec)
        ret_path = path
    else:
        for idx, sec in enumerate(sec_ref.child):
            child_path = path[:]
            child_path.append(idx)
            sr = h.SectionRef(sec=sec)
            p, l = _longest_path(sr,child_path)
            if l > length:
                length = l
                ret_path = p
    return ret_path, length

def _walk_path(sec_ref, path):
    rec_pos = np.zeros([len(path),3])
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
        idx_seg = int((h.n3d(sec=sec)-1)*sec_pos)
        rec_pos[cnt,0] = h.x3d(idx_seg,sec=sec)
        rec_pos[cnt,1] = h.y3d(idx_seg,sec=sec)
        rec_pos[cnt,2] = h.z3d(idx_seg,sec=sec)
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
    h.distance(0,0.5,sec=sr.root)
    path, dist = _longest_path(sr)
    return _walk_path(sr,path)

def getPositionData():
    """
    Get positions x, y, z for all segments and their diameter. 

    :returns: 
        4 lists: x,y,z,d. One element per section where each element is
        a :class:`~numpy.ndarray`.

    Example:
        .. code-block:: python

            x,y,z,d = getPositionData()
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
    return x,y,z,d

def getPositionDataShort():
    """
    Get positions of all segments currently loaded in Neuron in a simple matrix.
    Section position information is not available.

    :returns: 
        Matrix (nSegments x 3) With x,y,z positions. 
    :rtype: :class:`~numpy.ndarray`

    Example:
        .. code-block:: python

            points = getPositionDataShort()
    """
    n = 0
    for sec in h.allsec():
        n += int(h.n3d())
    points = np.zeros([n,3])
    cnt = 0
    for sec in h.allsec():
        for i in xrange(int(h.n3d())):
            points[cnt,0] = h.x3d(i)
            points[cnt,1] = h.y3d(i)
            points[cnt,2] = h.z3d(i)
            cnt += 1
    return points

def findMajorAxes():
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
    points = getPositionDataShort();
    pca = PCA(n_components=3)
    pca.fit(points)
    return pca.components_

def findWaveWidthsSimple(matrix, threshold=0.5, dt=1,amp_option='both'):
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

            widths = LFPy_util.data_extraction.findWaveWidthsSimple(LFP)
    """
    matrix = np.array(matrix)
    if len(matrix.shape) == 1:
        matrix = np.reshape(matrix, (1,-1))
    widths = np.zeros(matrix.shape[0])
    trace = np.empty(matrix.shape)
    trace[:] = np.NAN
    for row in xrange(matrix.shape[0]):
        signal = matrix[row,:]
        
        # Flip the signal if the negative side should be used. 
        if amp_option == 'neg':
            signal = -signal
        elif amp_option == 'both' and signal.max() > -signal.min():
            signal = -signal
        offset = signal[0]
        signal -= offset

        amp_max = np.max(signal)
        thresh_abs = amp_max*np.fabs(threshold)
        for i in xrange(matrix.shape[1]) :
            if signal[i] >= thresh_abs :
                widths[row] += 1
                trace[row,i] = thresh_abs + offset
        if amp_option == 'neg':
            trace[row] = -trace[row]
        elif amp_option == 'both' and signal.max() > -signal.min():
            trace[row] = -trace[row]
    # If the input matrix is a single signal, return the trace 
    # as a vector and not a matrix. 
    if matrix.shape[0] == 1:
        trace = trace.flatten()
    return widths*dt, trace

def findWaveWidths(matrix, threshold=0.5, dt=1):
    """
    Compute wave width at some fraction of max amplitude. Counts the number
    of indices above threshold and multiplies by **dt**. Begins counting indices
    from the first value above threshold and ends at the first value below.

    :param `~numpy.ndarray` matrix: 
        Matrix (nSignals x frames) of 1D signals at each row. . 
        What happens now. Is this to long and will go over two lines
        or does it linebreak.
    :param float threshold: 
        Between 0 and 1.
    :param float dt: 
        Time per frame.
    :returns: 
        Array (nSignals) with widths.
    """
    # Widths at negative side.
    widths_pos = np.zeros(matrix.shape[0])
    for row in xrange(matrix.shape[0]):
        amp_min = np.min(matrix[row,:])
        thresh_abs = amp_min*threshold
        in_spike = False
        for i in xrange(matrix.shape[1]) :
            if matrix[row,i] <= thresh_abs :
                widths_pos[row] += 1
                in_spike = True
            else :
                if (in_spike == True) :
                    break
    # Widths at positive side.
    widths_neg = np.zeros(matrix.shape[0])
    for row in xrange(matrix.shape[0]):
        amp_max = np.max(matrix[row,:])
        thresh_abs = amp_max*threshold
        in_spike = False
        for i in xrange(matrix.shape[1]) :
            if matrix[row,i] <= thresh_abs :
                widths_neg[row] += 1
                in_spike = True
            else :
                if (in_spike == True) :
                    break
    widths = np.maximum(widths_neg,widths_pos)
    return widths*dt


def findAmplitudeSimple(matrix,amp_option='both'):
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
    if matrix.ndim == 1 :
        matrix = matrix.reshape([1,-1])
    amp_low = np.zeros(matrix.shape[0])
    if amp_option == 'both' or amp_option == 'neg' :
        for row in xrange(matrix.shape[0]):
            amp_low[row] = np.abs(np.min(matrix[row,:]))
    amp_high = np.zeros(matrix.shape[0])
    if amp_option == 'both' or amp_option == 'pos' :
        for row in xrange(matrix.shape[0]):
            amp_high[row] = np.abs(np.max(matrix[row,:]))
    amp = np.maximum(amp_low,amp_high)
    return amp
