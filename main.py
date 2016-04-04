#!/usr/bin/env python

"""
    Compute sine modelling
"""

__author__ = "Patrice Guyot"
__version__ = "0.1"
__credits__ = ["Patrice Guyot", "Alice Eldridge", "Mika Peck"]
__email__ = ["guyot.patrice@gmail.com", "alicee@sussex.ac.uk", "m.r.peck@sussex.ac.uk"]
__status__ = "Development"



from scipy.io.wavfile import read as wavread
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import csv
from os import path
import pickle



#----------------------------------------------------------------------------
#                           Signal function
#----------------------------------------------------------------------------
def pcm2float(samples, dtype='float64'):
    """Convert PCM signal to floating point with a range from -1 to 1.

    Use dtype='float32' for single precision.

    Parameters
    ----------
    samples : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.

    Returns
    -------
    numpy.ndarray
        Normalized floating point data.


    """
    sig = np.asarray(samples)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max



#----------------------------------------------------------------------------
#                           Spectrogram functions
#----------------------------------------------------------------------------
def get_spectrogram(samples, sr, max_freq=None):
    """

    This function output the spectrogram of the signal. Used for plot only.

    :param samples: samples of the audio signal
    :param sr: sampling rate

    :return: spectrogram of the data
    """
    from numpy.fft import rfft
    from numpy import linspace
    from numpy import hamming

    if max_freq is None:

        max_freq = sr/2

    w = 256

    imax = max_freq/(sr/2) * 1024
    win = hamming(2*w)
    time_line = [t for t in range(w, len(samples)-w, w)]
    freq_line = linspace(1, sr/2, 1024)
    freq_line = freq_line[:imax]
    frames = [win*samples[t-w:t+w] for t in time_line]
    return map(list, zip(*map(lambda x: map(abs, rfft(x, 2048)[:imax]), frames))),\
           [float(time_line[0])/sr, float(time_line[-1])/sr, freq_line[0], freq_line[-1]]





#--------------------------------------------------------------------------------------------
def mean_spectro(samples, sr, w_length=0.032, w_step=0.016, low_freq=0.0, high_freq=None, n_bins=1024, windowing_function='hamming'):
    """
    This function output the mean of the spectrogram as computed in the "get_trackings" function.

    :param samples: samples of the audio signal
    :param sr: sampling rate
    :param w_length: window length for the fft (in seconds)
    :param w_step: hop size for the fft (in seconds)
    :param n_bins: size of the fft (in bins)
    :param windowing_function: type of window used for the windowing in the fft

    :return: mean value of each frequency row of the spectrogram (in dB)

    """

    from numpy import fft


    demi = int(w_length*sr)

    time_line_samples = range(demi, len(samples), int(w_step*sr))
    time_line = map(lambda x: float(x)/sr, time_line_samples)
    frames = [samples[t-demi:t+demi] for t in time_line_samples]
    low_freq_id = max([low_freq/(sr/2)*n_bins, 1])
    if high_freq is None:
        high_freq = sr/2
    high_freq_id = int(float(high_freq)/(sr/2)*n_bins)
    spectrogram = []

    for time, current_frame in zip(time_line, frames):



        if windowing_function is not None:
            W = signal.get_window(windowing_function, len(current_frame), fftbins=False)
            current_frame = current_frame * W

        spectrum = map(abs, fft.rfft(current_frame, n=2*n_bins)[low_freq_id:high_freq_id])
        spectrogram += [spectrum]


    s = np.array(spectrogram)
    s1= np.mean(s,axis=0)
    return 20*np.log10(s1)





#----------------------------------------------------------------------------
#                           Partial tracking functions
#----------------------------------------------------------------------------
class Node(object):

    tani_cf=100.0
    tani_cp=3.0

    def __init__(self, frequency, amplitude, time):

        self.frequency = frequency
        self.cent_freq = 1200*np.log2(self.frequency/(440*2**(3/11-5)))
        self.amplitude = amplitude
        self.tracking = None
        self.time = time

    def link(self, other):
        if self.tracking is None:
            self.tracking = Tracking(self)
        self.tracking.add_node(other)
        return self.tracking

    def __str__(self):
        return "%.2f : %d " % (self.time, self.frequency)

    def __repr__(self):
        return "(%s)" % self

    def tani_dist(self, node):
        return np.sqrt(((self.cent_freq-node.cent_freq)/Node.tani_cf)**2 +
                    ((np.log10(self.amplitude) - np.log10(node.amplitude))/Node.tani_cp) ** 2)


#----------------------------------------------------------------------------
class Tracking(object):

    cmpt = 0

    def __init__(self, node=None):
        self.nodes = set()
        self.nodes.add(node)
        self.start = node.time
        self.stop = node.time
        self.centroid = None
        self.last_node = node
        self.active = True
        self.id = Tracking.cmpt
        Tracking.cmpt +=1

    def __repr__(self):
        return "Tracking %d" % self.id

    def get_centroid(self):
        return np.mean([n.frequency for n in self.nodes])

    def get_node_at(self, time):

        for n in self.nodes :
            if n.time == time :
                return n

        return None

    def add_node(self, node, tani_th=1):

        if self.last_node.tani_dist(node) < tani_th:

            self.nodes.add(node)
            self.stop = node.time
            self.last_node = node

            return True

        else:

            return False

    def intersect(self, other):
        """
        Return the list of tuples corresponding to the intersection (highter node , lower node)
        """
        if other.get_centroid():
            return [(o, m) for m in self.nodes for o in other.nodes if o.time == m.time]
        else:
            return [(m, o) for m in self.nodes for o in other.nodes if o.time == m.time]

    def harmo_link(self, others, min_overlap_frames=3, var_max=0.008):

        linkables = []
        for other in others:

            if other is not self:

                simul_nodes = self.intersect(other)

                if len(simul_nodes) > min_overlap_frames:
                    """
                    ratios = [a.frequency/b.frequency if a.frequency > b.frequency else b.frequency/a.frequency
                              for a, b in simul_nodes ]

                    magnitude = round(mean(ratios))

                    if magnitude > 1 and std([abs(r-magnitude) for r in ratios]) < var_max:
                        linkables += [other]
                    """
                    linkables += [other]

        return linkables

    def get_portion(self, start, stop):

        return [n.frequency for n in sorted(self.nodes, key=lambda x:x.time) if start <= n.time <= stop]




#-------------------------------------------------------------------------------------------------------------
def get_trackings(samples, sr, w_length=0.032, w_step=0.016, low_freq=0.0, high_freq=None, n_bins=2048, n_peaks=5,
                  min_len=5, windowing_function='hamming', tani_cp = 3.0, tani_cf=100.0, threshold_dB=-25):
    """

    get_tracking creates Nodes and Tracks form an audio file.

    :param samples: samples of the audio signal
    :param sr: sampling rate
    :param w_length: window length for the fft (in seconds)
    :param w_step: hop size for the fft (in seconds)
    :param n_bins: size of the fft (in bins)
    :param windowing_function: type of window used for the windowing in the fft

    :param low_freq: minimum frequency to consider in the tracking (in Hertz)
    :param high_freq: maximum frequency to consider in the tracking (in Hertz)
    :param n_peaks: maximum number of peaks by frame
    :param min_lens: minimum length of the track (in frame)
    :param tani_cp: Amplitude distance between two peaks (in dB)
    :param tani_cf: Frequency distance between two peaks (in cents)
    :param threshold_dB: Amplitude threshold to consider peaks (in dB)


    :return: a list of instances of the object Tracking
    """

    from numpy import fft, linspace
    tani_ratio = 0.032 / w_length

    Node.tani_cf = tani_cf / tani_ratio
    Node.tani_cp = tani_cp / tani_ratio

    trackings = []
    demi = int(w_length*sr)
    frequency_line = linspace(0, sr/2, n_bins)
    time_line_samples = range(demi, len(samples), int(w_step*sr))
    time_line = map(lambda x: float(x)/sr, time_line_samples)
    frames = [samples[t-demi:t+demi] for t in time_line_samples]
    low_freq_id = max([low_freq/(sr/2)*n_bins, 1])
    if high_freq is None:
        high_freq = sr/2
    high_freq_id = int(float(high_freq)/(sr/2)*n_bins)
    spectrogram = []

    for time, current_frame in zip(time_line, frames):

        active_trackings = [t for t in trackings if t.active]

        if windowing_function is not None:
            W = signal.get_window(windowing_function, len(current_frame), fftbins=False)
            current_frame = current_frame * W

        spectrum = map(abs, fft.rfft(current_frame, n=2*n_bins)[low_freq_id:high_freq_id])
        spectrum_dB = 20*np.log10(spectrum) # Pat
        spectrogram += [spectrum]


        peaks = sorted([Node(frequency_line[low_freq_id+i], spectrum[i], time)
                        for i in range(1, len(spectrum)-1) if (spectrum[i-1] < spectrum[i] > spectrum[i+1]) & (spectrum_dB[i] > threshold_dB)],
                       key=lambda x: x.amplitude, reverse=True)[:n_peaks]

        for a in active_trackings:

            continue_loop = True

            i = 0

            while i < len(peaks) and continue_loop:

                if a.add_node(peaks[i]):

                    peaks.remove(peaks[i])

                    continue_loop = False

                else:

                    i += 1

            if continue_loop:

                if len(a.nodes) <= min_len:

                    trackings.remove(a)

                else:

                    a.active = False

        # Start tracking
        trackings += [Tracking(p) for p in peaks]

    return trackings



#----------------------------------------------------------------------------
#                           Stats from tracking function
#----------------------------------------------------------------------------
def stats_peak(trackings, time_begin, time_end, frequency_low, frequency_high, nb_f_quadrat = 10, nb_t_quadrat=10):



    """

    stats_peak output stats about the distribution of the peaks

    :param trackings: a list of instances of the object Tracking
    :param time_begin: start time to consider in the analysis (in seconds)
    :param time_end: end time to consider in the analysis (in seconds)
    :param frequency_low: minimum frequency to consider in the analysis (in Hertz)
    :param frequency_high: maximum frequency to consider in the analysis (in Hertz)
    :param nb_f_quadrat: number of frequency quadrat to consider
    :param nb_t_quadrat: number of time quadrat to consider




    :return ratio_quadrat_with_peaks: number of quadrat with peaks /  number of quadrat without peaks
    :return ci: Concentration Index
    :return len(peaks_centered): number of peaks in tracks
    """



    peaks = [list(t.nodes) for t in trackings]
    peaks2 = [item for sublist in peaks for item in sublist] #flatten the list of list to list
    peaks_centered = sorted([ [p.time - time_begin, p.frequency - frequency_low] for p in peaks2])

    f_quandrant_size = (frequency_high - frequency_low)/float(nb_f_quadrat)
    t_quandrant_size = (time_end - time_begin)/float(nb_t_quadrat)

    quadrats= sorted([ [int(t/t_quandrant_size), int(f/f_quandrant_size)] for t,f in peaks_centered])
    quadrats_values = [[t,f] for t in range(nb_t_quadrat) for f in range(nb_f_quadrat)]
    quadrat_distribution = [[x,quadrats.count(x)] for x in quadrats_values] #-> [[coordinate of the quadrat, nb_values]...]

    D = len(peaks_centered)/float(len(quadrats_values))

    values = sorted([[number_peaks, [b for [_,b] in quadrat_distribution].count(number_peaks)] for number_peaks in set([b for [_,b] in quadrat_distribution])]) # -> [[nb_values, nb_quadrats]...]


    ci = (np.sum( [K*(n - D)**2 for [n,K] in values]) / (float(len(quadrats_values)-1))/D)

    ratio_quadrat_with_peaks = len([b for [_,b] in quadrat_distribution if b>0])/float(len(quadrat_distribution))

    return ratio_quadrat_with_peaks, ci, len(peaks_centered)




#----------------------------------------------------------------------------
#                                       Main
#----------------------------------------------------------------------------
if __name__ == '__main__':


    # Audio files
    file_path='audio/BALMER-01_0_20150621_0317.wav'
    #file_path='audio/PL-11_0_20150603_0645.wav'
    #file_path='audio/KNEPP-10_0_20150510_0445.wav'
    #file_path='audio/KNEPP-11_0_20150511_0545.wav'


    # Output csv file
    output_csv_file = 'results.csv'
    output = dict()
    output['path']= file_path
    output['filename']= path.basename(file_path)




    # Reading of the audio file
    print 'Read the audio file:', file_path
    sr, sig = wavread(file_path)
    samples = pcm2float(sig,dtype='float64')




    # Compute the average value of the spectrogram

    low_freq=0.0
    high_freq=10000
    n_bins=2048

    low_freq_id = int(max([low_freq/(sr/2)*n_bins, 1]))
    high_freq_id = int(float(high_freq)/(sr/2)*n_bins)

    m = mean_spectro(samples, sr, low_freq=low_freq, high_freq=high_freq, n_bins=n_bins)
    print 'Mean of the spectrum:', np.mean(m), 'dB'
    threshold_dB = np.mean(m) + 9


    # Partial Tracking
    print '- Partial tracking'

    low_freq=1600.0
    high_freq=8000.0
    tani_cp = 1.0
    tani_cf = 200.0
    trackings = get_trackings(samples, sr, low_freq=low_freq, high_freq=high_freq, n_peaks=5, min_len=10, tani_cp=tani_cp, tani_cf=tani_cf, threshold_dB=threshold_dB)

    # Result analysis
    partials_number = len(trackings)
    if partials_number>0:
        ratio_quadrat_with_peaks, ci, nb_peaks = stats_peak(trackings, time_begin=0, time_end=len(sig)/float(sr), frequency_low=low_freq, frequency_high=high_freq, nb_f_quadrat= 12, nb_t_quadrat=60)#->version9
    else:
        ratio_quadrat_with_peaks=0
        IC=1
        nb_peaks=0


    print '- Stats - '
    print "Number of tracks:", partials_number
    print 'Number of peaks:', nb_peaks
    print 'ci:', ci
    print 'Ratio peaks in quadrats:', ratio_quadrat_with_peaks

    # plot spectrogram and tracks -------------------------------------------
    plot_spectro = 1
    if plot_spectro:
        img, ext = get_spectrogram(samples, sr, max_freq=9000.0)
        plt.imshow(np.log10(img), origin="lower", aspect="auto", extent=ext)

        for t in trackings:
            nodes = sorted(t.nodes, key=lambda x: x.time)
            plt.plot([n.time for n in nodes], [n.frequency for n in nodes], "k")
        plt.show()


    # Write a pkl file  -------------------------------------
    write_pkl = 0
    if write_pkl:
        output_pickle_file='output_pkl/' + path.splitext(path.basename(file_path))[0] +'.pkl'
        print '- Write tracks in:', output_pickle_file
        # Create a list of partials
        list_partials=[]
        for t in trackings:
            list_nodes=[]
            for n in t.nodes:
                list_nodes.append([n.time, n.frequency, 20*np.log10(n.amplitude)])
            list_partials.append(list_nodes)

        # Write it with pickle
        with open(output_pickle_file, 'wb') as f:
            pickle.dump(list_partials, f)


    # Remove superposed tracks -------------------------------------
    print "- Removing superposed tracks:"
    w_length=0.032
    w_step=0.016
    time_line_samples = range(int(w_length*sr), len(samples), int(w_step*sr))
    overlaps = np.zeros(time_line_samples[-1])



    removed_partials = 0
    if partials_number > 0:
         starts = [t.start for t in trackings] #pat
         stops = [t.stop for t in trackings] #pat
         for start, stop in zip(starts,stops):
             overlaps[start*sr:stop*sr] += 1
             if np.all(overlaps[start*sr:stop*sr]>1):
                 tracks = [x for x in trackings if x.start == start]
                 trackings.remove(tracks[0])
                 removed_partials +=1
                 overlaps[start*sr:stop*sr] -= 1

    print "Removed tracks:", removed_partials
    partials_number_removed = len(trackings)
    print "Number of tracks (after removing):", partials_number_removed


    # Output a csv file -------------------------------------
    output['partials_number']= partials_number_removed
    output['ratio_quadrat'] = ratio_quadrat_with_peaks
    output['CI'] = ci
    output['nb_peaks'] = nb_peaks

    # Write the csv file
    print '- Write Indices in:', output_csv_file
    keys = []
    values = []
    writer = csv.writer(open(output_csv_file, 'wb'))
    for key, value in output.iteritems():
        keys.append(key)
        values.append(value)

    writer.writerow(keys)
    writer.writerow(values)

    exit
