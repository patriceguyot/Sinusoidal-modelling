#!/usr/bin/env python

"""
    Compute and output sine modelling
"""

__author__ = "Patrice Guyot"
__version__ = "0.1"




from numpy import log2, log10, sqrt, mean, std
from scipy.io.wavfile import read as wavread
from scipy import signal
import os
import numpy as np
import csv
import argparse
import matplotlib.pyplot as plt
from scipy.stats import chisquare, poisson





#----------------------------------------------------------------------------
def pcm2float(sig, dtype='float64'):
    """Convert PCM signal to floating point with a range from -1 to 1.

    Use dtype='float32' for single precision.

    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.

    Returns
    -------
    numpy.ndarray
        Normalized floating point data.

    See Also
    --------
    float2pcm, dtype

    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max



#-------------------------------------------------------------------------
def get_spectrogram(samples, sr, max_freq=None):
    """

    :param samples:
    :param sr:
    :return:
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




#-------------------------------------------------------------------------------------------------------------
def get_trackings(samples, sr, w_length=0.032, w_step=0.016, low_freq=0.0, high_freq=None, n_bins=2048, n_peaks=5,
                  min_len=5, windowing_function='hamming', tani_cp = 3.0, tani_cf=100.0, threshold_dB=-25, threshold_list=None):
    """

    n_bins: number of bin in the fft. (If longuer than the window -> zero padding)
    :param samples:
    :param sr:
    :return:
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
        # TODO Ajouter la fonction sur l'amplitude


        if threshold_dB==None:
            peaks = sorted([Node(frequency_line[low_freq_id+i], spectrum[i], time)
                        for i in range(1, len(spectrum)-1) if (spectrum[i-1] < spectrum[i] > spectrum[i+1]) & (spectrum_dB[i] > threshold_list[i])],
                       key=lambda x: x.amplitude, reverse=True)[:n_peaks]  #Pat -> choisit les n_peaks plus grands pics d'amplitude dans la trame ?
        else:

            peaks = sorted([Node(frequency_line[low_freq_id+i], spectrum[i], time)
                        for i in range(1, len(spectrum)-1) if (spectrum[i-1] < spectrum[i] > spectrum[i+1]) & (spectrum_dB[i] > threshold_dB)],
                       key=lambda x: x.amplitude, reverse=True)[:n_peaks]  #Pat -> choisit les n_peaks plus grands pics d'amplitude dans la trame ?


            #peaks = sorted([Node(frequency_line[low_freq_id+i], spectrum[i], time)
                        #for i in range(1, len(spectrum)-1) if (np.all(spectrum[i-4:i-2] < spectrum[i-1:i+1] > spectrum[i+2:i+4])) & (spectrum_dB[i] > threshold_dB)],
                        #key=lambda x: x.amplitude, reverse=True)[:n_peaks]  #Pat -> choisit les n_peaks plus grands pics d'amplitude dans la trame ?

            #peaks = sorted([Node(frequency_line[low_freq_id+i], spectrum[i], time)
                        #for i in range(4, len(spectrum)-4) if (np.mean(spectrum[i-4:i-2]) < np.mean(spectrum[i-1:i+1]) > np.mean(spectrum[i+2:i+4])) & (spectrum_dB[i] > threshold_dB)],
                        #key=lambda x: x.amplitude, reverse=True)[:n_peaks]  #Pat -> choisit les n_peaks plus grands pics d'amplitude dans la trame ?

        # Pat -> ajout d'une condition sur l'energie minimum & spectrum_dB[i] > threshold_dB

        # Continuer les trackings avec les pics actifs
        for a in active_trackings:

            continue_loop = True

            i = 0

            while i < len(peaks) and continue_loop:

                if a.add_node(peaks[i]):
                    # On peut ajouter a au tracking
                    peaks.remove(peaks[i])

                    continue_loop = False

                else:

                    i += 1

            if continue_loop:

                if len(a.nodes) <= min_len:

                    trackings.remove(a)

                else:

                    a.active = False

        # Debuter les tracking
        trackings += [Tracking(p) for p in peaks]

    return trackings


#----------------------------------------------------------------------------------------------------------------------------------------------------



class Node(object):

    tani_cf=100.0
    tani_cp=3.0

    def __init__(self, frequency, amplitude, time):

        self.frequency = frequency
        self.cent_freq = 1200*log2(self.frequency/(440*2**(3/11-5)))
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
        return sqrt(((self.cent_freq-node.cent_freq)/Node.tani_cf)**2 +
                    ((log10(self.amplitude) - log10(node.amplitude))/Node.tani_cp) ** 2)

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
        return mean([n.frequency for n in self.nodes])

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




#--------------------------------------------------------------------------------------------
def mean_spectro(samples, sr, w_length=0.032, w_step=0.016, low_freq=0.0, high_freq=None, n_bins=1024, windowing_function='hamming'):
    """
    Return the mean of the spectro
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
    s1= np.median(s,axis=0)
    return 20*np.log10(s1)
    #return np.mean(20*np.log10(spectrogram))



#--------------------------------------------------------------------------------------------
# statistics about the distribution of the peaks
def stats_peak(trackings, time_begin, time_end, frequency_low, frequency_high, nb_f_quadrant = 10, nb_t_quadrant=10):
    peaks = [list(t.nodes) for t in trackings]
    peaks2 = [item for sublist in peaks for item in sublist] #flatten the list of list to list
    peaks_centered = sorted([ [p.time - time_begin, p.frequency - frequency_low] for p in peaks2])

    f_quandrant_size = (frequency_high - frequency_low)/float(nb_f_quadrant)
    t_quandrant_size = (time_end - time_begin)/float(nb_t_quadrant)

    quadrants= sorted([ [int(t/t_quandrant_size), int(f/f_quandrant_size)] for t,f in peaks_centered])
    quadrants_values = [[t,f] for t in range(nb_t_quadrant) for f in range(nb_f_quadrant)]
    quadrant_distribution = [[x,quadrants.count(x)] for x in quadrants_values] #-> [[coordinate of the quadrat, nb_values]...]

    ratio_quadrant_with_peaks = len([b for [_,b] in quadrant_distribution if b>0])/float(len(quadrant_distribution))
    D = len(peaks_centered)/float(len(quadrants_values))

    #nb_peak_max = np.max([b for [_,b] in quadrant_distribution])

    values = sorted([[number_peaks, [b for [_,b] in quadrant_distribution].count(number_peaks)] for number_peaks in set([b for [_,b] in quadrant_distribution])]) # -> [[nb_values, nb_quadrats]...]
    #values = [[number_peaks, [b for [_,b] in quadrant_distribution].count(number_peaks)] for number_peaks in range(len(peaks_centered))]

    IC = (np.sum( [K*(n - D)**2 for [n,K] in values]) / (float(len(quadrants_values)-1))/D)

    print '\nNumber of peaks:', len(peaks_centered)
    print 'Average theorical number of peak per quadrat (D):', D
    print 'quadrant_distribution:', quadrant_distribution

    print 'values:', values
    print 'IC:', IC


    #poisson_values_old = [[n, np.exp(-D) * len(quadrants_values) * (D ** n) / np.math.factorial(n)] for n in [x for [x,_] in values]]
    #chi2_old = [ (O-E)**2/E for O,E in zip([x for [_, x] in values],[x for [_, x] in poisson_values_old])]

    #print 'poisson values_old:', poisson_values_old
    #print 'chisq_old:', chi2_old
    #print 'chisq_old_sim:', np.sum(chi2_old)

    poisson_values = list(poisson.pmf([x for [x,_] in values], D)*len(quadrants_values))
    chi2 = chisquare([x for [_, x] in values], poisson_values)


    chisq = chi2[0]
    p_value = chi2[1]

    print 'poisson values:', poisson_values
    print 'chisq:', chisq
    print 'p_value', p_value


    return IC, len(peaks_centered), chisq, p_value


##---------------------------------- MAIN --------------------------------------------------------------------------

if __name__ == '__main__':


    #file_path='audio/extrait_BALMER-14_0_20150619_0645.wav'

    #


    #file_path='/Users/guyot/Desktop/Audio_UK_listening/UK_ID_named/PL-05_0_20150605_0430.wav'
    #file_path='/Users/guyot/Desktop/Audio_UK_listening/UK_ID_named/KNEPP-01_0_20150510_0715.wav'
    #'

    file_path='/Users/guyot/Desktop/Audio_UK_listening/UK_ID_named/BALMER-01_0_20150619_0445.wav'
    #file_path='/Users/guyot/Desktop/Audio_UK_listening/UK_ID_named/BALMER-01_0_20150619_0346.wav'
    #file_path='/Users/guyot/Desktop/Audio_UK_listening/UK_ID_named/PL-15_0_20150605_0330.wav'
    file_path='/Users/guyot/Desktop/Audio_UK_listening/UK_ID_named/BALMER-01_0_20150621_0317.wav'
    #file_path='audio/test.wav'
    #file_path='audio/KNEPP-11_0_20150511_0545.wav'
    #file_path='audio/PL-03_0_20150605_0500.wav'

    #file_path='/Users/guyot/Desktop/Audio_UK_listening/UK_ID_named/PL-11_0_20150603_0645.wav'
    #file_path='/Users/guyot/Desktop/Audio_UK_listening/UK_ID_named/KNEPP-10_0_20150510_0445.wav'

    output_csv_file = 'results.csv'


    output = dict()
    output['path']= file_path
    output['filename']= os.path.basename(file_path)

    sr, sig = wavread(file_path)
    samples = pcm2float(sig,dtype='float64')

    # Partial Tracking ------------------------------
    print 'Read the audio file:', file_path

    low_freq=0.0
    high_freq=10000
    n_bins=2048

    low_freq_id = int(max([low_freq/(sr/2)*n_bins, 1]))
    high_freq_id = int(float(high_freq)/(sr/2)*n_bins)



    print 'Partial tracking'


    m = mean_spectro(samples, sr, low_freq=low_freq, high_freq=high_freq, n_bins=n_bins)
    print 'mean of the spectrum :', np.mean(m)
    #threshold_dB = np.mean(m) + 12
    #threshold_dB = np.mean(m) + (250 * -1/np.mean(m))
    threshold_dB = np.mean(m) + 10
    threshold_list = m + 9
    print 'threshold_dB  :', threshold_dB

    low_freq=1600.0
    high_freq=8000
    tani_cp = 1.0
    tani_cf = 200.0
    #trackings = get_trackings(samples, sr, low_freq=1200.0, high_freq=10000, n_peaks=10, min_len=10, threshold_dB=threshold_dB) # Patrice changed that
    #trackings = get_trackings(samples, sr, low_freq=low_freq, high_freq=high_freq, n_peaks=20, min_len=8, threshold_dB=None, threshold_list=threshold_list) # Patrice changed that
    #trackings = get_trackings(samples, sr, low_freq=low_freq, high_freq=high_freq, n_peaks=5, min_len=5, tani_cp=tani_cp, tani_cf=tani_cf, threshold_dB=threshold_dB)


    #trackings = get_trackings(samples, sr, low_freq=low_freq, high_freq=high_freq, n_peaks=5, min_len=10, tani_cp=tani_cp, tani_cf=tani_cf, threshold_dB=threshold_dB)
    #trackings = get_trackings(samples, sr, low_freq=low_freq, high_freq=high_freq, n_peaks=5, min_len=10, tani_cp=tani_cp, tani_cf=tani_cf,threshold_dB=None, threshold_list=threshold_list)
    trackings = get_trackings(samples, sr, low_freq=low_freq, high_freq=high_freq, n_peaks=5, min_len=10, tani_cp=tani_cp, tani_cf=tani_cf, threshold_dB=threshold_dB)

    # Results -----------------------------------------

    partials_number = len(trackings)
    print "Nombre de partiels :", partials_number

    if partials_number>0:
        IC, nb_peaks, chisq, p_value = stats_peak(trackings, time_begin=0, time_end=len(sig)/float(sr), frequency_low=1600.0, frequency_high=8000, nb_f_quadrant = 12, nb_t_quadrant=30)#->version9
    else:
        IC=1
        nb_peaks=0
        chisq=0
        p_value=1
    #trackings = get_trackings(samples, sr, low_freq=low_freq, high_freq=high_freq, n_peaks=20, min_len=8, threshold_dB=None, threshold_list=threshold_list) # Patrice changed that

    # plots ------------------------------------------------
    plot_spectro = 1
    if plot_spectro:
        img, ext = get_spectrogram(samples, sr, max_freq=9000.0)
        plt.imshow(np.log10(img), origin="lower", aspect="auto", extent=ext)

        for t in trackings:
            nodes = sorted(t.nodes, key=lambda x: x.time)
            plt.plot([n.time for n in nodes], [n.frequency for n in nodes], "k")
        plt.show()


    exit
    # calcul des recouvrements
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

    print "removed partials :", removed_partials
    partials_number = len(trackings)

    plot_overlap=0
    if plot_overlap:
        plt.plot(overlaps)
        plt.show()

    # plots ------------------------------------------------
    plot_spectro = 0
    if plot_spectro:
        img, ext = get_spectrogram(samples, sr, max_freq=9000.0)
        plt.imshow(np.log10(img), origin="lower", aspect="auto", extent=ext)

        for t in trackings:
            nodes = sorted(t.nodes, key=lambda x: x.time)
            plt.plot([n.time for n in nodes], [n.frequency for n in nodes], "k")
        plt.show()







    if partials_number > 0:
        centroids = [t.get_centroid() for t in trackings] #pat
        starts = [t.start for t in trackings] #pat
        centroids_std = std(np.array(centroids))
        starts_std = std(np.array(starts))
    else:
        centroids_std = 0
        starts_std = 0

    print "STD des centroids :", centroids_std
    print "STD des debuts :", starts_std
    print "Nombre de partiels :", partials_number

    output['partials_number']= partials_number
    output['centroid_std']= centroids_std
    output['starts_std']= starts_std

    output['IC'] = IC
    output['nb_peaks'] = nb_peaks
    output['chisq'] = chisq
    output['p_value'] = p_value

    # Write csv file
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
