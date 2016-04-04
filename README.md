# Sinusoidal-modelling-for-Ecoacoustics

This python code use a sinusoidal modelling algorithm to compute some indices in the context of ecoacoustic.
It can output:
 * a csv files containing the indices
 * a pickle file containing the values of all the tracks
 * a plot of the sinusoidal modelling
 
# Remark: Some audio files have been computed in Synth_audio_from_pkl from the outputed plk files, and an adaptation of the [sms-tools](https://github.com/MTG/sms-tools)  


## Prerequisites

 * [Numpy](http://www.numpy.org/)
 * [Scipy](http://www.scipy.org/)
 * [Matlplotlib](http://matplotlib.org/) (for graphing)
 * os
 * csv, pickle (to write output files)
 
 ## Usage

$python main.py