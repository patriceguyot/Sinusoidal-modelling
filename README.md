# Sinusoidal-modelling-for-Ecoacoustics

This python code intend to illustrate the submission of a scientific paper. It uses a sinusoidal modelling algorithm to compute some audio indices in the context of Ecoacoustic.

It can output:
 * a csv files containing the indices
 * a pickle file containing the values of all the tracks
 * a plot of the sinusoidal modelling

This code shows a version of a tool in development. Further improvements could be added in the future, in terms of computational cost, precision of the results, and documentation. 

 
Remark: Some synthesized audio files have been computed (in the "Synth_audio_from_pkl" folder) from the outputted plk files, and an adaptation of the [sms-tools](https://github.com/MTG/sms-tools) (that is not available in this repository). 


## Prerequisites

This code is based on python (tested with python 2.7.8).

Some python package required:

 * [Numpy](http://www.numpy.org/)
 * [Scipy](http://www.scipy.org/)
 * [Matlplotlib](http://matplotlib.org/) (for graphing)
 * os
 * csv, pickle (to write output files)
 
## Usage

$python main.py


## Licence

All Rights Reserved.

## Author

Patrice Guyot

(Adapted from:
    * M. Le Coz, J. Pinquier, and R. André́-Obrecht, “Superposed speech localisation using frequency tracking,” in INTERSPEECH 2013 – 13th Annual Conference of the International Speech Communication Association, August 25-29, Lyon, France, Proceedings, 2013.)
    
Questions, comments and remarks by emails will be appreciated.   
    
Credits: Patrice Guyot, Alice Eldridge, Mika Peck
