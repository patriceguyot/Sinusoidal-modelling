# Sinusoidal-modelling-for-Ecoacoustics

This python code intend to illustrate a submission to a scientific paper. It uses a sinusoidal modelling algorithm to compute some audio indices in the context of ecoacoustic.

It can output:
 * a csv files containing the indices
 * a pickle file containing the values of all the tracks
 * a plot of the sinusoidal modelling

The code show a version of a tool in development. Further improvements in terms of computational cost and precision of the results could be added in teh future. 

 
Remark: Some synthesized audio files have been computed (in the "Synth_audio_from_pkl" folder) from the outputted plk files, and an adaptation of the [sms-tools](https://github.com/MTG/sms-tools) (that is not available in this repository). 


## Prerequisites

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

Patrice Guyot, based on a source code from:
    * M. Le Coz, J. Pinquier, and R. André́-Obrecht, “Superposed speech localisation using frequency tracking,” in INTERSPEECH 2013 – 13th Annual Conference of the International Speech Communication Association, August 25-29, Lyon, France, Proceedings, 2013.
    
All comments and remarks by emails will be appreciated.   
    

