# Sinusoidal-modelling-for-Ecoacoustics

This python code intend to illustrate a scientific paper: 

Guyot, P., Eldridge, A., Eyre-Walker, Y. C., Johnston, A., Pellegrini, T., & Peck, M. *Sinusoidal modelling for ecoacoustics*. In Annual conference Interspeech (INTERSPEECH 2016, pp-2602-2606. [link](https://hal.archives-ouvertes.fr/hal-01474894/document)

 It uses a sinusoidal modelling algorithm to compute some audio indices in the context of Ecoacoustic.

It can output:
 * a csv files containing the indices,
 * a pickle file containing the values of all the tracks,
 * a plot of the sinusoidal modelling.

This code shows a version of a tool in development. Further improvements could be added in the future, in terms of computational cost, precision of the results, and documentation. 

 
Remark: some synthesized audio files have been computed (in the "Synth_audio_from_pkl" folder) from the outputted plk files, using an adaptation of the [sms-tools](https://github.com/MTG/sms-tools) for the synthesis (that is not currently available in this repository). 


## Prerequisites

This code is based on python (tested with python 2.7.8).

Some python package are required:

 * [Numpy](http://www.numpy.org/)
 * [Scipy](http://www.scipy.org/)
 * [Matlplotlib](http://matplotlib.org/) (for graphic)
 * csv, pickle (to write output files)
 * os
 
## Usage

$python main.py


## Licence

[GPL v3] (Sinusoidal-modelling-for-Ecoacoustics/LICENSE.txt)


## Authors

Patrice Guyot
(code adapted from: M. Le Coz, J. Pinquier, and R. André-Obrecht, “Superposed speech localisation using frequency tracking,” in INTERSPEECH 2013 – 13th Annual Conference of the International Speech Communication Association, August 25-29, Lyon, France, Proceedings, 2013.)
    
Questions, comments and remarks (by emails) would be appreciated.   
    
Credits: Patrice Guyot, Alice Eldridge, Mika Peck
