# coding=utf-8

## readgssi.py
## intended to translate radar data from DZT to other more workable formats.
## DZT is a file format maintained by Geophysical Survey Systems Incorporated (GSSI).
## specifically, this script is intended for use with radar data recorded
## with GSSI SIR 3000 and 4000 field units. Other field unit models may record DZT slightly
## differently, in which case this script may need to be modified.

# readgssi was originally written as matlab code by
# Gabe Lewis, Dartmouth College Department of Earth Sciences.
# matlab code was adapted for python with permission by
# Ian Nesbitt, University of Maine School of Earth and Climate Sciences.
# Copyleft (c) 2017 Ian Nesbitt

# this code is freely available under the GNU Affero General Public License 3.0.
# if you did not receive a copy of the license upon obtaining this software, please visit
# (https://opensource.org/licenses/AGPL-3.0) to obtain a copy.

import sys, getopt, os
import numpy as np
from datetime import datetime, timedelta
from readgssi import config
from readgssi.constants import *
from readgssi.dzt import *


def printmsg(msg):
    """
    Prints with date/timestamp.

    :param str msg: Message to print
    """
    print('%s - %s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))

def readgssi(infile, outfile=None, verbose=False, antfreq=None, frmt='python',\
             plotting=False, figsize=7, dpi=150, stack=1, x='seconds',\
             z='nanoseconds', histogram=False, colormap='gray', colorbar=False,\
             zero=[None,None,None,None], gain=1, freqmin=None, freqmax=None, \
             reverse=False, bgr=False, win=0, dewow=False, absval=False,\
             normalize=False, specgram=False, noshow=False, spm=None,\
             start_scan=0, num_scans=-1, epsr=None, title=True, zoom=[0,0,0,0],\
             pausecorrect=False, showmarks=False):
    """
    This is the primary directive function. It coordinates calls to reading, filtering, translation, and plotting functions, and should be used as the overarching processing function in most cases.

    :param str infile: Input DZT data file
    :param str outfile: Base output file name for plots, CSVs, and other products. Defaults to :py:data:`None`, which will cause the output filename to take a form similar to the input. The default will let the file be named via the descriptive naming function :py:data:`readgssi.functions.naming()`.
    :param bool verbose: Whether or not to display (a lot of) information about the workings of the program. Defaults to :py:data:`False`. Can be helpful for debugging but also to see various header values and processes taking place.
    :param int antfreq: User setting for antenna frequency. Defaults to :py:data:`None`, which will cause the program to try to determine the frequency from the antenna name in the header of the input file. If the antenna name is not in the dictionary :py:data:`readgssi.constants.ANT`, the function will try to determine the frequency by decoding integers in the antenna name string.
    :param str frmt: The output format to be passed to :py:mod:`readgssi.translate`. Defaults to :py:data:`None`. Presently, this can be set to :py:data:`frmt='dzt'`, :py:data:`frmt='csv'`, :py:data:`'numpy'`, :py:data:`'gprpy'`, or :py:data:`'object'` (which will return the header dictionary, the image arrays, and the gps coordinates as objects). Plotting will not interfere with output (i.e. you can output to CSV and plot a PNG in the same command).
    :param bool plotting: Whether to plot the radargram using :py:func:`readgssi.plot.radargram`. Defaults to :py:data:`False`.
    :param int figsize: Plot size in inches to be passed to :py:func:`readgssi.plot.radargram`.
    :param int dpi: Dots per inch (DPI) for figure creation.
    :param int stack: Number of consecutive traces to stack (horizontally) using :py:func:`readgssi.arrayops.stack`. Defaults to 1 (no stacking). Especially good for handling long radar lines. Algorithm combines consecutive traces together using addition, which reduces noise and enhances signal. The more stacking is done, generally the clearer signal will become. The tradeoff is that you will reduce the length of the X-axis. Sometimes this is desirable (i.e. for long survey lines).
    :param str x: The units to display on the x-axis during plotting. Defaults to :py:data:`x='seconds'`. Acceptable values are :py:data:`x='distance'` (which sets to meters), :py:data:`'km'`, :py:data:`'m'`, :py:data:`'cm'`, :py:data:`'mm'`, :py:data:`'kilometers'`, :py:data:`'meters'`, etc., for distance; :py:data:`'seconds'`, :py:data:`'s'`, :py:data:`'temporal'` or :py:data:`'time'` for seconds, and :py:data:`'traces'`, :py:data:`'samples'`, :py:data:`'pulses'`, or :py:data:`'columns'` for traces.
    :param str z: The units to display on the z-axis during plotting. Defaults to :py:data:`z='nanoseconds'`. Acceptable values are :py:data:`z='depth'` (which sets to meters), :py:data:`'m'`, :py:data:`'cm'`, :py:data:`'mm'`, :py:data:`'meters'`, etc., for depth; :py:data:`'nanoseconds'`, :py:data:`'ns'`, :py:data:`'temporal'` or :py:data:`'time'` for seconds, and :py:data:`'samples'` or :py:data:`'rows'` for samples.
    :param bool histogram: Whether to plot a histogram of array values at plot time.
    :type colormap: :py:class:`str` or :class:`matplotlib.colors.Colormap`
    :param colormap: Plot using a Matplotlib colormap. Defaults to :py:data:`gray` which is colorblind-friendly and behaves similarly to the RADAN default, but :py:data:`seismic` is a favorite of many due to its diverging nature.
    :param bool colorbar: Whether to display a graded color bar at plot time.
    :param list[int,int,int,int] zero: A list of values representing the amount of samples to slice off each channel. Defaults to :py:data:`None` for all channels, which will end up being set by the :code:`rh_zero` variable in :py:func:`readgssi.dzt.readdzt`.
    :param int gain: The amount of gain applied to plots. Defaults to 1. Gain is applied as a ratio of the standard deviation of radargram values to the value set here.
    :param int freqmin: Minimum frequency value to feed to the vertical triangular FIR bandpass filter :py:func:`readgssi.filtering.triangular`. Defaults to :py:data:`None` (no filter).
    :param int freqmax: Maximum frequency value to feed to the vertical triangular FIR bandpass filter :py:func:`readgssi.filtering.triangular`. Defaults to :py:data:`None` (no filter).
    :param bool reverse: Whether to read the array backwards (i.e. flip horizontally; :py:func:`readgssi.arrayops.flip`). Defaults to :py:data:`False`. Useful for lining up travel directions of files run opposite each other.
    :param int bgr: Background removal filter applied after stacking (:py:func:`readgssi.filtering.bgr`). Defaults to :py:data:`False` (off). :py:data:`bgr=True` must be accompanied by a valid value for :py:data:`win`.
    :param int win: Window size for background removal filter (:py:func:`readgssi.filtering.bgr`). If :py:data:`bgr=True` and :py:data:`win=0`, the full-width row average will be subtracted from each row. If :py:data:`bgr=True` and :py:data:`win=50`, a moving window will calculate the average of 25 cells on either side of the current cell, and subtract that average from the cell value, using :py:func:`scipy.ndimage.uniform_filter1d` with :py:data:`mode='constant'` and :py:data:`cval=0`. This is useful for removing non-uniform horizontal average, but the tradeoff is that it creates ghost data half the window size away from vertical figures, and that a window size set too low will obscure any horizontal layering longer than the window size.
    :param bool dewow: Whether to apply a vertical dewow filter (experimental). See :py:func:`readgssi.filtering.dewow`.
    :param bool absval: If :py:data:`True`, displays the absolute value of the vertical gradient of the array when plotting. Good for displaying faint array features.
    :param bool normalize: Distance normalization (:py:func:`readgssi.arrayops.distance_normalize`). Defaults to :py:data:`False`.
    :param bool specgram: Produce a spectrogram of a trace in the array using :py:func:`readgssi.plot.spectrogram`. Defaults to :py:data:`False` (if :py:data:`True`, defaults to a trace roughly halfway across the profile). This is mostly for debugging and is not currently accessible from the command line.
    :param bool noshow: If :py:data:`True`, this will suppress the matplotlib interactive window and simply save a file. This is useful for processing many files in a folder without user input.
    :param float spm: User-set samples per meter. This overrides the value read from the header, and typically doesn't need to be set if the samples per meter value was set correctly at survey time. This value does not need to be set if GPS input (DZG file) is present and the user sets :py:data:`normalize=True`.
    :param int start_scan: zero based start scan to read data from. Defaults to zero.
    :param int num_scans: number of scans to read from the file, Defaults to -1, which reads from start_scan to end of file.
    :param float epsr: Epsilon_r, otherwise known as relative permittivity, or dielectric constant. This determines the speed at which waves travel through the first medium they encounter. It is used to calculate the profile depth if depth units are specified on the Z-axis of plots.
    :param bool title: Whether to display descriptive titles on plots. Defaults to :py:data:`True`.
    :param list[int,int,int,int] zoom: Zoom extents to set programmatically for matplotlib plots. Must pass a list of four integers: :py:data:`[left, right, up, down]`. Since the z-axis begins at the top, the "up" value is actually the one that displays lower on the page. All four values are axis units, so if you are working in nanoseconds, 10 will set a limit 10 nanoseconds down. If your x-axis is in seconds, 6 will set a limit 6 seconds from the start of the survey. It may be helpful to display the matplotlib interactive window at full extents first, to determine appropriate extents to set for this parameter. If extents are set outside the boundaries of the image, they will be set back to the boundaries. If two extents on the same axis are the same, the program will default to plotting full extents for that axis.
    :rtype: header (:py:class:`dict`), radar array (:py:class:`numpy.ndarray`), gps (False or :py:class:`pandas.DataFrame`)
    :param bool pausecorrect: If :py:data:`True`, search the DZG file for pauses, where GPS keeps recording but radar unit does not, and correct them if necessary. Defaults to :py:data:`False`.
    :param bool showmarks: If :py:data:`True`, display mark locations in plot. Defaults to :py:data:`False`.
    """
    if infile:
        # read the file
        try:
            header, data, gps = readdzt(infile, gps=normalize, spm=spm,
                                        start_scan=start_scan, num_scans=num_scans,
                                        epsr=epsr, antfreq=antfreq, zero=zero,
                                        verbose=verbose)
        except IOError as e: # the user has selected an inaccessible or nonexistent file
            raise IOError(e)
        infile_ext = os.path.splitext(infile)[1]
        infile_basename = os.path.splitext(infile)[0]
    else:
        raise IOError('ERROR: no input file specified')

    rhf_sps = header['rhf_sps']
    rhf_spm = header['rhf_spm']
    line_dur = header['sec']
    for chan in list(range(header['rh_nchan'])):
        try:
            ANT[header['rh_antname'][chan]]
        except KeyError as e:
            print('--------------------WARNING - PLEASE READ---------------------')
            printmsg('WARNING: could not read frequency for antenna name "%s"' % e)
            if (antfreq != None) and (antfreq != [None, None, None, None]):
                printmsg('using user-specified antenna frequency. Please ensure frequency value or list of values is correct.')
                printmsg('old values: %s' % (header['antfreq']))
                printmsg('new values: %s' % (antfreq))
                header['antfreq'] = antfreq
            else:
                printmsg('WARNING: trying to use frequencies of %s MHz (estimated)...' % (header['antfreq'][chan]))
            printmsg('more info: rh_ant=%s' % (header['rh_ant']))
            printmsg('           known_ant=%s' % (header['known_ant']))
            printmsg("please submit a bug report with this warning, the antenna name and frequency")
            printmsg('at https://github.com/iannesbitt/readgssi/issues/new')
            printmsg('or send via email to ian (dot) nesbitt (at) gmail (dot) com.')
            printmsg('if possible, please attach a ZIP file with the offending DZT inside.')
            print('--------------------------------------------------------------')

    chans = list(range(header['rh_nchan']))

    return (data, header)