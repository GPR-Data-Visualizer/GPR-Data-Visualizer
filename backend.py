from readgssi import readgssi as r
import numpy as np
from scipy.ndimage.filters import uniform_filter1d
from scipy.signal import firwin, lfilter, hilbert2
from scipy import fft as scipyfft
from PyEMD import EMD
# import emd
import pywt
from datetime import datetime


#----------- READING DATA ----------------#
def dzt_func(file_paths):
    data = list()
    headers = list()
    for file in file_paths:
        read = r.readgssi(file)
        data.append(read[0])
        headers.append(read[1])
    return data, headers


#------------- FILTERING ------------------#
def dzt_filters(data, headers, active_filters):
    for i in range(len(data)):
        for filt_active in active_filters:
            if filt_active == 'Horizontal background removal':
                data[i] = bgr(data[i], headers[i], win=float(active_filters[filt_active][0].split('=')[1]))
            elif filt_active == 'Vertical triangular FIR bandpass':
                data[i] = triangular(data[i], headers[i], float(active_filters[filt_active][0].split('=')[1]), float(active_filters[filt_active][1].split('=')[1]))
            elif filt_active == 'Fast Fourier Transform':
                data[i] = scipyfft.rfft2(data[i]).real
            elif filt_active == 'Hilbert Huang Transform':
                # print(datetime.now().strftime("%H:%M:%S"))
                x = 0
                emd = EMD()
                # NEED TO DO MORE RESEARCH INTO THESE ATRIBUTES
                emd.FIXE = 15
                emd.FIXE_H = 0
                for chan in data[i]:
                    # imfs = emd(chan)
                    imfs = emd(chan)
                    chan = imfs[len(imfs)-1]
                    x += 1
                data[i] = hilbert2(data[i]).real
                # print(datetime.now().strftime("%H:%M:%S"))
            elif filt_active == 'Wavelets':
                w = pywt.Wavelet(active_filters[filt_active][0])
                if w.name in pywt.wavelist(kind='discrete'):
                    for chan in data[i]:
                        chan, cD = pywt.dwt(chan, w)
                elif w.name in pywt.wavelist(kind='continuous'):
                    for chan in data[i]:
                        chan, cD = pywt.ContinuousWavelet(chan, w)
                else:
                    print("not implemented")
    return data


#========= FILTERING FUNCTIONS FROM READGSSI==================#
def bgr(ar, header, win=0):
    """
    Horizontal background removal (BGR). Subtracts off row averages for full-width or window-length slices. For usage see :ref:`Getting rid of horizontal noise`.

    :param numpy.ndarray ar: The radar array
    :param dict header: The file header dictionary
    :param int win: The window length to process. 0 resolves to full-width, whereas positive integers dictate the window size in post-stack traces.
    :rtype: :py:class:`numpy.ndarray`
    """
    if (int(win) > 1) & (int(win) < ar.shape[1]):
        window = int(win)
        how = 'boxcar (%s trace window)' % window
    else:
        how = 'full only'
    i = 0
    for row in ar:          # each row
        mean = np.mean(row)
        ar[i] = row - mean
        i += 1
    if how != 'full only':
        if window < 3:
            window = 3
        elif (window / 2. == int(window / 2)):
            window = window + 1
        ar -= uniform_filter1d(ar, size=window, mode='constant', cval=0, axis=1)
    return ar


def triangular(ar, header, freqmin, freqmax, zerophase=True):
    """
    Vertical triangular FIR bandpass. This filter is designed to closely emulate that of RADAN.

    Filter design is implemented by :py:func:`scipy.signal.firwin` with :code:`numtaps=25` and implemented with :py:func:`scipy.signal.lfilter`.

    .. note:: This function is not compatible with scipy versions prior to 1.3.0.

    :param np.ndarray ar: The radar array
    :param dict header: The file header dictionary
    :param int freqmin: The lower corner of the bandpass
    :param int freqmax: The upper corner of the bandpass
    :param bool zerophase: Whether to run the filter forwards and backwards in order to counteract the phase shift
    :rtype: :py:class:`numpy.ndarray`
    """
    samp_freq = header['samp_freq']
    freqmin = freqmin * 10 ** 6
    freqmax = freqmax * 10 ** 6
    
    numtaps = 25

    filt = firwin(numtaps=numtaps, cutoff=[freqmin, freqmax], window='triangle', pass_zero='bandpass', fs=samp_freq)

    far = lfilter(filt, 1.0, ar, axis=0).copy()
    if zerophase:
        far = lfilter(filt, 1.0, far[::-1], axis=0)[::-1]
    return far
