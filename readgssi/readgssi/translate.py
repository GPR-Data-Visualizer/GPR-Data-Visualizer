import h5py
import pandas as pd
import numpy as np
import json
import struct
from readgssi.gps import readdzg
import readgssi.functions as fx
from datetime import datetime

"""
contains translations to common formats
"""

def json_header(header, outfile_abspath, verbose=False):
    """
    Save header values as a .json so another script can take what it needs. This is used to export to `GPRPy <https://github.com/NSGeophysics/gprpy>`_.

    :param dict header: The file header dictionary
    :param str outfile_abspath: Output file path
    :param bool verbose: Verbose, defaults to False
    """
    with open('%s.json' % (outfile_abspath), 'w') as f:
        if verbose:
            fx.printmsg('serializing header as %s' % (f.name))
        json.dump(obj=header, fp=f, indent=4, sort_keys=True, default=str)

def csv(ar, outfile_abspath, header=None, verbose=False):
    """
    Output to csv. Data is read into a :py:class:`pandas.DataFrame`, then written using :py:func:`pandas.DataFrame.to_csv`.

    :param numpy.ndarray ar: Radar array
    :param str outfile_abspath: Output file path
    :param dict header: File header dictionary to write, if desired. Defaults to None.
    :param bool verbose: Verbose, defaults to False
    """
    if verbose:
        t = ''
        if header:
            t = ' with json header'
        fx.printmsg('output format is csv%s. writing data to: %s.csv' % (t, outfile_abspath))
    data = pd.DataFrame(ar) # using pandas to output csv
    data.to_csv('%s.csv' % (outfile_abspath)) # write
    if header:
        json_header(header=header, outfile_abspath=outfile_abspath, verbose=verbose)

def numpy(ar, outfile_abspath, header=None, verbose=False):
    """
    Output to binary numpy binary file (.npy) with the option of writing the header to .json as well.

    :param numpy.ndarray ar: Radar array
    :param str outfile_abspath: Output file path
    :param dict header: File header dictionary to write, if desired. Defaults to None.
    :param bool verbose: Verbose, defaults to False
    """
    if verbose:
        t = ''
        if header:
            t = ' with json header (compatible with GPRPy)'
        fx.printmsg('output format is numpy binary%s' % t)
        fx.printmsg('writing data to %s.npy' % outfile_abspath)
    np.save('%s.npy' % outfile_abspath, ar, allow_pickle=False)
    if header:
        json_header(header=header, outfile_abspath=outfile_abspath, verbose=verbose)

def writetime(d):
    '''
    Function to write dates to :code:`rfDateByte` binary objects in DZT headers.
    An inverse of the :py:func:`readgssi.dzt.readtime` function.

    DZT :code:`rfDateByte` objects are 32 bits of binary (01001010111110011010011100101111),
    structured as little endian u5u6u5u5u4u7 where all numbers are base 2 unsigned int (uX)
    composed of X number of bits. Four bytes is an unnecessarily high level of compression
    for a single date object in a filetype that often contains tens or hundreds of megabytes
    of array information anyway.

    So this function reads a datetime object and outputs
    (seconds/2, min, hr, day, month, year-1980).

    For more information on :code:`rfDateByte`, see page 55 of
    `GSSI's SIR 3000 manual <https://support.geophysical.com/gssiSupport/Products/Documents/Control%20Unit%20Manuals/GSSI%20-%20SIR-3000%20Operation%20Manual.pdf>`_.

    :param datetime d: the :py:class:`datetime.datetime` to be encoded
    :rtype: bytes
    '''
    # get binary values
    sec2 = int(bin(int(d.second / 2))[2:])
    mins = int(bin(d.minute)[2:])
    hr = int(bin(d.hour)[2:])
    day = int(bin(d.day)[2:])
    mo = int(bin(d.month)[2:])
    yr = int(bin(d.year - 1980)[2:])
    # create binary string with proper padding
    dtbits = '%07d%04d%05d%05d%06d%05d' % (yr, mo, day, hr, mins, sec2)
    # create four bytes that make up rfDateByte
    byt0 = int(dtbits[24:], 2)
    byt1 = int(dtbits[16:24], 2)
    byt2 = int(dtbits[8:16], 2)
    byt3 = int(dtbits[0:8], 2)
    # return a byte array
    return bytes([byt0, byt1, byt2, byt3])

def dzt(ar, outfile_abspath, header, verbose=False):
    """
    .. warning:: DZT output is only currently compatible with single-channel files.

    This function will output a RADAN-compatible DZT file after processing.
    This is useful to circumvent RADAN's distance-normalization bug
    when the desired outcome is array migration.

    Users can set DZT output via the command line by setting the
    :code:`-f dzt` flag, or in Python by doing the following: ::

        from readgssi.dzt import readdzt
        from readgssi import translate
        from readgssi.arrayops import stack, distance_normalize

        # first, read a data file
        header, data, gps = readdzt('FILE__001.DZT')

        # do some stuff
        # (distance normalization must be done before stacking)
        for a in data:
            header, data[a], gps = distance_normalize(header=header, ar=data[a], gps=gps)
            header, data[a], stack = stack(header=header, ar=data[a], stack=10)

        # output as modified DZT
        translate.dzt(ar=data, outfile_abspath='FILE__001-DnS10.DZT', header=header)

    This will output :code:`FILE__001-DnS10.DZT` as a distance-normalized DZT.

    :param numpy.ndarray ar: Radar array
    :param str infile_basename: Input file basename
    :param str outfile_abspath: Output file path
    :param dict header: File header dictionary to write, if desired. Defaults to None.
    :param bool verbose: Verbose, defaults to False
    """

    '''
    Assumptions:
    - constant velocity or distance between marks (may be possible to add a check)
    '''
    if len(ar) > 1:
        outfile_abspath = outfile_abspath.replace('c1', '')
    if not outfile_abspath.endswith(('.DZT', '.dzt')):
        outfile_abspath = outfile_abspath + '.DZT'
    
    outfile = open(outfile_abspath, 'wb')
    fx.printmsg('writing to: %s' % outfile.name)

    for i in range(header['rh_nchan']):
        fx.printmsg('writing DZT header for channel %s' % (i))
        # header should read all values per-channel no matter what
        outfile.write(struct.pack('<h', header['rh_tag']))
        outfile.write(struct.pack('<h', header['rh_data']))
        outfile.write(struct.pack('<h', header['rh_nsamp']))
        outfile.write(struct.pack('<h', 32)) # rhf_bits - for simplicity, just hard-coding 32 bit
        outfile.write(struct.pack('<h', header['rh_zero']))
        # byte 10
        outfile.write(struct.pack('<f', header['rhf_sps']))
        outfile.write(struct.pack('<f', header['rhf_spm'])) # dzt.py ln 94-97
        outfile.write(struct.pack('<f', header['rhf_mpm']))
        outfile.write(struct.pack('<f', header['rhf_position']))
        outfile.write(struct.pack('<f', header['rhf_range']))
        outfile.write(struct.pack('<h', header['rh_npass']))
        # byte 32
        outfile.write(writetime(header['rhb_cdt']))
        outfile.write(writetime(datetime.now())) # modification date/time
        # byte 40
        outfile.write(struct.pack('<h', header['rh_rgain']))
        outfile.write(struct.pack('<h', header['rh_nrgain']))
        outfile.write(struct.pack('<h', header['rh_text']))
        outfile.write(struct.pack('<h', header['rh_ntext']))
        outfile.write(struct.pack('<h', header['rh_proc']))
        outfile.write(struct.pack('<h', header['rh_nproc']))
        outfile.write(struct.pack('<h', header['rh_nchan']))
        outfile.write(struct.pack('<f', header['rhf_epsr'])) # dzt.py ln 121-126
        outfile.write(struct.pack('<f', header['rhf_top']))
        outfile.write(struct.pack('<f', header['rhf_depth']))
        # byte 66
        outfile.write(struct.pack('<f', header['rh_xstart'])) # part of rh_coordx
        outfile.write(struct.pack('<f', header['rh_xend'])) # part of rh_coordx
        outfile.write(struct.pack('<f', header['rhf_servo_level']))
        outfile.write(bytes(3)) # "reserved"
        outfile.write(struct.pack('B', header['rh_accomp']))
        outfile.write(struct.pack('<h', header['rh_sconfig']))
        outfile.write(struct.pack('<h', header['rh_spp']))
        outfile.write(struct.pack('<h', header['rh_linenum']))
        # byte 88
        outfile.write(struct.pack('<f', header['rh_ystart'])) # part of rh_coordy
        outfile.write(struct.pack('<f', header['rh_yend'])) # part of rh_coordy
        outfile.write(header['rh_96'])
        outfile.write(struct.pack('c', header['rh_dtype']))
        outfile.write(header['dzt_ant'][i])
        outfile.write(header['rh_112'])
        # byte 113
        outfile.write(header['vsbyte'])
        outfile.write(header['rh_name'])
        outfile.write(header['rh_chksum'])
        # byte 128
        outfile.write(header['INFOAREA'])
        outfile.write(header['rh_RGPS0'])
        outfile.write(header['rh_RGPS1'])
        i += 1

    outfile.write(header['header_extra'])

    stack = []
    i = 0
    for i in range(header['rh_nchan']):
        # replace zeroed rows
        stack.append(np.zeros((header['timezero'][i], ar[i].shape[1]),
                                    dtype=np.int32))
        stack.append(ar[i])
        i += 1

    writestack = np.vstack(tuple(stack))
    sh = writestack.shape
    writestack = writestack.T.reshape(-1)
    fx.printmsg('writing %s data samples for %s channels (%s x %s)'
          % (writestack.shape[0],
             int(len(stack)/2),
             sh[0], sh[1]))

    # hard coded to write 32 bit signed ints to keep lossiness to a minimum
    outfile.write(writestack.round().astype(np.int32, casting='unsafe').tobytes(order='C'))

    outfile.close()

