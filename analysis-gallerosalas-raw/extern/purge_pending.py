import os, sys
import h5py

fname = sys.argv[1]

with h5py.File(fname, 'a') as h5file:
    if 'lock' in h5file.keys():
        print('deleting lock')
        del h5file['lock']
    else:
        print('No lock found')
