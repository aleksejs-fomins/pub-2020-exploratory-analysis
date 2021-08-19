import os
import h5py

import numpy as np

from mesostat.utils.h5py_persist import h5persist


def touch_file(h5outname):
    if not os.path.isfile(h5outname):
        with h5persist(h5outname, 'w', 1, 300, True) as h5w:
            pass


def touch_group(h5outname, groupname):
    with h5persist(h5outname, 'a', 1, 300, True) as h5w:
        if groupname not in h5w.f.keys():
            h5w.f.create_group(groupname)


def lock_test_available(h5outname, lockKey):
    with h5persist(h5outname, 'a', 1, 300, True) as h5w:
        if lockKey in h5w.f:
            print(lockKey, 'already calculated, skipping')
            return False
        elif lockKey in h5w.f['lock']:
            print(lockKey, 'is currently being calculated, skipping')
            return False

        print(lockKey, 'not calculated, calculating')
        h5w.f['lock'][lockKey] = 1
        return True


def unlock_write(h5outname, idxsKey, valsKey, idxs, vals):
    # Save to file
    with h5persist(h5outname, 'a', 1, -1, True) as h5w:
        del h5w.f['lock'][valsKey]
        h5w.f[valsKey] = vals
        h5w.f[idxsKey] = np.array(idxs)