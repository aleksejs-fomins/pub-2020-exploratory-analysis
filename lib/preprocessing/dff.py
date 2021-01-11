import numpy as np

from mesostat.utils.arrays import numpy_transpose_byorder


def dff_func(x, timesBaseline):
    return x / np.mean(x[timesBaseline]) - 1


def dff(t, data3D, dimOrd, method, tBaseMin=None, tBaseMax=None):
    if method == 'raw':
        return data3D
    else:
        dimOrdCanon = 'psr'
        data3DCanon = numpy_transpose_byorder(data3D.copy(), dimOrd, dimOrdCanon)

        tBaseMin = tBaseMin if tBaseMin is not None else np.min(t) - 1
        tBaseMax = tBaseMax if tBaseMax is not None else np.max(t)

        timeIdxsBaseline = np.logical_and(t > tBaseMin, t <= tBaseMax)

        nChannel, _, nTrial = data3DCanon.shape

        if method == 'dff_session':
            for iChannel in range(nChannel):
                data3DCanon[iChannel] = dff_func(data3DCanon[iChannel], timeIdxsBaseline)
        elif method == 'dff_trial':
            for iTrial in range(nTrial):
                for iChannel in range(nChannel):
                    data3DCanon[iChannel, :, iTrial] = dff_func(data3DCanon[iChannel, :, iTrial], timeIdxsBaseline)
        else:
            raise ValueError('Unexpected method', method)

        return numpy_transpose_byorder(data3DCanon, dimOrdCanon, dimOrd)
