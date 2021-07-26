import os
import numpy as np

from mesostat.utils.h5py_lock import h5wrap
from mesostat.utils.pandas_helper import outer_product_df, drop_rows_byquery

import lib.analysis.pid_common as pid
from lib.analysis.brain_avg_activity import _get_data_concatendated_sessions


def _h5_touch_file(h5outname):
    if not os.path.isfile(h5outname):
        with h5wrap(h5outname, 'w', 1, 300, True) as h5w:
            pass


def _h5_touch_group(h5outname, groupname):
    with h5wrap(h5outname, 'a', 1, 300, True) as h5w:
        if groupname not in h5w.f.keys():
            h5w.f.create_group(groupname)


def _h5_lock_test_available(h5outname, lockKey):
    with h5wrap(h5outname, 'a', 1, 300, True) as h5w:
        if lockKey in h5w.f:
            print(lockKey, 'already calculated, skipping')
            return False
        elif lockKey in h5w.f['lock']:
            print(lockKey, 'is currently being calculated, skipping')
            return False

        print(lockKey, 'not calculated, calculating')
        h5w.f['lock'][lockKey] = 1
        return True


def _h5_unlock_write(h5outname, idxsKey, valsKey, idxs, vals):
    # Save to file
    with h5wrap(h5outname, 'a', 1, -1, True) as h5w:
        del h5w.f['lock'][valsKey]
        h5w.f[valsKey] = vals
        h5w.f[idxsKey] = np.array(idxs)


def pid_multiprocess_session(dataDB, mc, h5outname, argSweepDict, exclQueryLst, dim=3, nBin=4, metric='BivariatePID',
                             permuteTarget=False, dropChannels=None, timeSweep=False):

    _h5_touch_file(h5outname)            # If output file does not exist, create it
    _h5_touch_group(h5outname, 'lock')   # If lock group does not exist, create lock group

    sweepDF = outer_product_df(argSweepDict)
    sweepDF = drop_rows_byquery(sweepDF, exclQueryLst)

    for idx, row in sweepDF.iterrows():
        # channelNames = dataDB.get_channel_labels(row['mousename'])
        # nChannels = len(channelNames)

        keyDataMouse = 'PID_' + '_'.join([str(key) for key in row.values])
        keyLabelMouse = 'Label_' + '_'.join([str(key) for key in row.values])
        for session in dataDB.get_sessions(row['mousename'], datatype=row['datatype']):
            keyDataSession = keyDataMouse + '_' + session
            keyLabelSession = keyLabelMouse + '_' + session
            print(keyDataSession)

            # Test if this parameter combination not yet calculated
            if _h5_lock_test_available(h5outname, keyDataSession):
                kwargs = dict(row)
                del kwargs['mousename']
                kwargs = {k: v if v != 'None' else None for k, v in kwargs.items()}

                # Get data
                dataLst = dataDB.get_neuro_data({'session': session}, zscoreDim=None, **kwargs)

                # Calculate PID
                rezIdxs, rezVals = pid.pid(dataLst, mc, metric=metric, dim=dim, nBin=nBin, timeSweep=timeSweep,
                                           permuteTarget=permuteTarget, dropChannels=dropChannels)

                # Save to file
                _h5_unlock_write(h5outname, keyLabelSession, keyDataSession, np.array(rezIdxs), rezVals)


def pid_multiprocess_mouse(dataDB, mc, h5outname, argSweepDict, exclQueryLst, dim=3, nBin=4, metric='BivariatePID',
                           permuteTarget=False, dropChannels=None, timeSweep=False):
    _h5_touch_file(h5outname)            # If output file does not exist, create it
    _h5_touch_group(h5outname, 'lock')   # If lock group does not exist, create lock group

    sweepDF = outer_product_df(argSweepDict)
    sweepDF = drop_rows_byquery(sweepDF, exclQueryLst)

    for idx, row in sweepDF.iterrows():
        # channelNames = dataDB.get_channel_labels(row['mousename'])
        # nChannels = len(channelNames)
        keyDataMouse = 'PID_' + '_'.join([str(key) for key in row.values])
        keyLabelMouse = 'Label_' + '_'.join([str(key) for key in row.values])

        # Test if this parameter combination not yet calculated
        if _h5_lock_test_available(h5outname, keyDataMouse):
            kwargs = dict(row)
            del kwargs['mousename']
            kwargs = {k: v if v != 'None' else None for k, v in kwargs.items()}

            # Get data
            dataLst = dataDB.get_neuro_data({'mousename': row['mousename']}, zscoreDim=None, **kwargs)

            # Calculate PID
            rezIdxs, rezVals = pid.pid(dataLst, mc, metric=metric, dim=dim, nBin=nBin, timeSweep=timeSweep,
                                       permuteTarget=permuteTarget, dropChannels=dropChannels)

            # Save to file
            _h5_unlock_write(h5outname, keyLabelMouse, keyDataMouse, np.array(rezIdxs), rezVals)


def pid_multiprocess_mouse_trgsweep(dataDB, mc, h5outname, argSweepDict, exclQueryLst, dim=3, nBin=4,
                                    metric='BivariatePID', permuteTarget=False, dropChannels=None, timeSweep=False):
    _h5_touch_file(h5outname)            # If output file does not exist, create it
    _h5_touch_group(h5outname, 'lock')   # If lock group does not exist, create lock group

    sweepDF = outer_product_df(argSweepDict)
    sweepDF = drop_rows_byquery(sweepDF, exclQueryLst)

    channelLabels = dataDB.get_channel_labels()
    haveDelay = 'DEL' in dataDB.get_interval_names()

    for idx, row in sweepDF.iterrows():
        for iTrg, trgLabel in enumerate(channelLabels):
            # Ensure target is not dropped
            if (dropChannels is None) or (iTrg not in dropChannels):
                keyDataMouse = 'PID_' + '_'.join([str(key) for key in row.values] + [str(iTrg)])
                keyLabelMouse = 'Label_' + '_'.join([str(key) for key in row.values] + [str(iTrg)])

                # Test if this parameter combination not yet calculated
                if _h5_lock_test_available(h5outname, keyDataMouse):

                    # Sources are all channels - target - dropped
                    exclChannels = [iTrg]
                    if dropChannels is not None:
                        exclChannels += dropChannels
                    srcLabels = [ch for iCh, ch in enumerate(channelLabels) if iCh not in exclChannels]

                    kwargs = dict(row)
                    del kwargs['mousename']
                    kwargs = {k: v if v != 'None' else None for k,v in kwargs.items()}

                    # Get data
                    dataLst = [_get_data_concatendated_sessions(dataDB, haveDelay, row['mousename'],
                                                                zscoreDim=None, **kwargs)]

                    print(len(dataLst), dataLst[0].shape, haveDelay, row['mousename'], kwargs)

                    # dataLst = dataDB.get_neuro_data({'mousename': row['mousename']}, zscoreDim=None, **kwargs)

                    # Calculate PID
                    rezIdxs, rezVals = pid.pid(dataLst, mc, metric=metric, dim=dim, nBin=nBin, timeSweep=timeSweep,
                                               labelsAll=channelLabels, labelsSrc=srcLabels, labelsTrg=[trgLabel],
                                               permuteTarget=permuteTarget)

                    # Save to file
                    _h5_unlock_write(h5outname, keyLabelMouse, keyDataMouse, np.array(rezIdxs), rezVals)
