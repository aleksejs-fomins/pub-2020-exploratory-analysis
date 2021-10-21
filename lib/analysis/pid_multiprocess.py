import os
import numpy as np

import lib.analysis.pid_common as pid

import lib.common.h5lock as h5lock
from lib.common.param_sweep import DataParameterSweep
from lib.common.datawrapper import get_data_list


def pid_multiprocess_session(dataDB, mc, h5outname, argSweepDict, exclQueryLst, dim=3, nBin=4, metric='BivariatePID',
                             permuteTarget=False, dropChannels=None, timeSweep=False):

    h5lock.touch_file(h5outname)            # If output file does not exist, create it
    h5lock.touch_group(h5outname, 'lock')   # If lock group does not exist, create lock group

    dps = DataParameterSweep(dataDB, exclQueryLst, **argSweepDict)

    for idx, row in dps.sweepDF.iterrows():
        # channelNames = dataDB.get_channel_labels(row['mousename'])
        # nChannels = len(channelNames)

        keyDataMouse = 'PID_' + '_'.join([str(key) for key in row.values])
        keyLabelMouse = 'Label_' + '_'.join([str(key) for key in row.values])
        for session in dataDB.get_sessions(row['mousename'], datatype=row['datatype']):
            keyDataSession = keyDataMouse + '_' + session
            keyLabelSession = keyLabelMouse + '_' + session
            print(keyDataSession)

            # Test if this parameter combination not yet calculated
            if h5lock.lock_test_available(h5outname, keyDataSession):
                kwargs = dict(row)
                del kwargs['mousename']
                kwargs = {k: v if v != 'None' else None for k, v in kwargs.items()}

                # Get data
                dataLst = dataDB.get_neuro_data({'session': session}, zscoreDim=None, **kwargs)

                # Calculate PID
                rezIdxs, rezVals = pid.pid(dataLst, mc, metric=metric, dim=dim, nBin=nBin, timeSweep=timeSweep,
                                           permuteTarget=permuteTarget, dropChannels=dropChannels)

                # Save to file
                h5lock.unlock_write(h5outname, keyLabelSession, keyDataSession, np.array(rezIdxs), rezVals)


def pid_multiprocess_mouse(dataDB, mc, h5outname, argSweepDict, exclQueryLst, dim=3, nBin=4, metric='BivariatePID',
                           permuteTarget=False, dropChannels=None, timeSweep=False):
    h5lock.touch_file(h5outname)            # If output file does not exist, create it
    h5lock.touch_group(h5outname, 'lock')   # If lock group does not exist, create lock group

    dps = DataParameterSweep(dataDB, exclQueryLst, **argSweepDict)

    for idx, row in dps.sweepDF.iterrows():
        # channelNames = dataDB.get_channel_labels(row['mousename'])
        # nChannels = len(channelNames)
        keyDataMouse = 'PID_' + '_'.join([str(key) for key in row.values])
        keyLabelMouse = 'Label_' + '_'.join([str(key) for key in row.values])

        # Test if this parameter combination not yet calculated
        if h5lock.lock_test_available(h5outname, keyDataMouse):
            kwargs = dict(row)
            del kwargs['mousename']
            kwargs = {k: v if v != 'None' else None for k, v in kwargs.items()}

            # Get data
            dataLst = dataDB.get_neuro_data({'mousename': row['mousename']}, zscoreDim=None, **kwargs)

            # Calculate PID
            rezIdxs, rezVals = pid.pid(dataLst, mc, metric=metric, dim=dim, nBin=nBin, timeSweep=timeSweep,
                                       permuteTarget=permuteTarget, dropChannels=dropChannels)

            # Save to file
            h5lock.unlock_write(h5outname, keyLabelMouse, keyDataMouse, np.array(rezIdxs), rezVals)


def pid_multiprocess_mouse_trgsweep(dataDB, mc, h5outname, argSweepDict, exclQueryLst, dim=3, nBin=4,
                                    metric='BivariatePID', permuteTarget=False, dropChannels=None, timeSweep=False):
    h5lock.touch_file(h5outname)            # If output file does not exist, create it
    h5lock.touch_group(h5outname, 'lock')   # If lock group does not exist, create lock group

    dps = DataParameterSweep(dataDB, exclQueryLst, **argSweepDict)

    channelLabels = dataDB.get_channel_labels()
    haveDelay = 'DEL' in dataDB.get_interval_names()

    for idx, row in dps.sweepDF.iterrows():
        for iTrg, trgLabel in enumerate(channelLabels):
            # Ensure target is not dropped
            if (dropChannels is None) or (iTrg not in dropChannels):
                keyDataMouse = 'PID_' + '_'.join([str(key) for key in row.values] + [str(iTrg)])
                keyLabelMouse = 'Label_' + '_'.join([str(key) for key in row.values] + [str(iTrg)])

                # Test if this parameter combination not yet calculated
                if h5lock.lock_test_available(h5outname, keyDataMouse):

                    # Sources are all channels - target - dropped
                    exclChannels = [iTrg]
                    if dropChannels is not None:
                        exclChannels += dropChannels
                    srcLabels = [ch for iCh, ch in enumerate(channelLabels) if iCh not in exclChannels]

                    kwargs = dict(row)
                    del kwargs['mousename']
                    kwargs = {k: v if v != 'None' else None for k,v in kwargs.items()}

                    # Get data
                    dataLst = get_data_list(dataDB, haveDelay, row['mousename'], zscoreDim=None, **kwargs)

                    print(len(dataLst), dataLst[0].shape, haveDelay, row['mousename'], kwargs)

                    # dataLst = dataDB.get_neuro_data({'mousename': row['mousename']}, zscoreDim=None, **kwargs)

                    # Calculate PID
                    rezIdxs, rezVals = pid.pid(dataLst, mc, metric=metric, dim=dim, nBin=nBin, timeSweep=timeSweep,
                                               labelsAll=channelLabels, labelsSrc=srcLabels, labelsTrg=[trgLabel],
                                               permuteTarget=permuteTarget)

                    # Save to file
                    h5lock.unlock_write(h5outname, keyLabelMouse, keyDataMouse, np.array(rezIdxs), rezVals)
