import os
import numpy as np

from mesostat.utils.h5py_lock import h5wrap
from mesostat.utils.pandas_helper import outer_product_df, drop_rows_byquery

import lib.analysis.pid_common as pid


def pid_multiprocess_session(dataDB, mc, h5outname, argSweepDict, exclQueryLst, dim=3, nBin=4, metric='BivariatePID',
                             permuteTarget=False, dropChannels=None):
    # If output file does not exist, create it
    if not os.path.isfile(h5outname):
        with h5wrap(h5outname, 'w', 1, 300, True) as h5w:
            pass

    # If lock group does not exist, create lock group
    with h5wrap(h5outname, 'a', 1, 300, True) as h5w:
        if 'lock' not in h5w.f.keys():
            h5w.f.create_group('lock')

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

            with h5wrap(h5outname, 'a', 1, 300, True) as h5w:
                if keyDataSession in h5w.f:
                    print(keyDataSession, 'already calculated, skipping')
                    continue
                elif keyDataSession in h5w.f['lock']:
                    print(keyDataSession, 'is currently being calculated, skipping')
                    continue

                print(keyDataSession, 'not calculated, calculating')
                h5w.f['lock'][keyDataSession] = 1

            kwargs = dict(row)
            del kwargs['mousename']

            # Get data
            dataLst = dataDB.get_neuro_data({'session': session}, zscoreDim=None, **kwargs)

            # Calculate PID
            rezIdxs, rezVals = pid.pid(dataLst, mc, metric=metric, dim=dim, nBin=nBin,
                                       permuteTarget=permuteTarget, dropChannels=dropChannels)

            # Save to file
            with h5wrap(h5outname, 'a', 1, -1, True) as h5w:
                del h5w.f['lock'][keyDataSession]
                h5w.f[keyDataSession] = rezVals
                h5w.f[keyLabelSession] = np.array(rezIdxs)

            # rezDF.to_hdf(h5outname, sessionDataLabel, mode='a', format='table', data_columns=True)


def pid_multiprocess_mouse(dataDB, mc, h5outname, argSweepDict, exclQueryLst, dim=3, nBin=4, metric='BivariatePID',
                           permuteTarget=False, dropChannels=None):
    # If output file does not exist, create it
    if not os.path.isfile(h5outname):
        with h5wrap(h5outname, 'w', 1, 300, True) as h5w:
            pass

    # If lock group does not exist, create lock group
    with h5wrap(h5outname, 'a', 1, 300, True) as h5w:
        if 'lock' not in h5w.f.keys():
            h5w.f.create_group('lock')

    sweepDF = outer_product_df(argSweepDict)
    sweepDF = drop_rows_byquery(sweepDF, exclQueryLst)

    for idx, row in sweepDF.iterrows():
        # channelNames = dataDB.get_channel_labels(row['mousename'])
        # nChannels = len(channelNames)
        keyDataMouse = 'PID_' + '_'.join([str(key) for key in row.values])
        keyLabelMouse = 'Label_' + '_'.join([str(key) for key in row.values])

        with h5wrap(h5outname, 'a', 1, 300, True) as h5w:
            if keyDataMouse in h5w.f:
                print(keyDataMouse, 'already calculated, skipping')
                continue
            elif keyDataMouse in h5w.f['lock']:
                print(keyDataMouse, 'is currently being calculated, skipping')
                continue

            print(keyDataMouse, 'not calculated, calculating')
            h5w.f['lock'][keyDataMouse] = 1

        kwargs = dict(row)
        del kwargs['mousename']

        # Get data
        dataLst = dataDB.get_neuro_data({'mousename': row['mousename']}, zscoreDim=None, **kwargs)

        # Calculate PID
        rezIdxs, rezVals = pid.pid(dataLst, mc, metric=metric, dim=dim, nBin=nBin,
                                   permuteTarget=permuteTarget, dropChannels=dropChannels)

        # Save to file
        with h5wrap(h5outname, 'a', 1, -1, True) as h5w:
            del h5w.f['lock'][keyDataMouse]
            h5w.f[keyDataMouse] = rezVals
            h5w.f[keyLabelMouse] = np.array(rezIdxs)

        # Save to file
        # rezDF.to_hdf(h5outname, mouseDataLabel, mode='a', format='table', data_columns=True)
