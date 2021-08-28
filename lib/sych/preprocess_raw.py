import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from mesostat.utils.system import getfiles_walk
from mesostat.utils.hdf5_io import DataStorage
from mesostat.utils.matlab_helper import loadmat
from mesostat.utils.signals.resample import downsample_int
from mesostat.utils.pandas_helper import pd_is_one_row, pd_query
from lib.sych.data_read import read_neuro_perf
import lib.preprocessing.polyfit as polyfit
from mesostat.utils.signals.fit import polyfit_transform


def h5_overwrite_group(h5file, groupName, **kwargs):
    if groupName in h5file.keys():
        del h5file[groupName]
    h5file.create_group(groupName, **kwargs)

###########################
# Reading raw files
###########################

def find_all_substring_locations(s, subs):
    start = 0
    while True:
        start = s.find(subs, start)
        if start == -1: return
        yield start
        start += len(subs)  # use start += 1 to find overlapping matches


def find_all_substring_rows(l, subs):
    return [i for i in range(len(l)) if subs in l[i]]


def raw_get_files_df(fpathData):
    fileswalk = getfiles_walk(fpathData, ['.lvm'])

    # Convert to pandas
    df = pd.DataFrame(fileswalk, columns=['path', 'fname'])

    # Drop all LVM files that are not of correct format
    df = df[df['fname'].str.contains("mvg")]

    df['session'] = [os.path.basename(path) for path in df['path']]
    df['path'] = [os.path.join(path, fname) for path, fname in zip(df['path'], df['fname'])]
    df['mousename'] = [os.path.splitext(fname)[0] for fname in df['fname']]

    return df.drop('fname', axis=1)


def raw_parse_header(headerText):
    dataHeader = headerText.replace("\t", " ").split('\n')
    dataHeader = [d.strip() for d in dataHeader]
    idxsDate = find_all_substring_rows(dataHeader, "Date ")
    idxsTime = find_all_substring_rows(dataHeader, "Time ")

    dateThis = dataHeader[idxsDate[-1]].split(' ')[-1]
    timeThis = dataHeader[idxsTime[-1]].split(' ')[-1][:15]  # Too many decimal points in seconds bad
    return datetime.strptime(dateThis + ' ' + timeThis, '%Y/%m/%d %H:%M:%S.%f')


def raw_parse_main(mainText):
    dataMain = mainText.replace("\t", " ").split('\n')[2:]
    arrValue = np.array([s.split(' ')[-1] for s in dataMain if len(s) > 0], dtype=float)

    nChannel = 49
    nEntry = len(arrValue)
    nTimestep = len(arrValue) // 49
    nRemainder = len(arrValue) % 49

    if nRemainder != 0:
        raise IOError("Unexpected array length", nEntry)

    return arrValue.reshape((nTimestep, nChannel))


def raw_pool_data(dfFiles):
    for mousename in set(dfFiles['mousename']):
        ds = DataStorage('raw_' + mousename + '.h5')
        rows = dfFiles[dfFiles['mousename'] == mousename]

        for idx, row in rows.iterrows():
            print('Processing', mousename, row['session'])

            with open(row['path'], 'r') as f:
                data = f.read()

                headerEndKey = "***End_of_Header***"
                splitIdx = list(find_all_substring_locations(data, headerEndKey))[-1] + len(headerEndKey)

                dateTimeThis = raw_parse_header(data[:splitIdx])

                caIndMat = raw_parse_main(data[splitIdx:])

                attrsDict = {
                    'mousename': mousename,
                    'metric': "raw",
                    'target_dim': "(timesteps, channels)",
                    'datetime': dateTimeThis
                }

                ds.save_data(row['session'], caIndMat, attrsDict)


###########################
# Adjusting pooled files
###########################

def pooled_get_files_df(fpath):
    fileswalk = getfiles_walk(fpath, ['raw', '.h5'])
    df = pd.DataFrame(fileswalk, columns=['path', 'fname'])
    df['mousename'] = [os.path.splitext(f)[0][4:] for f in df['fname']]
    df['path'] = [os.path.join(path, fname) for path, fname in zip(df['path'], df['fname'])]
    return df.drop('fname', axis=1)


def pooled_move_data_subfolder(dfRawH5):
    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'a') as h5file:
            if 'data' not in h5file.keys():
                h5file.create_group("data")

            for key in h5file.keys():
                if key != 'data':
                    print(key)
                    session = ''.join(list(h5file[key].attrs['name']))
                    h5file.move(key, 'data/' + session)


def pooled_move_sanity_check(dfRawH5):
    # Sanity check
    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'r') as h5file:
            for session in h5file['data'].keys():
                print(session, h5file['data'][session].shape)


def pooled_mark_trial_starts_ends(dfRawH5):
    for idx, row in dfRawH5.iterrows():
        print(row['mousename'])

        with h5py.File(row['path'], 'a') as h5file:
            h5_overwrite_group(h5file, 'trialStartIdxs')
            h5_overwrite_group(h5file, 'interTrialStartIdxs')

            for session in list(h5file['data'].keys()):
                print(session)

                traceThis = h5file['data'][session][:, -1]
                traceBin = (traceThis > 2).astype(int)
                traceDT = traceBin[1:] - traceBin[:-1]

                idxTrialStart = np.where(traceDT == 1)[0] + 1
                idxIntervStart = np.hstack(([0], np.where(traceDT == -1)[0] + 1))

                nTrial = len(idxTrialStart)
                nInterv = len(idxIntervStart)

                if nTrial == nInterv:
                    idxIntervStart = np.hstack((idxIntervStart, [len(traceThis)]))
                    nInterv += 1

                tTrial = idxIntervStart[1:] - idxTrialStart

                FPS = 20 if np.median(tTrial) < 250 else 40

                h5file['trialStartIdxs'].create_dataset(session, data=idxTrialStart)
                h5file['interTrialStartIdxs'].create_dataset(session, data=idxIntervStart)
                h5file['data'][session].attrs['FPS'] = FPS

    #             print(nTrial, nInterv, FPS)
    #             print('low', tTrial[tTrial < 8 * FPS] / FPS)
    #             print('high', tTrial[tTrial > 12 * FPS] / FPS)


def pooled_get_path_session(dfRawH5, session):
    mousename = session[:5]
    row = dfRawH5[dfRawH5['mousename'] == mousename]
    return list(row['path'])[0]


###########################
# Adding channel labels
###########################

def channel_labels_get_files_df(fpath):
    fileswalk = getfiles_walk(fpath, ['channel_labels.mat'])
    df = pd.DataFrame(fileswalk, columns=['path', 'fname'])

    df['mousename'] = [os.path.basename(p) for p in df['path']]
    df['path'] = [os.path.join(path, fname) for path, fname in zip(df['path'], df['fname'])]
    return df.drop('fname', axis=1)


def pooled_mark_channel_labels(dfRawH5, dfLabels):
    for idx, row in dfLabels.iterrows():
        print(row['mousename'])

        M = loadmat(row['path'])

        rowH5 = dfRawH5[dfRawH5['mousename'] == row['mousename']]
        pathH5 = list(rowH5['path'])[0]

        with h5py.File(pathH5, 'a') as h5file:
            if 'channelLabels' not in h5file.keys():
                h5file.create_dataset('channelLabels', data=M['channel_labels'].astype('S'))


def update_channel_labels_unique(dfRawH5):
    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'a') as h5file:
            labelsDict = {}
            labels = [l.decode('UTF8') for l in h5file['channelLabels']]
            labelsNew = []

            for l in labels:
                if l not in labelsDict:
                    labelsDict[l] = 0
                    labelsNew += [l]
                else:
                    labelsDict[l] += 1
                    labelsNew += [l + '_' + str(labelsDict[l])]

            del h5file['channelLabels']
            h5file['channelLabels'] = np.array(labelsNew).astype('S')
            print(labelsNew)


###########################
# Adding trial type info
###########################

def orig_neuro_get_files_df(fpath):
    fileswalk = getfiles_walk(fpath, ['behaviorvar.mat'])
    df = pd.DataFrame(fileswalk, columns=['path', 'fname'])

    df['session'] = [os.path.basename(p) for p in df['path']]
    df['mousename'] = [os.path.basename(os.path.dirname(p)) for p in df['path']]
    return df #df.drop('fname', axis=1)


def pooled_mark_trial_types_performance(dfRawH5, dfNeuro):
    keysNeeded = ['iGO', 'iNOGO', 'iFA', 'iMISS']

    for mousename in set(dfNeuro['mousename']):
        rowsOrig = dfNeuro[dfNeuro['mousename'] == mousename]

        rowH5 = dfRawH5[dfRawH5['mousename'] == mousename]
        pathH5 = list(rowH5['path'])[0]

        with h5py.File(pathH5, 'a') as h5file:
            # Store trial type keys
            if 'trialTypeNames' not in h5file.keys():
                h5file.create_dataset('trialTypeNames', data=np.array(keysNeeded).astype('S'))

            h5_overwrite_group(h5file, 'trialTypes')
            h5_overwrite_group(h5file, 'performance')

            for idx, rowOrig in rowsOrig.iterrows():
                session = rowOrig['session']

                _, behavior, performance = read_neuro_perf(rowOrig['path'], verbose=False)

                fixint = lambda v: v if not isinstance(v, int) else np.array([v])
                behavior = {k: fixint(v) for k, v in behavior.items()}

                # Find all behaviour keys used in this session
                keysLst = set([key for key in behavior.keys() if len(behavior[key]) > 0])
                keysLst -= {'trials'}
                keysLst = list(sorted(keysLst))

                # Test if multiple keys are assigned to the same trial - currently not possible under our method
                #             for keyA in keysLst:
                #                 for keyB in keysLst:
                #                     if keyA != keyB:
                #                         inter = set(behavior[keyA]).intersection(set(behavior[keyB]))
                #                         if len(inter) > 0:
                #                             print(keyA, keyB, len(inter))


                minTrial = np.min([np.min(behavior[k]) for k in keysLst if len(behavior[k]) > 0])
                maxTrial = np.max([np.max(behavior[k]) for k in keysLst if len(behavior[k]) > 0])

                nTrialsExp = len(h5file['trialStartIdxs'][session])

                # Enumerate all required keys, and set values
                enumArr = np.full(maxTrial, -1, dtype=int)
                for i, key in enumerate(keysNeeded):
                    if key in behavior.keys():
                        idxs = (behavior[key] - 1).astype(int)
                        assert np.all(enumArr[idxs] == -1)
                        enumArr[idxs] = i

                print(session, np.sum(enumArr == -1), minTrial, maxTrial, nTrialsExp, maxTrial == nTrialsExp)

                # Crop number of trials if behaviour or raw data have more than the other
                if maxTrial != nTrialsExp:
                    minNTrial = min(maxTrial, nTrialsExp)

                    print('--cropping to smallest', minNTrial)
                    enumArr = enumArr[:minNTrial]

                h5file['trialTypes'].create_dataset(session, data=enumArr)
                h5file['performance'].create_dataset(session, data=performance)


# def pooled_mark_trial_types_performance(dfRawH5, dfNeuro):
#     keysNeeded = ['iGO', 'iNOGO', 'iFA', 'iMISS']
#
#     for mousename in set(dfNeuro['mousename']):
#         rowsOrig = dfNeuro[dfNeuro['mousename'] == mousename]
#
#         rowH5 = dfRawH5[dfRawH5['mousename'] == mousename]
#         pathH5 = list(rowH5['path'])[0]
#
#         with h5py.File(pathH5, 'a') as h5file:
#             # Store trial type keys
#             if 'trialTypeNames' not in h5file.keys():
#                 h5file.create_dataset('trialTypeNames', data=np.array(keysNeeded).astype('S'))
#
#             if 'trialTypes' in h5file.keys():
#                 del h5file['trialTypes']
#             h5file.create_group('trialTypes')
#
#             for idx, rowOrig in rowsOrig.iterrows():
#                 session = rowOrig['session']
#
#                 pwd = os.path.join(rowOrig['path'], 'behaviorvar.mat')
#
#                 behavior = loadmat(pwd)
#                 #         behavior['trials'] = merge_dicts([matstruct2dict(obj) for obj in behavior['trials']])
#                 fixint = lambda v: v if not isinstance(v, int) else np.array([v])
#                 behavior = {k: fixint(v) for k, v in behavior.items()}
#
#                 # Find all behaviour keys used in this session
#                 keysLst = set([key for key in behavior.keys() if len(behavior[key]) > 0])
#                 keysLst -= set(['trials'])
#                 keysLst = list(sorted(keysLst))
#
#                 # Test if multiple keys are assigned to the same trial - currently not possible under our method
#                 #             for keyA in keysLst:
#                 #                 for keyB in keysLst:
#                 #                     if keyA != keyB:
#                 #                         inter = set(behavior[keyA]).intersection(set(behavior[keyB]))
#                 #                         if len(inter) > 0:
#                 #                             print(keyA, keyB, len(inter))
#
#
#                 minTrial = np.min([np.min(behavior[k]) for k in keysLst if len(behavior[k]) > 0])
#                 maxTrial = np.max([np.max(behavior[k]) for k in keysLst if len(behavior[k]) > 0])
#
#                 nTrialsExp = len(h5file['trialStartIdxs'][session])
#
#                 # Enumerate all required keys, and set values
#                 enumArr = np.full(maxTrial, -1, dtype=int)
#                 for i, key in enumerate(keysNeeded):
#                     if key in behavior.keys():
#                         idxs = (behavior[key] - 1).astype(int)
#                         assert np.all(enumArr[idxs] == -1)
#                         enumArr[idxs] = i
#
#                 print(session, np.sum(enumArr == -1), minTrial, maxTrial, nTrialsExp, maxTrial == nTrialsExp)
#
#                 # Crop number of trials if behaviour or raw data have more than the other
#                 if maxTrial != nTrialsExp:
#                     minNTrial = min(maxTrial, nTrialsExp)
#
#                     print('--cropping to smallest', minNTrial)
#                     enumArr = enumArr[:minNTrial]
#
#                 h5file['trialTypes'].create_dataset(session, data=enumArr)


def pooled_trunc_trial_starts_ntrials(dfRawH5):
    def trunc_h5(grp, key, newLen):
        data = np.copy(grp[key])
        data = data[:newLen]
        del grp[key]
        grp.create_dataset(key, data=data)

    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'a') as h5file:
            for session in h5file['trialTypes'].keys():
                nTrialBeh = len(h5file['trialTypes'][session])
                nTrialIdx = len(h5file['trialStartIdxs'][session])
                nInterIdx = len(h5file['interTrialStartIdxs'][session])

                print(session, nTrialBeh, nTrialIdx, nInterIdx)
                if nTrialIdx > nTrialBeh:
                    print('-- correcting')
                    trunc_h5(h5file['trialStartIdxs'], session, nTrialBeh)
                    trunc_h5(h5file['interTrialStartIdxs'], session, nTrialBeh + 1)


def pooled_trial_length_summary_excel(dfRawH5):
    rezLst = []

    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'a') as h5file:
            for session in h5file['trialTypes'].keys():
                FPS = h5file['data'][session].attrs['FPS']
                types = np.array(h5file['trialTypes'][session])
                starts = h5file['trialStartIdxs'][session]
                intervs = h5file['interTrialStartIdxs'][session]
                lens = intervs[1:] - starts

                for iType, trialType in enumerate(h5file['trialTypeNames']):
                    trialTypeDec = trialType.decode("utf-8")

                    idxs = types == iType
                    lensThis = lens[idxs]
                    lenIdxsGlob = np.arange(len(lens))[idxs]

                    if len(lensThis) > 0:
                        tMin = np.min(lensThis) / FPS
                        tMax = np.max(lensThis) / FPS
                        aMin = lenIdxsGlob[np.argmin(lensThis)] + 1
                        aMax = lenIdxsGlob[np.argmax(lensThis)] + 1

                        rezLst += [[session, trialTypeDec, len(lensThis), tMin, tMax, aMin, aMax]]
                    else:
                        rezLst += [[session, trialTypeDec, None, None, None, None, None]]

    dfTest = pd.DataFrame(rezLst, columns=['session', 'type', 'nTrial', 'Tmin', 'Tmax', 'idxMin', 'idxMax'])
    dfTest.to_excel('test.xlsx')


###########################
# Comparing with behaviour timing files
###########################

def behav_timing_get_files_df(fpathData):
    fileswalk = getfiles_walk(fpathData, ['mvg', '.txt'])

    # Convert to pandas
    df = pd.DataFrame(fileswalk, columns=['path', 'fname'])

    # Drop all LVM files that are not of correct format
#     df = df[df['fname'].str.contains("mvg")]

    df['session'] = [os.path.basename(os.path.dirname(path)) for path in df['path']]
    df['path'] = [os.path.join(path, fname) for path, fname in zip(df['path'], df['fname'])]
    df['mousename'] = [session[:5] for session in df['session']]

    return df.drop('fname', axis=1)


def behav_timing_read_get_trial_lengths(dfRawH5, dfBehavTiming):
    for idxData, rowData in dfRawH5.iterrows():
        with h5py.File(rowData['path'], 'a') as h5file:
            rowsBehavThis = dfBehavTiming[dfBehavTiming['mousename'] == rowData['mousename']]

            if 'trialDurationBehavior' not in h5file.keys():
                h5file.create_group('trialDurationBehavior')

            for idxBehav, rowBehav in rowsBehavThis.iterrows():
                print(rowBehav['session'])

                dfFile = pd.read_csv(rowBehav['path'], sep='\t')
                dfFile['datetime'] = [datetime.strptime(d + ' ' + t, '%d.%m.%Y %H:%M:%S.%f') for d,t in zip(dfFile['Date'], dfFile['Time'])]

                datetimesStart = dfFile[dfFile['Event'] == 'Begin Trial / Recording']['datetime']
                datetimesEnd = dfFile[dfFile['Event'] == 'End Trial ']['datetime']

                # trialLengths = list(datetimesEnd - datetimesStart)
                trialLengths = [(r - l).total_seconds() for r,l in zip(datetimesEnd, datetimesStart)]

                h5file['trialDurationBehavior'].create_dataset(rowBehav['session'], data=trialLengths)


def behav_timing_compare_neuro(dfRawH5):
    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'a') as h5file:
            for session in list(h5file['data'].keys()):
                print(session)

                trialStartIdxs = h5file['trialStartIdxs'][session]
                intervStartIdxs = h5file['interTrialStartIdxs'][session]
                FPS = h5file['data'][session].attrs['FPS']
                trialDurationBehav = h5file['trialDurationBehavior'][session]
                trialDurationNeuro = (intervStartIdxs[1:] - trialStartIdxs) / FPS

                nTrialBehav = len(trialDurationBehav)
                nTrialNeuro = len(trialDurationNeuro)
                if nTrialBehav > nTrialNeuro:
                    trialDurationBehav = trialDurationBehav[:nTrialNeuro]

                tMin = min(np.min(trialDurationBehav), np.min(trialDurationNeuro))
                tMax = max(np.max(trialDurationBehav), np.max(trialDurationNeuro))
                xfake = np.linspace(tMin, tMax, 20)

                plt.figure()
                plt.plot(xfake, xfake, '--', color='gray')
                plt.plot(trialDurationBehav, trialDurationNeuro, '.')
                plt.xlabel('duration (behaviour), s')
                plt.ylabel('duration (TTL), s')
                plt.title(session)
                plt.savefig(session+'.png')
                plt.close()


###########################
# Dropping bad sessions and trials
###########################

def drop_session(dfRawH5, session):
    path = pooled_get_path_session(dfRawH5, session)

    groupKeys = ['data', 'interTrialStartIdxs', 'trialDurationBehavior', 'trialStartIdxs']

    with h5py.File(path, 'a') as h5file:
        for key in groupKeys:
            if session in h5file[key].keys():
                print('deleting', key, session)
                del h5file[key][session]


def drop_sessions_not_in_neuro(dfNeuro, dfRawH5):
    sessionsNeuro = list(dfNeuro['session'])
    sessionsDrop = []

    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'r') as h5file:
            for session in list(h5file['data'].keys()):
                if session not in sessionsNeuro:
                    sessionsDrop += [session]

    for session in sessionsDrop:
        drop_session(dfRawH5, session)


def drop_trials(dfRawH5, session, idxsTrial):
    path = pooled_get_path_session(dfRawH5, session)

    with h5py.File(path, 'a') as h5file:
        trialTypes = np.array(h5file['trialTypes'][session])

        # Update trialTypes
        if not np.all(trialTypes[idxsTrial] == -1):
            print('dropping', session, idxsTrial)
            trialTypes[idxsTrial] = -1

            del h5file['trialTypes'][session]
            h5file['trialTypes'].create_dataset(session, data=trialTypes)
        else:
            print('Trials already dropped, ignoring:', idxsTrial)

        # Update trialTypesSelected
        if 'trialTypesSelected' not in h5file.keys():
            print("trialTypesSelected not yet computed")
        elif session not in h5file['trialTypesSelected'].keys():
            print("trialTypesSelected not yet computed for session", session)
        else:
            trialTypesSelected = np.array(h5file['trialTypesSelected'][session])
            idxsSelected = trial_types_selected_idxs(trialTypes)
            trialTypesSelectedNew = trialTypes[idxsSelected]

            if np.array_equal(trialTypesSelected, trialTypesSelectedNew):
                print("trialTypesSelected already consistent (size ", len(trialTypesSelected), "), no need to update")
            else:
                print("Updating trialTypesSelected to", trialTypesSelectedNew)
                del h5file['trialTypesSelected'][session]
                h5file['trialTypesSelected'].create_dataset(session, data=trialTypesSelectedNew)


# FIXME: How can a trial be short if all trials are set to 8s ???
def find_short_trials(dfRawH5):
    rezDict = {}

    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'r') as h5file:
            for session in list(h5file['data'].keys()):
                trialIdxs = np.array(h5file['trialTypes'][session]) >= 0

                data = h5file['data'][session]
                startIdxs = np.array(h5file['trialStartIdxs'][session])
                FPS = h5file['data'][session].attrs['FPS']

                postSh = startIdxs + int(8 * FPS)
                dataTrialsLst = [data[l:r, :48] for l, r in zip(startIdxs, postSh)]

                # Testing for short trials
                trialLengths = np.array([len(d) for d in dataTrialsLst])
                idxsShort = np.where(trialLengths < np.max(trialLengths))[0]

                if len(idxsShort) > 0:
                    print(session, 'short trials', idxsShort)

                    trialIdxEnumAll = np.arange(len(trialIdxs))
                    idxsShortGlob = trialIdxEnumAll[idxsShort]
                    rezDict[session] = np.array(idxsShortGlob)

    return rezDict


def find_large_trials(dfRawH5):
    rezDict = {}

    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'r') as h5file:
            for session in list(h5file['data'].keys()):
                trialIdxs = np.array(h5file['trialTypes'][session]) >= 0
                trialIdxEnumAll = np.arange(len(trialIdxs))

                data = h5file['data'][session]
                startIdxs = np.array(h5file['trialStartIdxs'][session])
                FPS = h5file['data'][session].attrs['FPS']

                postSh = startIdxs + int(8 * FPS)
                dataTrials = np.array([data[l:r, :48] for l, r in zip(startIdxs, postSh)])
                dataTrialsFilter = np.array(list(dataTrials[trialIdxs]))

                # print(session, data.shape, dataTrialsFilter.shape, np.sum(trialIdxs))

                trialMag = np.linalg.norm(dataTrialsFilter, axis=(1,2)) / np.prod(dataTrialsFilter.shape[1:])

                idxsLarge = np.where(trialMag > 2 * np.median(trialMag))[0]

                if len(idxsLarge) > 0:
                    idxsLargeGlob = trialIdxEnumAll[trialIdxs][idxsLarge]
                    typesLarge = np.array(h5file['trialTypes'][session])[idxsLargeGlob]
                    typeNamesLarge = np.array(h5file['trialTypeNames'])[typesLarge]
                    typeNamesLarge = [t.decode('UTF8') for t in typeNamesLarge]

                    print(session, np.array(idxsLargeGlob)+1, typeNamesLarge)

                    rezDict[session] = np.array(idxsLargeGlob)

#                     for idx in idxsLargeGlob:
#                         plt.figure()
#                         for iCh in range(48):
#                             plt.semilogy(x, dataTrials[idx, :, iCh])
#                         plt.show()
    return rezDict


###########################
# Background subtraction
###########################


def pooled_plot_background_polyfit_residuals(dfRawH5, ordMax=5):
    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'r') as h5file:
            for session in h5file['data'].keys():
                data = h5file['data'][session]

                print(data.shape)

                plt.figure()
                for i in range(48):
                    relres = polyfit.poly_fit_discrete_parameter_selection(data[:, i], ordMax=ordMax)
                    plt.plot(relres)

                plt.title(session)
                plt.show()


def get_sessions(dfRawH5, mousename):
    row = pd_is_one_row(pd_query(dfRawH5, {'mousename' : mousename}))[1]
    with h5py.File(row['path'], 'r') as h5file:
        return list(h5file['data'].keys())


def plot_raw(dfRawH5, session, iChannel, onlyTrials=False, onlySelected=False, figsize=(12,4)):
    path = pooled_get_path_session(dfRawH5, session)
    with h5py.File(path, 'r') as h5file:
        data = np.copy(h5file['data'][session][:, iChannel])
        if onlyTrials:
            x, data = data_mark_trials(h5file, session, data=data, tmin=-2, tmax=8, onlySelected=onlySelected)
        else:
            x = np.arange(len(data))

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, data)
        ax.set_ylabel(iChannel)
        plt.show()


# Fit polynomial to data, return fitted result
# If partial specified, fit different polynomials to left and right side of the data
def poly_fit_partial(x, data, ord, xPart):
    if xPart is None:
        return polyfit.poly_fit_transform(x, data, ord)
    else:
        xThr, ordL, ordR = xPart
        idxsL = x < xThr
        idxsR = ~idxsL
        rezL = polyfit.poly_fit_transform(x[idxsL], data[idxsL], ordL)
        rezR = polyfit.poly_fit_transform(x[idxsR], data[idxsR], ordR)
        return np.hstack([rezL, rezR])


def poly_view_fit(dfRawH5, session, channel, ord, onlyTrials=False, onlySelected=False, xPart=None):
    path = pooled_get_path_session(dfRawH5, session)
    with h5py.File(path, 'r') as h5file:
        data = np.copy(h5file['data'][session][:, channel])
        if onlyTrials:
            x, data = data_mark_trials(h5file, session, data=data, tmin=-2, tmax=8, onlySelected=onlySelected)
        else:
            x = np.arange(len(data))

        print(len(data))

        dataFit = poly_fit_partial(x, data, ord, xPart) # polyfit.poly_fit_transform(x, data, ord)
        fig, ax = plt.subplots(ncols=3, figsize=(12,4))
        ax[0].plot(x, data)
        ax[0].plot(x, dataFit)
        ax[1].plot(x, data-dataFit)
        ax[2].plot(x, data/dataFit - 1)
        plt.show()


###############################
# Baseline Normalization
###############################

# Return indices of selected trial types
def trial_types_selected_idxs(trialTypes):
    return trialTypes >= 0


# Return data partitioned into trials
def data_partition_trials(h5file, session, data=None, tmin=-2, tmax=8, onlySelected=True):
    if data is None:
        data = np.copy(h5file['data'][session])
    startIdxs = np.array(h5file['trialStartIdxs'][session])
    FPS = h5file['data'][session].attrs['FPS']
    trialTypesAll = np.array(h5file['trialTypes'][session])

    minFPS = int(tmin * FPS)
    maxFPS = int(tmax * FPS)
    preSh = startIdxs + minFPS
    postSh = startIdxs + maxFPS
    dataTrials = [data[l:r] for l, r in zip(preSh, postSh)]

    # Pad some trials if they cut abruptly
    trgLen = maxFPS - minFPS
    for i in range(len(dataTrials)):
        trueLen = len(dataTrials[i])
        if trueLen != trgLen:
            print('--Warning: trial', i, 'too short have =', trueLen, 'need', trgLen, '; padding' )
            dataTrials[i] = np.vstack([dataTrials[i], np.zeros((trgLen-trueLen, data.shape[1]))])
    dataTrials = np.array(dataTrials)

    if onlySelected:
        trialIdxs = trial_types_selected_idxs(trialTypesAll)
        dataTrials = dataTrials[trialIdxs]
        if dataTrials.ndim == 1:
            dataTrials = np.array(list(dataTrials))

        trialTypesSelected = trialTypesAll[trialIdxs]
    else:
        trialTypesSelected = trialTypesAll

    t = np.round(np.arange(tmin, tmax, 1 / FPS), 4)
    return t, dataTrials, trialTypesSelected


# Return a single time-sequence with non-trial datapoints dropped
# Return the timesteps axis to note which datapoints were kept
#   Note: we do not convert timestep idxs to times, so they can be comfortably used later for indexing
def data_mark_trials(h5file, session, data=None, tmin=-2, tmax=8, onlySelected=True):
    if data is None:
        data = np.copy(h5file['data'][session])
    startIdxs = np.array(h5file['trialStartIdxs'][session])
    FPS = h5file['data'][session].attrs['FPS']
    trialTypesAll = np.array(h5file['trialTypes'][session])

    nData = len(data)
    minFPS = int(tmin * FPS)
    maxFPS = int(tmax * FPS)

    if onlySelected:
        trialIdxs = np.where(trialTypesAll >= 0)[0]
    else:
        trialIdxs = np.arange(len(startIdxs))

    xTrials = []
    yTrials = []
    for iTrial in trialIdxs:
        l = startIdxs[iTrial] + minFPS
        r = np.min([startIdxs[iTrial] + maxFPS, nData])   # Some trials exceed total trial time

        xTrials += [np.arange(l, r)]
        yTrials += [data[l:r]]

    return np.concatenate(xTrials, axis=0), np.concatenate(yTrials, axis=0)


def check_pre_trial_activity_small(dfRawH5):
    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'r') as h5file:
            for session in list(h5file['data'].keys()):
                t, dataTrials, _ = data_partition_trials(h5file, session, tmin=-2, tmax=8)
                dataMu = np.mean(dataTrials, axis=0)

                # Subtract mean pre-trial activity over session for each channel
                idxsPre = t < 0
                dataMu = np.array([dataMu[:, iCh] - np.mean(dataMu[idxsPre, iCh]) for iCh in range(48)]).T

                plt.figure()
                for iCh in range(48):
                    plt.plot(t, dataMu[:, iCh])
                plt.title(session)
                plt.savefig(session + '.png')
                plt.close()


def DFF(x, timesPre):
    return x / np.mean(x[timesPre]) - 1


def baseline_normalization(t, data3D):
    timesPre = t < 0

    nTrial, nTimes, nChannel = data3D.shape
    dataRez = np.zeros(data3D.shape)

    for iTrial in range(nTrial):
        for iChannel in range(nChannel):
            dataRez[iTrial, :, iChannel] = DFF(data3D[iTrial, :, iChannel], timesPre)

    return dataRez


def extract_store_trial_data(dfRawH5, xPartMap, targetFPS=20, bgOrd=2,
                             fitOnlySelectedTrials=True, keepExisting=True, targetSessions=None, cropTimestep=None):
    nChannel = 48
    baselineMethods = ['raw', 'bn_session', 'bn_trial']

    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'a') as h5file:
            dataNames = {}
            for baselineMethod in baselineMethods:
                dataName = 'data_' + baselineMethod
                dataNames[baselineMethod] = dataName
                if dataName in h5file.keys():
                    if (not keepExisting) and (targetSessions is None):
                        del h5file[dataName]
                        h5file.create_group(dataName)
                else:
                    h5file.create_group(dataName)

            if 'trialTypesSelected' not in h5file.keys():
                h5file.create_group('trialTypesSelected')

            sessions = h5file['data'].keys()
            if targetSessions is not None:
                sessions = set(sessions).intersection(set(targetSessions))

            for session in sessions:
                print(session)

                doneAll = np.all([session in h5file[dataName].keys() for dataName in dataNames.values()])
                if doneAll and keepExisting:
                    print('all done, skipping')
                else:
                    # Get Data
                    data = np.copy(h5file['data'][session][:, :nChannel])

                    # Get polyfit partition if it is defined
                    xPart = None if session not in xPartMap.keys() else xPartMap[session]

                    # Perform background subtraction
                    if fitOnlySelectedTrials:
                        # Fit polynomial only to parts of the trial that are relevant
                        dataBG = np.copy(data)
                        xIdxs, y = data_mark_trials(h5file, session, data=data, tmin=-2, tmax=8, onlySelected=True)
                        for iChannel in range(nChannel):
                            # yFit = polyfit.poly_fit_transform(xIdxs, y[:, iChannel], bgOrd)
                            yFit = poly_fit_partial(xIdxs, y[:, iChannel], bgOrd, xPart)
                            dataBG[xIdxs, iChannel] = yFit
                    else:
                        # Fit polynomial to entire trial
                        dataBG = np.zeros(data.shape)
                        xIdxs = np.arange(len(data))
                        for iChannel in range(nChannel):
                            # dataBG[:, iChannel] = polyfit.poly_fit_transform(xIdxs, data[:, iChannel], bgOrd)
                            dataBG[:, iChannel] = poly_fit_partial(xIdxs, data[:, iChannel], bgOrd, xPart)

                    dataDict = {
                        'raw' : data - dataBG,
                        'bn_session': data / dataBG - 1,
                        'bn_trial': data,
                    }

                    for baselineMethod, dataMethod in dataDict.items():
                        print('--', baselineMethod)

                        # Slice data into trials
                        # IMPORTANT: must use background-subtracted data
                        t, dataSession, trialTypesSelected = data_partition_trials(h5file, session, data=dataMethod, tmin=-2, tmax=8)

                        # Perform baseline normalization
                        if baselineMethod == 'bn_trial':
                            dataSession = baseline_normalization(t, dataSession)

                        # Downsample the result if does not match target FPS
                        FPS = h5file['data'][session].attrs['FPS']
                        if FPS != targetFPS:
                            print('--downsampling', FPS, dataSession.shape)

                            nTimesDownsample = FPS // targetFPS
                            dataSession = dataSession.transpose((1,0,2))
                            t2, dataSession = downsample_int(t, dataSession, nTimesDownsample)
                            dataSession = dataSession.transpose((1,0,2))

                        # Store trial data
                        h5file[dataNames[baselineMethod]].create_dataset(session, data=dataSession)

                        # Store selected trial types
                        if session not in h5file['trialTypesSelected'].keys():
                            h5file['trialTypesSelected'].create_dataset(session, data=trialTypesSelected)


#####################
# Cleanup
#####################

def fix_adjust_drop_channel(dfRawH5, session, channel, intervLst, valLst, update=False):
    def plotfit(data, ax):
        fit = polyfit_transform(np.arange(len(data)), data, 15)
        print(np.linalg.norm(data - fit))

        ax.plot(data)
        ax.plot(fit)

    # Read data
    pwd = pooled_get_path_session(dfRawH5, session)
    with h5py.File(pwd, 'r') as h5file:
        data = np.copy(h5file['data'][session])

    dataCh = data[:, channel]

    fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
    plotfit(dataCh, ax[0])

    for iInt in range(len(intervLst)-1):
        dataCh[intervLst[iInt]:intervLst[iInt+1]] += valLst[iInt]

    plotfit(dataCh, ax[1])
    plt.show()

    if update:
        data[:, channel] = dataCh
        with h5py.File(pwd, 'a') as h5file:
            del h5file['data'][session]
            h5file['data'].create_dataset(session, data=data)


def postprocess_crop_bad_trials(dfRawH5, session, idxTrialsKeep):
    pwd = pooled_get_path_session(dfRawH5, session)
    with h5py.File(pwd, 'a') as h5file:
        for key in ['trialTypesSelected', 'data_raw', 'data_bn_trial', 'data_bn_session']:
            if session in h5file[key].keys():
                data = np.copy(h5file[key][session])

                print(key, 'Cropping', len(data), 'to', len(idxTrialsKeep))
                data = data[idxTrialsKeep]

                del h5file[key][session]
                h5file[key].create_dataset(session, data=data)


def get_trial_idxs_by_interval(dfRawH5, session, idxMinSession, idxMaxSession, tMinTrial, tMaxTrial, inside=False):
    pwd = pooled_get_path_session(dfRawH5, session)
    with h5py.File(pwd, 'r') as h5file:
        FPS = h5file['data'][session].attrs['FPS']
        trialStartIdxs = np.copy(h5file['trialStartIdxs'][session])

    trialLeftIdxs = trialStartIdxs + tMinTrial * FPS
    trialRightIdxs = trialStartIdxs + tMaxTrial * FPS

    rez = []
    for iTrial in range(len(trialStartIdxs)):
        fitLeft = trialLeftIdxs[iTrial] >= idxMinSession
        fitRight = trialRightIdxs[iTrial] <= idxMaxSession
        fitBoth = fitLeft and fitRight

        if inside == fitBoth:
        # if not (fitLeft and fitRight):
            rez += [iTrial]

    # print(trialLeftIdxs[rez])
    print('Selected', len(rez), 'of', len(trialStartIdxs), 'trials')
    return rez
