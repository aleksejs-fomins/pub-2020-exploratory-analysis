import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from mesostat.utils.system import getfiles_walk
from mesostat.utils.hdf5_io import DataStorage
from mesostat.utils.matlab_helper import loadmat
from mesostat.utils.signals import downsample_int
from lib.sych.data_read import read_neuro_perf


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
        tmp = np.array(h5file['trialTypes'][session])

        if not np.all(tmp[idxsTrial] == -1):
            print('dropping', session, idxsTrial)
            tmp[idxsTrial] = -1

            del h5file['trialTypes'][session]
            h5file['trialTypes'].create_dataset(session, data=tmp)


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

def poly_fit_transform(x, y, ord):
    coeff = np.polyfit(x, y, ord)
    p = np.poly1d(coeff)
    return p(x)


def poly_fit_discrete_residual(y, ord):
    xFake = np.arange(len(y))
    return y - poly_fit_transform(xFake, y, ord)


def poly_fit_discrete_parameter_selection(y, ordMax=5):
    relResidualNorms = []
    normOld = np.linalg.norm(y)
    for ord in range(0, ordMax+1):
        resThis = poly_fit_discrete_residual(y, ord)
        normThis = np.linalg.norm(resThis)
        relResidualNorms += [1 - normThis / normOld]
        normOld = normThis

    return relResidualNorms


def pooled_plot_background_polyfit_residuals(dfRawH5, ordMax=5):
    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'r') as h5file:
            for session in h5file['data'].keys():
                data = h5file['data'][session]

                print(data.shape)

                plt.figure()
                for i in range(48):
                    relres = poly_fit_discrete_parameter_selection(data[:, i], ordMax=ordMax)
                    plt.plot(relres)

                plt.title(session)
                plt.show()


###############################
# Baseline Normalization
###############################

def get_trial_data(h5file, session, tmin=-2, tmax=8, onlySelected=True):
    data = h5file['data'][session]
    startIdxs = np.array(h5file['trialStartIdxs'][session])
    FPS = h5file['data'][session].attrs['FPS']

    # Baseline-subtract data
    # dataSub = np.array([poly_fit_discrete_residual(data[:, iCh], 2) for iCh in range(48)]).T

    preSh = startIdxs + int(tmin * FPS)
    postSh = startIdxs + int(tmax * FPS)
    dataTrials = np.array([data[l:r] for l, r in zip(preSh, postSh)])


    trialTypesAll = np.array(h5file['trialTypes'][session])
    if onlySelected:
        trialIdxs = trialTypesAll >= 0
        dataTrials = dataTrials[trialIdxs]
        if dataTrials.ndim == 1:
            dataTrials = np.array(list(dataTrials))

        trialTypesSelected = trialTypesAll[trialIdxs]
    else:
        trialTypesSelected = trialTypesAll

    t = np.round(np.arange(tmin, tmax, 1 / FPS), 4)
    return t, dataTrials, trialTypesSelected


def check_pre_trial_activity_small(dfRawH5):
    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'r') as h5file:
            for session in list(h5file['data'].keys()):
                t, dataTrials, _ = get_trial_data(h5file, session, tmin=-2, tmax=8)
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


def DFF(x, timesPre, timesPost):
    return x[timesPost] / np.mean(x[timesPre]) - 1


def baseline_normalization(t, data3D, method):
    timesPre = t < 0
    timesPost = t >= 0
    nTimesPost = np.sum(timesPost)

    if method == 'raw':
        return t[timesPost], data3D[:, timesPost]
    else:
        nTrial, _, nChannel = data3D.shape
        dataRez = np.zeros((nTrial, nTimesPost, nChannel))

        if method == 'bn_session':
            for iChannel in range(nChannel):
                dataRez[:, :, iChannel] = DFF(data3D[:, :, iChannel].T, timesPre, timesPost).T
        elif method == 'bn_trial':
            for iTrial in range(nTrial):
                for iChannel in range(nChannel):
                    dataRez[iTrial, :, iChannel] = DFF(data3D[iTrial, :, iChannel], timesPre, timesPost)
        else:
            raise ValueError('Unexpected method', method)

        return t[timesPost], dataRez


def extract_store_trial_data(dfRawH5, baselineMethod='raw', targetFPS=20):
    dataName = 'data_' + baselineMethod

    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'a') as h5file:
            if dataName in h5file.keys():
                del h5file[dataName]
            h5file.create_group(dataName)

            if 'trialTypesSelected' not in h5file.keys():
                h5file.create_group('trialTypesSelected')

            for session in h5file['data'].keys():
                print(session)

                t, dataSession, trialTypesSelected = get_trial_data(h5file, session, tmin=-2, tmax=8)

                tPost, dataTrial = baseline_normalization(t, dataSession[:, :, :48], baselineMethod)

                FPS = h5file['data'][session].attrs['FPS']
                if FPS != targetFPS:
                    print('--downsampling', FPS, dataTrial.shape)

                    nTimesDownsample = FPS // targetFPS
                    dataTrial = dataTrial.transpose((1,0,2))
                    t2, dataTrial = downsample_int(tPost, dataTrial, nTimesDownsample)
                    dataTrial = dataTrial.transpose((1,0,2))

                h5file[dataName].create_dataset(session, data=dataTrial)

                if session not in h5file['trialTypesSelected'].keys():
                    h5file['trialTypesSelected'].create_dataset(session, data=trialTypesSelected)