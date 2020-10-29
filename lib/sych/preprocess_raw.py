import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from mesostat.utils.system import getfiles_walk
from mesostat.utils.hdf5_io import DataStorage
from mesostat.utils.matlab_helper import loadmat #, matstruct2dict
#from lib.sych.data_read import read_neuro_perf


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
                grp = h5file.create_group("data")

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
            if 'trialStartIdxs' not in h5file.keys():
                grp = h5file.create_group('trialStartIdxs')
            if 'interTrialStartIdxs' not in h5file.keys():
                grp = h5file.create_group('interTrialStartIdxs')

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

                FPS = 20 if np.median(tTrial) < 200 else 40

                h5file['trialStartIdxs'].create_dataset(session, data=idxTrialStart)
                h5file['interTrialStartIdxs'].create_dataset(session, data=idxIntervStart)
                h5file['data'][session].attrs['FPS'] = FPS

    #             print(nTrial, nInterv, FPS)
    #             print('low', tTrial[tTrial < 8 * FPS] / FPS)
    #             print('high', tTrial[tTrial > 12 * FPS] / FPS)


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


def pooled_mark_trial_types(dfRawH5, dfNeuro):
    keysNeeded = ['iGO', 'iNOGO', 'iFA', 'iMISS']

    for mousename in set(dfNeuro['mousename']):
        rowsOrig = dfNeuro[dfNeuro['mousename'] == mousename]

        rowH5 = dfRawH5[dfRawH5['mousename'] == mousename]
        pathH5 = list(rowH5['path'])[0]

        with h5py.File(pathH5, 'a') as h5file:
            # Store trial type keys
            if 'trialTypeNames' not in h5file.keys():
                h5file.create_dataset('trialTypeNames', data=np.array(keysNeeded).astype('S'))

            if 'trialTypes' in h5file.keys():
                del h5file['trialTypes']
            h5file.create_group('trialTypes')

            for idx, rowOrig in rowsOrig.iterrows():
                session = rowOrig['session']

                pwd = os.path.join(rowOrig['path'], 'behaviorvar.mat')

                behavior = loadmat(pwd)
                #         behavior['trials'] = merge_dicts([matstruct2dict(obj) for obj in behavior['trials']])
                fixint = lambda v: v if not isinstance(v, int) else np.array([v])
                behavior = {k: fixint(v) for k, v in behavior.items()}

                # Find all behaviour keys used in this session
                keysLst = set([key for key in behavior.keys() if len(behavior[key]) > 0])
                keysLst -= set(['trials'])
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

                print(nTrialBeh, nTrialIdx, nInterIdx)
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
                lens = starts - intervs[:-1]

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
    fileswalk = getfiles_walk(fpathData, ['behavior', '.txt'])

    # Convert to pandas
    df = pd.DataFrame(fileswalk, columns=['path', 'fname'])

    # Drop all LVM files that are not of correct format
#     df = df[df['fname'].str.contains("mvg")]

    df['session'] = [os.path.basename(os.path.dirname(path)) for path in df['path']]
    df['path'] = [os.path.join(path, fname) for path, fname in zip(df['path'], df['fname'])]
    df['mousename'] = [session[:4] for session in df['session']]

    return df.drop('fname', axis=1)


def behav_timing_read_get_trial_lengths(dfBehavTiming):
    for idx, row in dfBehavTiming.iterrows():
        dfFile = pd.read_csv(row['path'], sep='\t')
        dfFile['datetime'] = [datetime.strptime(d + ' ' + t, '%d.%m.%Y %H:%M:%S.%f') for d,t in zip(dfFile['Date'], dfFile['Time'])]

        datetimesStart = np.array(dfFile[dfFile['Event'] == 'Begin Trial / Recording']['datetime'])
        datetimesEnd = np.array(dfFile[dfFile['Event'] == 'End Trial']['datetime'])

        trialLengths = datetimesEnd - datetimesStart
        trialLengths = [t.total_seconds() for t in trialLengths]

        print(row['session'], trialLengths)


###########################
# Background subtraction
###########################

def poly_fit_transform(x, y, ord):
    coeff = np.polyfit(x, y, ord)
    p = np.poly1d(coeff)
    return p(x)


def poly_fit_discrete_residuals(y, ordMax=5):
    x = np.arange(len(y))

    relResiduals = []
    resOld = np.linalg.norm(y)
    for ord in range(0, ordMax+1):
        yfit = poly_fit_transform(x, y, ord)
        resThis = np.linalg.norm(y - yfit)
        relResiduals += [1 - resThis / resOld]
        resOld = resThis

    return relResiduals


def pooled_plot_background_polyfit_residuals(dfRawH5, ordMax=5):
    for idx, row in dfRawH5.iterrows():
        with h5py.File(row['path'], 'r') as h5file:
            for session in h5file['data'].keys():
                data = h5file['data'][session]

                print(data.shape)

                plt.figure()
                for i in range(48):
                    relres = poly_fit_discrete_residuals(data[:, i], ordMax=ordMax)
                    plt.plot(relres)

                plt.title(session)
                plt.show()
