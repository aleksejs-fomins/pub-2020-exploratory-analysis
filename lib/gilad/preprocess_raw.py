import os
import h5py
import numpy as np
import pandas as pd
from _collections import defaultdict

import dcimg
import skimage.transform as skt
from IPython.display import display

from mesostat.utils.matlab_helper import loadmat


def _session_from_mousename(session):
    return session[:3]


def get_mice(pwd):
    l1 = os.listdir(pwd)

    # Select mouse folders
    l1 = [l for l in l1 if l[:5] == 'mouse']

    # Add to dataframe
    df = pd.DataFrame()
    df['path_rel'] = l1
    df['mouseIdx'] = [int(l.split('_')[2]) for l in l1]
    df['mousename'] = ['m'+str(i) for i in df['mouseIdx']]

    # Select only late mouse folders
    df = df[df['mouseIdx'] >= 14]

    return df


def get_sessions(pwd, dfMouse):
    dfSessions = pd.DataFrame()

    for idx, row in dfMouse.iterrows():
        pwdMouse = os.path.join(pwd, row['path_rel'])

        # Day folders are just data in numbers
        days = os.listdir(pwdMouse)
        days = [s for s in days if len(s) == 8 and s.isnumeric()]

        for day in days:
            pwdDay = os.path.join(pwdMouse, day)

            # Sessions are labeled alphabetically with a single letter
            sessions = os.listdir(pwdDay)
            sessions = [s for s in sessions if len(s) == 1]

            dfTmp = pd.DataFrame()
            dfTmp['path_rel'] = [os.path.join(row['path_rel'], day, s) for s in sessions]
            dfTmp['mousename'] = row['mousename']
            dfTmp['session'] = [row['mousename'] + '_' + day + '_' + s for s in sessions]

            dfSessions = dfSessions.append(dfTmp)

    return dfSessions


def drop_session(pwdH5, dfSession, session):
    mousename = _session_from_mousename(session)
    h5fname = os.path.join(pwdH5, mousename + '.h5')

    with h5py.File(h5fname, 'a') as h5file:
        if 'metadata' in h5file.keys():
            if session in h5file['metadata'].keys():
                print('Deleting', session, 'from', os.path.basename(h5fname))
                del h5file['metadata'][session]

    return dfSession[dfSession['session'] != session]


# Extract trial types for each trial
# Extract start time of each trial relative to first trial
def parse_metadata_labview(fpath):
    df = pd.read_csv(fpath, sep='\t')
    df['Datetime'] = df['Date'] + df['Time']
    df['Datetime'] = pd.to_datetime(df['Datetime'], infer_datetime_format=True)

    df = df.drop(['Date', 'Time'], axis=1)
    timeStart = df['Datetime'][0]

    rezDict = {'startTime' : [], 'trialType' : [], 'havePuff' : []}

    for iTrial in sorted(list(set(df['Trial']))):
        tmp = df[df['Trial'] == iTrial].reset_index()

        if 'Reward' in list(tmp['Event']):
            tmp = tmp[tmp['Event'] != 'Reward'].reset_index()

        if 'Puff' in list(tmp['Event']):
            rezDict['havePuff'] += [True]
            tmp = tmp[tmp['Event'] != 'Puff'].reset_index()
        else:
            rezDict['havePuff'] += [False]

        if len(tmp) != 9:
            display(tmp)
            raise ValueError('??')

        rezDict['trialType'] += [tmp['Event'][6]]
        rezDict['startTime'] += [(tmp['Datetime'][6] - timeStart).total_seconds()]

    return pd.DataFrame(rezDict)


def get_metadata(pwdH5, pwd, dfSession, rewrite=False):
    for mousename, dfMouse in dfSession.groupby(['mousename']):

        h5fname = os.path.join(pwdH5, mousename + '.h5')

        with h5py.File(h5fname, 'a') as h5file:
            if 'metadata' not in h5file.keys():
                h5file.create_group('metadata')

        for idx, row in dfMouse.iterrows():
            print(row['session'])

            with h5py.File(h5fname, 'a') as h5file:
                if row['session'] in h5file['metadata'].keys():
                    if rewrite:
                        del h5file['metadata'][row['session']]
                    else:
                        print('-- already have', row['session'], 'skipping')
                        continue

            # Stitch path to metadata file
            tmp = row['path_rel'].split('/')
            fname = tmp[0] + '_ses_' + tmp[2] + tmp[1] + '.txt'
            fpath = os.path.join(pwd, row['path_rel'], fname)

            # Read and parse metadata
            assert os.path.isfile(fpath)
            dfMeta = parse_metadata_labview(fpath)
            dfMeta.replace({'Inappropriate Response' : 'FA', 'No Response' : 'Miss'}, inplace=True)

            # Store to H5
            dfMeta.to_hdf(h5fname, '/metadata/'+row['session'])
            print('--', sorted(set(dfMeta['trialType'])))


def validate_trials_ind_from_orig_videos(pwdH5, pwd, dfSession):
    trialTypes = ['Go', 'No Go', 'Miss', 'FA']
    trIndMap = {'tr_100' : 'Go', 'tr_1200' : 'No Go', 'tr_MISS' : 'Miss', 'tr_FA' : 'FA'}

    for idx, row in dfSession.iterrows():
        print(row['session'])

        # Count trial types in metadata
        fpathH5 = os.path.join(pwdH5, row['mousename']+'.h5')
        dfMeta = pd.read_hdf(fpathH5, '/metadata/' + row['session'])
        countsMeta = {tt : (dfMeta['trialType'] == tt).sum() for tt in trialTypes}

        # Count trial types in trials_ind
        fpathTrInd = os.path.join(pwd, row['path_rel'], 'Matt_files', 'trials_ind.mat')
        mTrInd = loadmat(fpathTrInd)
        countsTrInd = {}
        for key, mat in mTrInd.items():
            L = 1 if isinstance(mat, int) else len(mat)
            if key in trIndMap:
                countsTrInd[trIndMap[key]] = L
            else:
                assert L == 0

        # Flip Go and NoGo for latter mice
        if (row['mousename'] == 'm18') or (row['mousename'] == 'm20'):
            countsTrInd['Go'], countsTrInd['No Go'] = countsTrInd['No Go'], countsTrInd['Go']

        # Test the number of trials of each type
        for tt in trialTypes:
            assert countsMeta[tt] == countsTrInd[tt]


def validate_trials_ind_from_reg_videos(pwdH5, pwd, dfSession):
    trialTypes = ['Go', 'No Go', 'Miss', 'FA']
    trIndMap = {'100' : 'Go', '1200' : 'No Go', 'MISS' : 'Miss', 'FA' : 'FA'}

    for mousename, dfMouse in dfSession.groupby(['mousename']):

        # Count number of trials over all sessions
        countsMetaAll = defaultdict(int)
        for idx, row in dfMouse.iterrows():
            fpathH5 = os.path.join(pwdH5, row['mousename'] + '.h5')
            dfMeta = pd.read_hdf(fpathH5, '/metadata/' + row['session'])
            for tt in trialTypes:
                countsMetaAll[tt] += (dfMeta['trialType'] == tt).sum()

        print("Meta", mousename, dict(countsMetaAll))

        # Count number of trials of each type in registered videos
        mouseDir = row['path_rel'].split(os.path.sep)[0]
        pwdReg = os.path.join(pwd, mouseDir, 'Registered_data_ALLEN')

        vidFiles = os.listdir(pwdReg)
        vidFiles = [f for f in vidFiles if f[:4] == 'cond' and 'trial' in f]
        dfVid = pd.DataFrame()
        dfVid['fname'] = vidFiles
        dfVid['type'] = [f.split('_')[1] for f in vidFiles]
        countsRegAll = {trIndMap[t]: (dfVid['type'] == t).sum() for t in set(dfVid['type'])}
        countsRegAll = {t : countsRegAll[t] for t in trialTypes}  # Just to sort
        if (row['mousename'] == 'm18') or (row['mousename'] == 'm20'):
            countsRegAll['Go'], countsRegAll['No Go'] = countsRegAll['No Go'], countsRegAll['Go']

        print("Reg", mousename, countsRegAll)


def read_videos(pwdH5, pwd, dfSession):
    for idx, row in dfSession.iterrows():
        # Get video directory
        pwdVid = os.path.join(pwd, row['path_rel'])

        # Parse video file names
        sessionKey = os.path.basename(os.path.dirname(row['path_rel']))
        sessionFlipKey = sessionKey[:4] + sessionKey[6:] + sessionKey[4:6]  # Lol they flip date format between folders and vids
        vidNames = os.listdir(pwdVid)
        vidNames = [v for v in vidNames if v[:8] == sessionFlipKey]

        # Read videos
        for vidName in vidNames:
            fpathVid = os.path.join(pwdVid, vidName)
            data = dcimg.DCIMGFile(fpathVid)[:]
            dataDS = np.array([skt.downscale_local_mean(d, (2, 2)) for d in data])

            return dataDS

