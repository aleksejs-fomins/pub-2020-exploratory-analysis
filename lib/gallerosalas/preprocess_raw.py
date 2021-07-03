import h5py
import dcimg
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict
import pymatreader
# from scipy.ndimage import affine_transform
import skimage.transform as skt

# from mesostat.utils.arrays import numpy_merge_dimensions
from mesostat.utils.pandas_helper import pd_append_row, pd_query, pd_is_one_row
# from mesostat.utils.matlab_helper import loadmat
from mesostat.stat.performance import accuracy, d_prime

# import lib.preprocessing.polyfit as polyfit
import lib.gallerosalas.preprocess_common as prepcommon

# import mesostat.utils.image_processing as msimg



class preprocess:
    def __init__(self, pathDict):
        # Parse L_modified file
        self.pathLmod = os.path.join(pathDict['Preferences'], 'L_modified.mat')
        prepcommon._testfile(self.pathLmod)
        self.allen = pymatreader.read_mat(self.pathLmod)['L']

        # Parse sessions file
        pathSessions = os.path.join(pathDict['Preferences'], 'sessions_tex.csv')
        self.dfSession = pd.read_csv(pathSessions, sep='\t')

        # with open(pathSessions, 'r') as json_file:
        #     self.sessions = json.load(json_file)

        # Find parse TGT files
        self.dataPaths = pd.DataFrame(
            columns=['mouse', 'day', 'session', 'sessionPath', 'trialIndPath', 'trialStructPath', 'pathActivePassive'])
        self.pathT1 = defaultdict(dict)

        if 'TGT' in pathDict:
            self.find_parse_tdt(pathDict['TGT'])

        # Find parse Overlay
        self.pathRef = {}
        self.pathT2 = {}
        if 'Overlay' in pathDict:
            self.find_parse_overlay(pathDict['Overlay'])


############################
#  Raw data segmentation and pooling
############################

    # Convert session name e.g. 'Session01' to letters like 'a', 'b', 'c'
    def _sessions_to_letters(self, name):
        sessionIdx = int(name[len('session'):]) - 1
        return chr(97 + sessionIdx)

    def get_mice(self):
        return sorted(list(set(self.dataPaths['mouse'])))

    # Find necessary file paths in the TDT folder
    def find_parse_tdt(self, pathTDT):
        for mouseName, dfMouse in self.dfSession.groupby(['mousename']):
            pathMouse = os.path.join(pathTDT, mouseName)
            prepcommon._testdir(pathMouse)

            for dayName, dfDay in dfMouse.groupby(['dateKey']):
                pathDay = os.path.join(pathMouse, 'TDT', dayName, 'widefield_labview')
                pathAllen = os.path.join(pathDay, 'ROI_Allen', 'registration_transform_1510.mat')
                prepcommon._testdir(pathDay)
                prepcommon._testfile(pathAllen)

                self.pathT1[mouseName][dayName] = pathAllen

                for idx, row in dfDay.iterrows():
                    sessionSuffix = row['sessionKey']

                    pathSession = os.path.join(pathDay, sessionSuffix)
                    pathMat = os.path.join(pathSession, 'Matt_files')
                    pathTrialInd = os.path.join(pathMat, 'trials_ind.mat')

                    if mouseName == 'mou_5':
                        matFiles = [f for f in os.listdir(pathMat) if os.path.isfile(os.path.join(pathMat, f))]
                        matFiles = [f for f in matFiles if '.mat' in f]
                        matFiles = [f for f in matFiles if dayName[:4] in f]

                        assert len(matFiles) == 1
                        pathTrialStruct = os.path.join(pathMat, matFiles[0])
                    else:
                        pathTrialStruct = os.path.join(pathMat, dayName + sessionSuffix + '.mat')

                    pathActivePassive = os.path.join(pathMat, 'trials_with_and_wo_initial_moves_OCIA_from_movie.mat')

                    prepcommon._testdir(pathSession)
                    prepcommon._testdir(pathMat)
                    prepcommon._testfile(pathTrialInd)
                    prepcommon._testfile(pathTrialStruct)
                    prepcommon._testfile(pathActivePassive, critical=False)

                    self.dataPaths = pd_append_row(self.dataPaths, [
                        mouseName, dayName, sessionSuffix, pathSession, pathTrialInd, pathTrialStruct, pathActivePassive
                    ])

    # Find necessary file paths in overlay folder
    def find_parse_overlay(self, pathOverlay):
        for mouseName, dfMouse in self.dfSession.groupby(['mousename']):
            pathMouse = os.path.join(pathOverlay, mouseName)
            pathRef = os.path.join(pathMouse, 'refImg_ROIs.mat')
            pathT2 = os.path.join(pathMouse, 't2str.mat')

            prepcommon._testdir(pathMouse)
            prepcommon._testfile(pathRef)
            prepcommon._testfile(pathT2)

            self.pathRef[mouseName] = pathRef
            self.pathT2[mouseName] = pathT2

    def parse_video_paths(self, path, day):
        # filesTmp = os.listdir(path)
        (_, _, filenames) = next(os.walk(path))
        filenames = [f for f in filenames if f[:4] == day[:4]]              # Test that 1st 4 letters coincide with the year
        filenames = [f for f in filenames if os.path.splitext(f)[1] == '']  # Test that file has no extension
        filepaths = [os.path.join(path, f) for f in filenames]

        return list(sorted(filepaths))

    # Apply transforms to reference files, overlay allen. Inspect by eye
    def test_transform_ref(self):
        prepcommon.test_transform_ref(self.pathT2, self.pathRef)

    # Apply transforms to time-averaged single trials
    # Average result over examples from different days
    #  1) T1 must make averages over days less blurry, as it aligns different days to same position
    def test_transform_vids(self, mousename):
        pathsMouse = self.dataPaths[self.dataPaths['mouse'] == mousename].reset_index()
        prepcommon.test_transform_vids(pathsMouse, self.pathT1[mousename], self.pathT2[mousename], self.allen,
                                       self.parse_video_paths)

    # Read video files, extract channel data and save to HDF5
    def process_video_files(self, mouseName, skipExisting=False):
        mouseRows = self.dataPaths[self.dataPaths['mouse'] == mouseName]

        # for mouseName, mouseRows in self.dataPaths.groupby(['mouse']):
        t2 = np.array(prepcommon.load_t2(self.pathT2[mouseName]))

        with h5py.File(mouseName + '.h5', 'a') as h5file:
            if 'data' not in h5file.keys():
                h5file.create_group('data')

        for idx, row in mouseRows.iterrows():
            sessionName = row['day'] + '_' + row['session']

            t1 = prepcommon.load_t1(self.pathT1[mouseName][row['day']])

            with h5py.File(mouseName + '.h5', 'a') as h5file:
                sessionProcessed = sessionName in h5file['data'].keys()

            if sessionProcessed and not skipExisting:
                print('>> Have', sessionName, '; skipping')
            else:
                print('processing', sessionName)
                filePaths = self.parse_video_paths(row['sessionPath'], row['day'])

                dataRSP = []
                for iVid, vidpath in enumerate(filePaths):
                    print('-', iVid, '/', len(filePaths))

                    try:
                        dataTrial = dcimg.DCIMGFile(vidpath)[:]

                        dataTrialTr = []
                        for img in dataTrial:
                            imgDS = skt.downscale_local_mean(img, (2, 2))
                            imgT1, imgT2 = prepcommon.transform_img(imgDS, t2, t1=t1)
                            dataTrialTr += [imgT2]

                        dataImgTr = np.array(dataTrialTr)
                        dataRSP += [prepcommon.extract_channel_data(dataImgTr)]

                    except:
                        print('---warning, reading video failed, filling with NAN')
                        dataRSP += [np.full(dataRSP[-1].shape, np.nan)]

                with h5py.File(mouseName + '.h5', 'a') as h5file:
                    h5file['data'].create_dataset(sessionName, data=np.array(dataRSP))

############################
#  Metadata Pooling
############################


    # Different mice have different Go/NoGo testures
    def tex_go_nogo_bymouse(self, mouseName):
        # return 'P100', 'P1200'
        if (mouseName == 'mou_5') or (mouseName == 'mou_7'):
            return 'P100', 'P1200'
        else:
            return 'P1200', 'P100'

    def trial_map_go_nogo_bymouse(self, mouseName):
        if (mouseName == 'mou_5') or (mouseName == 'mou_7'):
            return {'100' : 'Hit', '1200' : 'CR'}
        else:
            return {'1200' : 'Hit', '100' : 'CR'}

    # Convert stimulus and decision into trial type
    def parse_trial_type(self, stimulus, decision, mouseName):
        texGo, texNoGo = self.tex_go_nogo_bymouse(mouseName)

        if decision == 'Early':
            return decision
        else:
            assert (texGo in stimulus) or (texNoGo in stimulus)
            goTex = texGo in stimulus

            if decision == 'Go' and goTex:
                return 'Hit'
            elif decision == 'No Go' and not goTex:
                return 'CR'
            elif decision == 'No Response' and goTex:
                return 'Miss'
            elif decision == 'Inappropriate Response' and not goTex:
                return 'FA'
            else:
                raise ValueError('Unexpected combination', stimulus, decision)

    # Read trial structure file, drop unnecessary columns, compute trialType, return DF
    def read_trial_structure_as_pd(self, path, mouseName):
        df = pd.DataFrame(pymatreader.read_mat(path)['trials'])
        df['trialType'] = [self.parse_trial_type(s, d, mouseName) for s, d in zip(df['stimulus'], df['decision'])]
        df.drop(['id', 'no', 'puff', 'report', 'auto_reward', 'stimulus', 'decision'], inplace=True, axis=1)
        return df

    # Read all structure files, process, save to H5. Also compute performance and save to H5
    def process_metadata_files(self, pwd):
        for mouseName, mouseRows in self.dataPaths.groupby(['mouse']):
            h5name = os.path.join(pwd, mouseName + '.h5')
            prepcommon._h5_append_group(h5name, 'metadata')
            prepcommon._h5_append_group(h5name, 'accuracy')
            prepcommon._h5_append_group(h5name, 'dprime')

            for idx, row in mouseRows.iterrows():
                sessionName = row['day'] + '_' + row['session']

                with h5py.File(h5name, 'a') as h5f:
                    metaProcessed = sessionName in h5f['metadata'].keys()

                if metaProcessed:
                    print('already processed', mouseName, sessionName)
                else:
                    dfTrialStruct = self.read_trial_structure_as_pd(row['trialStructPath'], mouseName)
                    dfTrialStruct.to_hdf(h5name, '/metadata/' + sessionName)

                    # Calculate and store accuracy and dprime
                    ttDict = prepcommon.count_trial_types(dfTrialStruct)
                    acc = accuracy(ttDict['Hit'], ttDict['Miss'], ttDict['FA'], ttDict['CR'])
                    dp = d_prime(ttDict['Hit'], ttDict['Miss'], ttDict['FA'], ttDict['CR'])

                    with h5py.File(h5name, 'a') as h5f:
                        h5f['accuracy'].create_dataset(sessionName, data=acc)
                        h5f['dprime'].create_dataset(sessionName, data=dp)

    def get_append_delay_times(self, pathPreferences):
        # Calculate delay times from metadata
        rezLst = []
        for idx, row in self.dataPaths.iterrows():
            session = row['day'] + '_' + row['session']
            fpath = os.path.join(pathPreferences, row['mouse'] + '.h5')

            try:
                df = pd.read_hdf(fpath, '/metadata/' + session)
                dfHit = df[df['trialType'] == 'Hit']

                lags = dfHit['decision_time'] - dfHit['stimulus_time'] - 2000
                minLag = np.round(np.min(lags) / 1000, 2)

                rezLst += [[row['mouse'], row['day'], row['session'], minLag]]
            except:
                print('session', session, 'has no metadata')

        df = pd.DataFrame(rezLst, columns=['mousename', 'dateKey', 'sessionKey', 'delay'])

        # Read sessions file
        pwdSess = os.path.join(pathPreferences, 'sessions_tex.csv')
        dfSess = pd.read_csv(pwdSess, sep='\t')

        if 'delay' in dfSess.columns:
            del dfSess['delay']

        # Join the two
        # dfJoin = pd.concat([dfSess, df], ignore_index=True, sort=False)
        dfJoin = pd.merge(dfSess, df, how='left', on=['mousename', 'dateKey', 'sessionKey'])
        dfJoin.to_csv(
            os.path.join(pathPreferences, 'sessions_tex.csv'),
            sep='\t', index=False
        )

        print(list(dfJoin['delay']))

    def get_append_active_passive(self, pathPreferences):
        '''
        1. Loop over metadata, print, check 1 row per trial
        2. Find activePassive in paths for this session, test exists
        3. If exists, augment
        4. If not exists, set all to none manually
        '''
        for mouseName, dfMouse in self.dataPaths.groupby(['mouse']):
            fpath = os.path.join(pathPreferences, mouseName + '.h5')
            with h5py.File(fpath) as f:
                sessionsMeta = list(f['metadata'].keys())

            # Construct map from texture name to Hit/CR
            mapCanon = self.trial_map_go_nogo_bymouse(mouseName)

            for idx, row in dfMouse.iterrows():
                session = row['day'] + '_' + row['session']
                print(mouseName, session)

                if session not in sessionsMeta:
                    print('--Warning, skipping session with no metadata')
                    continue

                # Get metadata
                df = pd.read_hdf(fpath, '/metadata/' + session)

                # Get active/passive
                if os.path.isfile(row['pathActivePassive']):
                    df = prepcommon.parse_active_passive(df, row['pathActivePassive'], mapCanon)
                else:
                    print('--Warning, no active/passive, filling with None')
                    df['Activity'] = None

                df.to_hdf(fpath, '/metadata/' + session)

                # print(df)

    # For a given session, compute time of each timestep of each trial relative to start of session
    # Return as 2D array (nTrial, nTime)
    def get_pooled_data_rel_times(self, pwd, mouseName, session, FPS=20.0):
        fpath = os.path.join(pwd, mouseName + '.h5')
        with h5py.File(fpath, 'r') as f:
            dataRSP = np.copy(f['data'][session])

        '''
            1. Load session metadata
            2. Convert all times to timestamps
            3. From all timestamps, subtract first, convert to seconds
            4. Extract data, get nTimes from shape
            5. Set increment, return
        '''

        fpath = os.path.join(pwd, mouseName + '.h5')

        df = pd.read_hdf(fpath, '/metadata/' + session)
        timeStamps = pd.to_datetime(df['time_stamp'], format='%H:%M:%S.%f')
        timeDeltas = timeStamps - timeStamps[0]

        timesSh = np.arange(dataRSP.shape[1]) / FPS
        timesRS = np.array([t.total_seconds() + timesSh for t in timeDeltas])

        if timesRS.shape[0] < dataRSP.shape[0]:
            print('Warning: shape mismatch', timesRS.shape[0], dataRSP.shape[0])
            dataRSP = dataRSP[:timesRS.shape[0]]

        return timesRS, dataRSP

############################
#  Baseline Subtraction
############################

    # For each trial, compute DFF, store back to h5
    def baseline_subtraction_dff(self, pwd, iMin, iMax, overwrite=False):
        for mouseName, dfMouse in self.dataPaths.groupby(['mouse']):
            h5fname = os.path.join(pwd, mouseName + '.h5')

            prepcommon._h5_append_group(h5fname, 'bn_trial', overwrite=overwrite)

            for idx, row in dfMouse.iterrows():
                session = row['day'] + '_' + row['session']
                with h5py.File(h5fname, 'a') as h5f:
                    if session not in h5f['metadata'].keys():
                        print(mouseName, session, 'has no metadata, skipping')
                        continue

                    if session in h5f['bn_trial'].keys():
                        if overwrite:
                            del h5f['bn_trial'][session]
                        else:
                            print(mouseName, session, 'already exists, skipping')
                            continue

                print(mouseName, session)
                times, dataRSP = self.get_pooled_data_rel_times(pwd, mouseName, session)
                if len(times) != len(dataRSP):
                    print('-- trial mismatch', times.shape, dataRSP.shape)
                    # continue

                dataBN = np.zeros(dataRSP.shape)

                for iTr in range(dataRSP.shape[0]):
                    mu = np.nanmean(dataRSP[iTr, iMin:iMax], axis=0)
                    dataBN[iTr] = dataRSP[iTr] / mu - 1

                with h5py.File(h5fname, 'a') as h5f:
                    h5f['bn_trial'].create_dataset(session, data=dataBN)

    # For each session: fit poly, do poly-DFF, store back to h5
    def baseline_subtraction_poly(self, pwd, ord=2, alpha=0.01, overwrite=False):
        for mouseName, dfMouse in self.dataPaths.groupby(['mouse']):
            h5fname = os.path.join(pwd, mouseName + '.h5')

            prepcommon._h5_append_group(h5fname, 'bn_session', overwrite=overwrite)
            prepcommon._h5_append_group(h5fname, 'bn_fit', overwrite=overwrite)
            prepcommon._h5_append_group(h5fname, 'raw', overwrite=overwrite)

            for idx, row in dfMouse.iterrows():
                session = row['day'] + '_' + row['session']
                with h5py.File(h5fname, 'a') as h5f:
                    if session not in h5f['metadata'].keys():
                        print(mouseName, session, 'has no metadata, skipping')
                        continue

                    if session in h5f['bn_session'].keys():
                        if overwrite:
                            del h5f['bn_session'][session]
                            del h5f['bn_fit'][session]
                            del h5f['raw'][session]
                        else:
                            print(mouseName, session, 'already exists, skipping')
                            continue

                print(mouseName, session)
                times, dataRSP = self.get_pooled_data_rel_times(pwd, mouseName, session)
                if len(times) != len(dataRSP):
                    print('-- trial mismatch', times.shape, dataRSP.shape)
                    # continue

                dataRSPfit = prepcommon.polyfit_data_3D(times, dataRSP, ord, alpha)

                dataRaw = dataRSP - dataRSPfit
                dataBN = dataRaw / dataRSPfit

                with h5py.File(h5fname, 'a') as h5f:
                    h5f['raw'].create_dataset(session, data=dataRaw)
                    h5f['bn_fit'].create_dataset(session, data=dataRSPfit)
                    h5f['bn_session'].create_dataset(session, data=dataBN)

############################
#  Cleanup
############################

    def drop_preprocess_session(self, pwd, mouseName, session):
        pwd = os.path.join(pwd, mouseName + '.h5')

        with h5py.File(pwd, 'a') as f:
            for datatype in ['bn_trial', 'raw', 'bn_fit', 'bn_session']:
                if session in f[datatype]:
                    del f[datatype][session]
                    print('deleted', session, 'from', datatype)

            for metaclass in ['metadata', 'accuracy', 'dprime']:
                if session in f[metaclass]:
                    del f[metaclass][session]
                    print('deleted', session, 'from', metaclass)

    def crop_mismatch_trials(self, pwd):
        for mouseName, dfMouse in self.dataPaths.groupby(['mouse']):
            h5fname = os.path.join(pwd, mouseName + '.h5')
            for idx, row in dfMouse.iterrows():
                session = row['day'] + '_' + row['session']

                with h5py.File(h5fname, 'r') as h5f:
                    if session in h5f['metadata']:
                        # if mouseName == 'mou_9':
                        #     print(h5f['bn_session'].keys())
                        #     return

                        nTrialData = h5f['data'][session].shape[0]
                    else:
                        print('skipping non-existent session', session)
                        continue

                df = pd.read_hdf(h5fname, '/metadata/' + session)

                nTrialMeta = len(df)

                if nTrialData != nTrialMeta:
                    if nTrialData < nTrialMeta:
                        print(mouseName, session, nTrialData, nTrialMeta, 'driopping useless metadata')
                        nDiff = nTrialMeta - nTrialData
                        df.drop(df.tail(nDiff).index, inplace=True)  # drop last n rows
                        df.to_hdf(h5fname, '/metadata/' + session)
                    else:
                        print(mouseName, session, nTrialData, nTrialMeta, 'dropping extra unassociated data')
                        nCrop = nTrialData - nTrialMeta

                        with h5py.File(h5fname, 'a') as h5f:
                            data = np.copy(h5f['data'][session])
                            del h5f['data'][session]
                            h5f['data'][session] = data[:-nCrop]

############################
#  Sanity Checking
############################

    def check_reward_in_data(self, pwd):
        for mouseName, dfMouse in self.dataPaths.groupby(['mouse']):
            h5fname = os.path.join(pwd, mouseName + '.h5')

            for idx, row in dfMouse.iterrows():
                session = row['day'] + '_' + row['session']
                with h5py.File(h5fname, 'a') as h5f:
                    if session not in h5f['metadata'].keys():
                        print(mouseName, session, 'has no metadata, skipping')
                        continue

                    dataRAW = np.copy(h5f['data'][session])

                delay = pd_is_one_row(pd_query(self.dfSession, {'mousename' : mouseName, 'dateKey' : row['day'], 'sessionKey' : row['session']}))[1]['delay']

                nTimestepVid = dataRAW.shape[1]
                rewStartIdx = int((5 + delay) * 20)
                overlap = max(0, nTimestepVid - rewStartIdx)

                print(mouseName, session, delay, nTimestepVid, rewStartIdx, overlap)
