import h5py
import dcimg
import os, sys
import numpy as np
import pandas as pd
from collections import defaultdict
import pymatreader
# from scipy.ndimage import affine_transform
import skimage.transform as skt
import matplotlib.pyplot as plt


from mesostat.utils.pandas_helper import pd_append_row, pd_query, pd_is_one_row
from mesostat.stat.performance import accuracy, d_prime
from mesostat.utils.labview_helper import labview_parse_log_as_pandas

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
            columns=['mouse', 'day', 'session', 'sessionPath',
                     'trialIndPath', 'trialStructPathMat', 'trialStructPathLabview',
                     'pathActivePassive', 'pathMovementVectors'])
        self.pathT1 = defaultdict(dict)

        if 'TDT' in pathDict:
            self.find_parse_tdt(pathDict['TDT'])

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

    def _letters_to_sessions(self, name):
        return ord(name) - 96

    def get_mice(self):
        return sorted(list(set(self.dataPaths['mouse'])))

    # Find necessary file paths in the TDT folder
    def find_parse_tdt(self, pathTDT):
        print('Parsing TDT:')

        for mouseName, dfMouse in self.dfSession.groupby(['mousename']):
            pathMouse = os.path.join(pathTDT, mouseName)
            prepcommon._testdir(pathMouse)

            for dayName, dfDay in dfMouse.groupby(['dateKey']):
                print('--', mouseName, dayName)

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
                        pathTrialStructMat = os.path.join(pathMat, matFiles[0])
                    else:
                        pathTrialStructMat = os.path.join(pathMat, dayName + sessionSuffix + '.mat')


                    if mouseName == 'mou_5':
                        sessionKeyLabview = '_s' + sessionSuffix[-1]
                    elif mouseName == 'mou_6':
                        sessionKeyLabview = '_s' + str(self._letters_to_sessions(sessionSuffix))
                    else:
                        sessionKeyLabview = '_' + sessionSuffix

                    pathTrialStructLabview = os.path.join(pathSession, mouseName + sessionKeyLabview + ''.join(dayName.split('_')) + '.txt')

                    pathActivePassive = os.path.join(pathMat, 'trials_with_and_wo_initial_moves_OCIA_from_movie.mat')

                    pathMovementVectors = os.path.join(pathMat, 'move_vectors_from_movie.mat')

                    prepcommon._testdir(pathSession)
                    prepcommon._testdir(pathMat)
                    prepcommon._testfile(pathTrialInd)
                    prepcommon._testfile(pathTrialStructMat)
                    prepcommon._testfile(pathTrialStructLabview, critical=False)
                    prepcommon._testfile(pathActivePassive, critical=False)
                    prepcommon._testfile(pathMovementVectors, critical=False)

                    self.dataPaths = pd_append_row(self.dataPaths, [
                        mouseName, dayName, sessionSuffix, pathSession,
                        pathTrialInd, pathTrialStructMat, pathTrialStructLabview,
                        pathActivePassive, pathMovementVectors
                    ])

    # Find necessary file paths in overlay folder
    def find_parse_overlay(self, pathOverlay):
        print('Parsing overlay:')

        for mouseName, dfMouse in self.dfSession.groupby(['mousename']):
            print('--', mouseName)

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

    def mouse_is_flipped_hit_cr(self, mouseName):
        return (mouseName == 'mou_6') or (mouseName == 'mou_9')

    # Different mice have different Go/NoGo testures
    def tex_go_nogo_bymouse(self, mouseName):
        if self.mouse_is_flipped_hit_cr(mouseName):
            return 'P1200', 'P100'
        else:
            return 'P100', 'P1200'

    def trial_map_go_nogo_bymouse(self, mouseName):
        if self.mouse_is_flipped_hit_cr(mouseName):
            return {'1200': 'Hit', '100': 'CR'}
        else:
            return {'100': 'Hit', '1200': 'CR'}

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
        _drop_non_floats = lambda lst: [el if isinstance(el, float) else np.nan for el in lst]

        df = pd.DataFrame(pymatreader.read_mat(path)['trials'])
        df['trialType'] = [self.parse_trial_type(s, d, mouseName) for s, d in zip(df['stimulus'], df['decision'])]

        df['decision_time'] = _drop_non_floats(df['decision_time'])
        df['stimulus_time'] = _drop_non_floats(df['stimulus_time'])
        df['delayLength'] = (df['decision_time'] - df['stimulus_time'] - 2000) / 1000

        df.drop(['id', 'no', 'puff', 'report', 'auto_reward', 'stimulus', 'decision'], inplace=True, axis=1)
        return df

    def read_parse_labview_trial_structure_as_pd(self, pwd):
        timestamp, dfMeta = labview_parse_log_as_pandas(pwd, sep='\t', endEvent='End Trial')

        # Compute delay lengths
        dfMeta['DelayLength'] = dfMeta['Report'] - dfMeta['Delay']

        # Compute trial types
        dfMeta['TrialType'] = 'NotSure'
        srcTrialTypes = ['Go', 'No Go', 'No Response', 'Inappropriate Response']
        trgTrialTypes = ['Hit', 'CR', 'Miss', 'FA']

        for srcTT, trgTT in zip(srcTrialTypes, trgTrialTypes):
            if srcTT in dfMeta.columns:
                dfMeta.loc[dfMeta[srcTT].notna(), 'TrialType'] = trgTT

        return timestamp, dfMeta

    # Take the best of two arrays
    def reconcile_trial_structure(self, dfMat, dfLabview):
        # Compare number of trials
        assert len(dfMat) == len(dfLabview)

        # Compare trial idxs for hit, miss, CR, FA
        for trialType in ['Hit', 'CR', 'Miss', 'FA']:
            idxsTrMat = np.array(dfMat['trialType'] == trialType)
            idxsTrLV = np.array(dfLabview['TrialType'] == trialType)
            np.testing.assert_array_equal(idxsTrMat, idxsTrLV)

        # Delay is wrong in MAT, use LV instead
        delay = np.round(np.nanmean(dfLabview['DelayLength']),2)

        # Trial labels are better in MAT, reuse
        dfLabview['TrialType'] = list(dfMat['trialType'])

        # print('Mat',
        #       np.round(np.nanmin(dfMat['delayLength']),2),
        #       np.round(np.nanmax(dfMat['delayLength']),2),
        #       np.round(np.nanmean(dfMat['delayLength']),2))
        # print('LV',
        #       np.round(np.nanmin(dfLabview['DelayLength']),2),
        #       np.round(np.nanmax(dfLabview['DelayLength']),2),
        #       np.round(np.nanmean(dfLabview['DelayLength']),2))

        return delay, dfLabview

    # Read all structure files, process, save to H5. Also compute performance and save to H5
    def process_metadata_files(self, pwd):
        for mouseName, mouseRows in self.dataPaths.groupby(['mouse']):
            h5name = os.path.join(pwd, mouseName + '.h5')
            prepcommon._h5_append_group(h5name, 'metadataTrial')
            prepcommon._h5_append_group(h5name, 'accuracy')
            prepcommon._h5_append_group(h5name, 'dprime')

            timeStampLst = []
            delayLst = []

            for idx, row in mouseRows.iterrows():
                sessionName = row['day'] + '_' + row['session']

                with h5py.File(h5name, 'a') as h5f:
                    metaProcessed = sessionName in h5f['metadataTrial'].keys()

                print(mouseName, sessionName)
                if metaProcessed:
                    print('-- already processed')
                    continue

                # Read metadata from two file types, reconcile
                dfTrialStructMat = self.read_trial_structure_as_pd(row['trialStructPathMat'], mouseName)
                timeStamp, dfTrialStructLabview = self.read_parse_labview_trial_structure_as_pd(row['trialStructPathLabview'])
                delay, dfTrialStruct = self.reconcile_trial_structure(dfTrialStructMat, dfTrialStructLabview)

                # Store to file
                timeStampLst += [timeStamp]
                delayLst += [delay]
                dfTrialStruct.to_hdf(h5name, '/metadataTrial/' + sessionName)

                # Calculate and store accuracy and dprime
                ttDict = prepcommon.count_trial_types(dfTrialStruct)

                print(ttDict)

                acc = accuracy(ttDict['Hit'], ttDict['Miss'], ttDict['FA'], ttDict['CR'])
                dp = d_prime(ttDict['Hit'], ttDict['Miss'], ttDict['FA'], ttDict['CR'])

                with h5py.File(h5name, 'a') as h5f:
                    h5f['accuracy'].create_dataset(sessionName, data=acc)
                    h5f['dprime'].create_dataset(sessionName, data=dp)

            pd.DataFrame({'timestamp': timeStampLst, 'delay': delayLst}).to_hdf(h5name, 'metadataSession')

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
                sessionsMeta = list(f['metadataTrial'].keys())

            # Construct map from texture name to Hit/CR
            mapCanon = self.trial_map_go_nogo_bymouse(mouseName)

            for idx, row in dfMouse.iterrows():
                session = row['day'] + '_' + row['session']
                print(mouseName, session)

                if session not in sessionsMeta:
                    print('--Warning, skipping session with no metadata')
                    continue

                # Get metadata
                df = pd.read_hdf(fpath, '/metadataTrial/' + session)

                # Get active/passive
                if os.path.isfile(row['pathActivePassive']):
                    df = prepcommon.parse_active_passive(df, row['pathActivePassive'], mapCanon)
                else:
                    print('--Warning, no active/passive, filling with None')
                    df['Activity'] = None

                df.to_hdf(fpath, '/metadataTrial/' + session)

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

        # fpath = os.path.join(pwd, mouseName + '.h5')
        #
        # df = pd.read_hdf(fpath, '/metadataTrial/' + session)
        # timeStamps = pd.to_datetime(df['time_stamp'], format='%H:%M:%S.%f')
        # timeDeltas = timeStamps - timeStamps[0]
        #
        # timesSh = np.arange(dataRSP.shape[1]) / FPS
        # timesRS = np.array([t.total_seconds() + timesSh for t in timeDeltas])

        df = pd.read_hdf(fpath, '/metadataTrial/' + session)
        timesSh = np.arange(dataRSP.shape[1]) / FPS
        timesRS = np.array([timesSh + t for t in df['Time']])

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
                    if session not in h5f['metadataTrial'].keys():
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
                    if session not in h5f['metadataTrial'].keys():
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

            for metaclass in ['metadataTrial', 'metadataSession', 'accuracy', 'dprime']:
                if session in f[metaclass]:
                    del f[metaclass][session]
                    print('deleted', session, 'from', metaclass)

    def crop_mismatch_trials(self, pwd):
        for mouseName, dfMouse in self.dataPaths.groupby(['mouse']):
            h5fname = os.path.join(pwd, mouseName + '.h5')
            for idx, row in dfMouse.iterrows():
                session = row['day'] + '_' + row['session']

                with h5py.File(h5fname, 'r') as h5f:
                    if session in h5f['metadataTrial']:
                        # if mouseName == 'mou_9':
                        #     print(h5f['bn_session'].keys())
                        #     return

                        nTrialData = h5f['data'][session].shape[0]
                    else:
                        print('skipping non-existent session', session)
                        continue

                df = pd.read_hdf(h5fname, '/metadataTrial/' + session)

                nTrialMeta = len(df)

                if nTrialData != nTrialMeta:
                    if nTrialData < nTrialMeta:
                        print(mouseName, session, nTrialData, nTrialMeta, 'dropping excess metadata')
                        nDiff = nTrialMeta - nTrialData
                        df.drop(df.tail(nDiff).index, inplace=True)  # drop last n rows
                        df.to_hdf(h5fname, '/metadataTrial/' + session)
                    else:
                        print(mouseName, session, nTrialData, nTrialMeta, 'dropping excess data')
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
                    if session not in h5f['metadataTrial'].keys():
                        print(mouseName, session, 'has no metadata, skipping')
                        continue

                    dataRAW = np.copy(h5f['data'][session])

                delay = pd_is_one_row(pd_query(self.dfSession, {'mousename' : mouseName, 'dateKey' : row['day'], 'sessionKey' : row['session']}))[1]['delay']

                nTimestepVid = dataRAW.shape[1]
                rewStartIdx = int((5 + delay) * 20)
                overlap = max(0, nTimestepVid - rewStartIdx)

                print(mouseName, session, delay, nTimestepVid, rewStartIdx, overlap)

############################
#  Behaviour
############################

    def behaviour_tune_resample_kernel(self, mousename, session, sig2,
                                       trialType='Hit', trialIdx=0, srcFreq=30.0, trgFreq=20.0):
        dayKey = '_'.join(session.split('_')[:3])
        sessionKey = session.split('_')[3]

        idx, row = pd_is_one_row(pd_query(self.dataPaths, {'mouse': mousename, 'day': dayKey, 'session':sessionKey}))

        prepcommon.behaviour_tune_resample_kernel(row['pathMovementVectors'], sig2,
                                                  trialType=trialType, trialIdx=trialIdx, srcFreq=srcFreq, trgFreq=trgFreq)

    def read_parse_behaviour(self, pwd, skipExisting=False, srcFreq=30.0, trgFreq=20.0):
        srcNames = np.array(['100', '1200', 'FA', 'MISS', 'early'])
        trgNames = np.array(['Hit', 'CR', 'FA', 'Miss', 'Early'])
        sig2 = (1.0 / trgFreq)**2 / 16  # Choose a sharper kernel than normal because of binary nature of data

        # Read behaviour file
        for mousename, dfMouse in self.dataPaths.groupby('mouse'):
            h5Path = pwd + '/' + mousename + '.h5'

            if self.mouse_is_flipped_hit_cr(mousename):
                trialNameMap = {'moveVect_' + s: t for s, t in zip(srcNames, trgNames[[1, 0, 2, 3, 4]])}
            else:
                trialNameMap = {'moveVect_' + s: t for s, t in zip(srcNames, trgNames)}

            for idx, row in dfMouse.iterrows():
                pwdMove = row['pathMovementVectors']
                if not os.path.isfile(pwdMove):
                    print('Warning:', mousename, row['day'], row['session'], 'has no movement, skipping')
                else:
                    # Get timesteps from h5 file
                    session = row['day'] + '_' + row['session']
                    with h5py.File(h5Path, 'r') as f:
                        if session not in f['metadataTrial'].keys():
                            print(session, 'dropped from final dataset, ignore')
                            continue

                        if not skipExisting and ('movementVectors' in f.keys()) and (session in f['movementVectors'].keys()):
                            print(session, 'already calculated, skipping')
                            continue

                        nTrialData, nSampleData, _ = f['raw'][session].shape

                    print('doing', mousename, session)

                    # Get trialTypeNames from h5 file
                    dfMeta = pd.read_hdf(h5Path, 'metadataTrial/' + session)
                    trialTypeNames = np.array(dfMeta['TrialType'])
                    assert nTrialData == len(trialTypeNames)

                    # Read and resample movement data from mat file
                    movementRS = prepcommon.read_resample_movement_data(pwdMove, trialTypeNames, nTrialData,
                                                                        nSampleData, trialNameMap, sig2,
                                                                        srcFreq=srcFreq, trgFreq=trgFreq)

                    with h5py.File(h5Path, 'a') as f:
                        # Create movement group if not created
                        if 'movementVectors' not in f.keys():
                            f.create_group('movementVectors')

                        # Overwrite dataset if it is already there and was not skipped
                        if session in f['movementVectors'].keys():
                            del f['movementVectors'][session]

                        # Write to h5
                        f['movementVectors'][session] = movementRS
