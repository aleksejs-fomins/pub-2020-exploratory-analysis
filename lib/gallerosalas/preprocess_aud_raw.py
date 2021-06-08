import json
import h5py
import dcimg
import os, sys
import numpy as np
import pandas as pd
import pymatreader
from datetime import datetime
from collections import defaultdict
import skimage.transform as skt

from mesostat.utils.pandas_helper import pd_append_row
from mesostat.stat.performance import accuracy, d_prime

import lib.gallerosalas.preprocess_common as prepcommon


class preprocess:
    def __init__(self, pathDict):
        # Parse L_modified file
        self.pathLmod = os.path.join(pathDict['Preferences'], 'L_modified.mat')
        prepcommon._testfile(self.pathLmod)
        self.allen = pymatreader.read_mat(self.pathLmod)['L']

        # Parse sessions file
        pathSessions = os.path.join(pathDict['Preferences'], 'sessions_aud.csv')
        self.dfSession = pd.read_csv(pathSessions, sep='\t')

        # Find parse TGT files
        self.dataPaths = pd.DataFrame(columns=['mouse', 'day', 'session', 'sessionPath', #'trialIndPath',
                                               'trialStructPath', 'pathActivePassive'])
        self.pathT1 = defaultdict(dict)
        self.find_parse_data_paths(pathDict['TGT'])

        # Find parse Overlay
        # self.pathRef = {}
        self.pathT2 = {}
        self.find_parse_overlay(pathDict['Overlay'])

    # Find necessary file paths in the TDT folder
    def find_parse_data_paths(self, pathTDT):
        for mouseName, dfMouse in self.dfSession.groupby(['mousename']):
            pathMouse = os.path.join(pathTDT, mouseName)
            prepcommon._testdir(pathMouse)

            for dayName, dfDay in dfMouse.groupby(['dateKey']):
                print(mouseName, dayName)
                pathDay = os.path.join(pathMouse, dayName, 'widefield_labview')
                pathAllen = os.path.join(pathDay, 'ROI_Allen', 'registration_transform_1510.mat')
                haveDir = prepcommon._testdir(pathDay, critical=False)
                haveFile = prepcommon._testfile(pathAllen, critical=False)

                if haveDir and haveFile:
                    self.pathT1[mouseName][dayName] = pathAllen
                    for idx, row in dfDay.iterrows():
                        sessionSuffix = row['sessionKey']

                        pathSession = os.path.join(pathDay, sessionSuffix)
                        pathMat = os.path.join(pathSession, 'Matt_files')
                        # pathTrialInd = os.path.join(pathMat, 'trials_ind.mat')

                        matFiles = [f for f in os.listdir(pathMat) if os.path.splitext(f)[1] == '.mat' and f[:9] == 'Behavior_']
                        assert len(matFiles) == 1
                        pathTrialStruct = os.path.join(pathMat, matFiles[0])

                        pathActivePassive = os.path.join(pathMat, 'trials_with_and_wo_initial_moves_OCIA_from_movie.mat')

                        prepcommon._testdir(pathSession)
                        prepcommon._testdir(pathMat)
                        # prepcommon._testfile(pathTrialInd)
                        prepcommon._testfile(pathTrialStruct)
                        prepcommon._testfile(pathActivePassive)

                        self.dataPaths = pd_append_row(self.dataPaths, [
                            mouseName, dayName, sessionSuffix, pathSession, #pathTrialInd,
                            pathTrialStruct,
                            pathActivePassive
                        ])

    # Find necessary file paths in overlay folder
    def find_parse_overlay(self, pathOverlay):
        for mouseName, dfMouse in self.dfSession.groupby(['mousename']):
            pathMouse = os.path.join(pathOverlay, mouseName)
            # pathRef = os.path.join(pathMouse, 'refImg_ROIs.mat')
            pathT2 = os.path.join(pathMouse, 't2str.mat')

            prepcommon._testdir(pathMouse)
            # self._testfile(pathRef)
            prepcommon._testfile(pathT2)

            # self.pathRef[mouseName] = pathRef
            self.pathT2[mouseName] = pathT2

    def parse_video_paths(self, path, day):
        # filesTmp = os.listdir(path)
        (_, _, filenames) = next(os.walk(path))
        filenames = [f for f in filenames if f[:4] == day[:4]]              # Test that 1st 4 letters coincide with the year
        filenames = [f for f in filenames if os.path.splitext(f)[1] == '']  # Test that file has no extension
        filepaths = [os.path.join(path, f) for f in filenames]

        return list(sorted(filepaths))

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
                        dataRSP += [prepcommon.extract_channel_data(dataImgTr, self.allen)]

                    except:
                        print('---warning, reading video failed, filling with NAN')
                        dataRSP += [np.full(dataRSP[-1].shape, np.nan)]

                with h5py.File(mouseName + '.h5', 'a') as h5file:
                    h5file['data'].create_dataset(sessionName, data=np.array(dataRSP))


    def parse_trial_structure(self, pwd):
        trialStruct = pymatreader.read_mat(pwd)
        trialTypes = trialStruct['out']['respTypes']
        return pd.DataFrame({'trialType' : trialTypes}).replace({
            1: "Hit", 2: "CR", 3: "FA", 4: "Miss", 5: "Early"})


    def read_trial_structure_as_pd(self, trialStructPath, activePassivePath):
        dfTrialStruct = self.parse_trial_structure(trialStructPath)
        dfTrialStruct['Activity'] = None

        mapHit = lambda tt: 'Hit' if tt == 'hit' else tt

        activePassiveStruct = pymatreader.read_mat(activePassivePath)
        for tt in ['hit', 'CR']:
            rezDict = {}

            for activity in ['delay_move', 'no_prior_move', 'noisy', 'prior_move', 'quiet_sens', 'quiet_then_move']:
                keyAct = 'tr_' + tt + '_' + activity
                if keyAct in activePassiveStruct:
                    keys = activePassiveStruct[keyAct]
                    if isinstance(keys, int):
                        keys = [keys]

                    vals = [activity] * len(keys)
                    rezDict = {**rezDict, **dict(zip(keys, vals))}

            print('--', tt, (dfTrialStruct['trialType'] == mapHit(tt)).sum(), len(rezDict))

            iTT = 0
            for idx, row in dfTrialStruct.iterrows():
                if row['trialType'] == mapHit(tt):
                    if iTT + 1 in rezDict:
                        dfTrialStruct['Activity'][idx] = rezDict[iTT + 1]
                    iTT += 1

        return dfTrialStruct

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
                    dfTrialStruct = self.read_trial_structure_as_pd(row['trialStructPath'], row['pathActivePassive'])

                    # If there are any NANs in the trial type
                    # Test that the NAN block is contiguous and is at the end of the dataset
                    nanRows = dfTrialStruct['trialType'].isnull()
                    nanIdxs = list(dfTrialStruct[nanRows].index)
                    if len(nanIdxs) != 0:
                        minIdx = np.min(nanIdxs)
                        maxIdx = np.max(nanIdxs)
                        assert maxIdx == len(dfTrialStruct) - 1
                        assert nanIdxs == list(np.arange(minIdx, maxIdx+1))
                        print("-- passed test", len(dfTrialStruct), len(nanIdxs))

                    dfTrialStruct = dfTrialStruct[~nanRows]
                    dfTrialStruct.to_hdf(h5name, '/metadata/' + sessionName)

                    # Calculate and store accuracy and dprime
                    ttDict = prepcommon.count_trial_types(dfTrialStruct)
                    acc = accuracy(ttDict['Hit'], ttDict['Miss'], ttDict['FA'], ttDict['CR'])
                    dp = d_prime(ttDict['Hit'], ttDict['Miss'], ttDict['FA'], ttDict['CR'])
                    # print(ttDict, acc, dp)

                    with h5py.File(h5name, 'a') as h5f:
                        h5f['accuracy'].create_dataset(sessionName, data=acc)
                        h5f['dprime'].create_dataset(sessionName, data=dp)


    def extract_timestamps_video(self, pwd):
        for mouseName, dfMouse in self.dataPaths.groupby(['mouse']):
            h5name = os.path.join(pwd, mouseName + '.h5')
            for idx, row in dfMouse.iterrows():
                sessionName = row['day'] + '_' + row['session']

                print(mouseName, sessionName)
                filePaths = self.parse_video_paths(row['sessionPath'], row['day'])
                fileBases = [os.path.basename(f) for f in filePaths]
                timeStampStrings = [
                    ':'.join([f[:4], f[4:6], f[6:8], f[9:11], f[11:13], f[13:15]])
                    for f in fileBases
                ]

                timeStamps = [datetime.strptime(f, "%Y:%d:%m:%H:%M:%S") for f in timeStampStrings]

                metadata = pd.read_hdf(h5name, '/metadata/'+sessionName)

                if len(timeStamps) != len(metadata):
                    print(mouseName, sessionName, len(timeStamps), len(metadata))

                if len(timeStamps) < len(metadata):
                    timeStamps += [None] * (len(metadata) - len(timeStamps))
                elif len(timeStamps) == len(metadata) + 1:
                    metadata = pd_append_row(metadata, [None] * len(metadata.columns))
                    print(metadata.tail())

                metadata['timeStamps'] = timeStamps
                metadata.to_hdf(h5name, '/metadata/' + sessionName)