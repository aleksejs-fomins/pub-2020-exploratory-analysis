import json
import h5py
import dcimg
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pymatreader
# from scipy.ndimage import affine_transform
import skimage.transform as skt

from mesostat.utils.arrays import numpy_merge_dimensions
from mesostat.utils.pandas_helper import pd_append_row
from mesostat.utils.matlab_helper import loadmat
from mesostat.stat.performance import accuracy, d_prime
from mesostat.utils.signals.fit import natural_cubic_spline_fit_reg

import lib.preprocessing.polyfit as polyfit

# import mesostat.utils.image_processing as msimg



class preprocess:
    def __init__(self, pathDict):
        # Parse L_modified file
        self.pathLmod = os.path.join(pathDict['Preferences'], 'L_modified.mat')
        self._testfile(self.pathLmod)
        self.allen = pymatreader.read_mat(self.pathLmod)['L']

        # Parse sessions file
        pathSessions = os.path.join(pathDict['Preferences'], 'sessions.json')
        with open(pathSessions, 'r') as json_file:
            self.sessions = json.load(json_file)

        # Find parse TGT files
        self.dataPaths = pd.DataFrame(
            columns=['mouse', 'day', 'session', 'sessionPath', 'trialIndPath', 'trialStructPath'])
        self.pathT1 = defaultdict(dict)
        self.find_parse_tdt(pathDict['TGT'])

        # Find parse Overlay
        self.pathRef = {}
        self.pathT2 = {}
        self.find_parse_overlay(pathDict['Overlay'])

    # Test if folder is found
    def _testdir(self, path):
        if not os.path.isdir(path):
            raise ValueError('Not found path:', path)

    # Test if file is found
    def _testfile(self, path):
        if not os.path.isfile(path):
            print('WARNING: Not found file:', path)

    # Convert session name e.g. 'Session01' to letters like 'a', 'b', 'c'
    def _sessions_to_letters(self, name):
        sessionIdx = int(name[len('session'):]) - 1
        return chr(97 + sessionIdx)

    def _h5_append_group(self, h5path, group):
        with h5py.File(h5path, 'a') as h5file:
            if group not in h5file.keys():
                h5file.create_group(group)

    # Find necessary file paths in the TDT folder
    def find_parse_tdt(self, pathTDT):
        for mouseName, mouseSessionDict in self.sessions.items():
            pathMouse = os.path.join(pathTDT, mouseName)
            self._testdir(pathMouse)

            for dayName, sessionSuffixLst in mouseSessionDict.items():
                pathDay = os.path.join(pathMouse, 'TDT', dayName, 'widefield_labview')
                pathAllen = os.path.join(pathDay, 'ROI_Allen', 'registration_transform_1510.mat')
                self._testdir(pathDay)
                self._testfile(pathAllen)

                self.pathT1[mouseName][dayName] = pathAllen

                for sessionSuffix in sessionSuffixLst:
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

                    self._testdir(pathSession)
                    self._testdir(pathMat)
                    self._testfile(pathTrialInd)
                    self._testfile(pathTrialStruct)

                    self.dataPaths = pd_append_row(self.dataPaths, [
                        mouseName, dayName, sessionSuffix, pathSession, pathTrialInd, pathTrialStruct
                    ])

    # Find necessary file paths in overlay folder
    def find_parse_overlay(self, pathOverlay):
        for mouseName, mouseSessionDict in self.sessions.items():
            pathMouse = os.path.join(pathOverlay, mouseName)
            pathRef = os.path.join(pathMouse, 'refImg_ROIs.mat')
            pathT2 = os.path.join(pathMouse, 't2str.mat')

            self._testdir(pathMouse)
            self._testfile(pathRef)
            self._testfile(pathT2)

            self.pathRef[mouseName] = pathRef
            self.pathT2[mouseName] = pathT2

    def parse_video_paths(self, path, day):
        # filesTmp = os.listdir(path)
        (_, _, filenames) = next(os.walk(path))
        filenames = [f for f in filenames if f[:4] == day[:4]]              # Test that 1st 4 letters coincide with the year
        filenames = [f for f in filenames if os.path.splitext(f)[1] == '']  # Test that file has no extension
        filepaths = [os.path.join(path, f) for f in filenames]

        return list(sorted(filepaths))

    # Load T1: Affine Transformation from file
    def load_t1(self, path):
        d = pymatreader.read_mat(path)['tform']['tdata']
        t = d['T']
        tinv = d['Tinv']
        assert np.linalg.norm(t.dot(tinv) - np.eye(3)) < 1.0E-10
        return t

    # Load T2: Polynomial Transformation from file
    def load_t2(self, path):
        d = pymatreader.read_mat(path)['tform_str']
        assert d['Degree'] == 3
        assert d['Dimensionality'] == 2

        # Canonical Monads:  1, x, y, x2, xy, y2, x3, x2y, xy2, x3
        # Matlab Monads:     1, x, y, xy, x2, y2, x2y, xy2, x3, y3

        monadOrder = [0,1,2,4,3,5,8,6,7,9]
        return d['A'][monadOrder], d['B'][monadOrder]

    # Load reference image from path
    def load_ref_img(self, path):
        return loadmat(path)['refImg']

    # Apply T1 and T2 to a flat 2D image. Return results of each transform
    def transform_img(self, img, t2, t1=None):
        cval = 100 * np.nanmax(img)
        print(cval)

        rez = [img]
        if t1 is not None:
            at = skt.AffineTransform(t1.T)
            rez += [skt.warp(img, at.inverse, mode='constant', cval=cval)]

        A, B = t2
        pt = skt.PolynomialTransform(np.array([A, B]))
        polyRez = skt.warp(rez[-1], pt, mode='constant', cval=cval)
        polyRez = polyRez[::-1].T
        polyRez[polyRez > cval / 10] = np.nan
        rez += [polyRez]

        if t1 is not None:
            rez[-2][rez[-2] > cval / 10] = np.nan

        #     if t1 is not None:
        #         at = skt.AffineTransform(t1.T)
        #         trans = [lambda x: msimg.dest_est(x, at)]
        #         matCSRT1 = msimg.mapping_list_to_matrix(img.shape, trans)
        #         rez += [msimg.map_with_matrix(img, matCSRT1)]

        #     A, B = t2
        # #     pt = skt.PolynomialTransform(np.array([A, B]))
        # #     transLst += [lambda x: msimg.dest_est(x, pt)]
        #     trans = [lambda x: msimg.poly_transform(x, [A, B])]
        #     matCSRT2 = msimg.mapping_list_to_matrix(img.shape, trans)
        #     rez += [msimg.map_with_matrix(rez[-1], matCSRT2)]

        return rez[1:]

    # Plot the results of transforms
    def plot_transforms(self, img, imgT2, imgT1=None):
        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        ax[0].imshow(img)
        ax[0].set_title('Raw')

        if imgT1 is not None:
            ax[1].imshow(imgT1)
            ax[1].set_title('T1')

        # Overlay Allen
        imgT2 = imgT2 / np.nanmax(imgT2) + 0.3 * (self.allen == 0).astype(int)

        ax[2].imshow(imgT2)
        ax[2].set_title('T2')
        plt.show()

    # Apply transforms to reference files, overlay allen. Inspect by eye
    def test_transform_ref(self):
        for mousename, t2path in self.pathT2.items():
            t2data = self.load_t2(t2path)
            refImg = self.load_ref_img(self.pathRef[mousename])
            transImg = self.transform_img(refImg, t2data)

            fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
            fig.set_title(mousename)
            ax[0].imshow(refImg)
            ax[1].imshow(transImg)
            plt.show()

    # Apply transforms to time-averaged single trials
    # Average result over examples from different days
    #  1) T1 must make averages over days less blurry, as it aligns different days to same position
    def test_transform_vids(self, mousename):
        pathsMouse = self.dataPaths[self.dataPaths['mouse'] == mousename].reset_index()

        imgLst = []
        imgT1Lst = []
        imgT2Lst = []
        for day in set(pathsMouse['day']):
            print('--', day)
            pathsDay = pathsMouse[pathsMouse['day'] == day].reset_index()
            sessionPath = pathsDay['sessionPath'][0]

            filePaths = self.parse_video_paths(sessionPath, day)
            data = dcimg.DCIMGFile(filePaths[0])[:]

            A, B = self.load_t2(self.pathT2[mousename])

            t1 = self.load_t1(self.pathT1[mousename][day])

            imgMean = np.mean(data, axis=0)
            imgDS = skt.downscale_local_mean(imgMean, (2, 2))
            imgT1, imgT2 = self.transform_img(imgDS, np.array([A, B]), t1=t1)

            imgLst += [imgDS]
            imgT1Lst += [imgT1]
            imgT2Lst += [imgT2]

        imgLst = np.mean(imgLst, axis=0)
        imgT1Lst = np.mean(imgT1Lst, axis=0)
        imgT2Lst = np.mean(imgT2Lst, axis=0)
        self.plot_transforms(imgLst, imgT2Lst, imgT1Lst)

    # Convert transformed video -> (nTime, nChannel)
    def extract_channel_data(self, vid):
        keys = sorted(set(self.allen.flatten()))[2:]
        assert len(keys) == 27

        rez = np.zeros((vid.shape[0], 27))
        for ikey, key in enumerate(keys):
            idxs = self.allen == key
            # print(vid.shape)
            vidPix = vid[:, idxs]
            # print(vidPix.shape)
            rez[:, ikey] = np.nanmean(vidPix, axis=1)

        return rez

    # Read video files, extract channel data and save to HDF5
    def process_video_files(self, mouseName, skipExisting=False):
        mouseRows = self.dataPaths[self.dataPaths['mouse'] == mouseName]

        # for mouseName, mouseRows in self.dataPaths.groupby(['mouse']):
        t2 = np.array(self.load_t2(self.pathT2[mouseName]))

        with h5py.File(mouseName + '.h5', 'a') as h5file:
            if 'data' not in h5file.keys():
                h5file.create_group('data')

        for idx, row in mouseRows.iterrows():
            sessionName = row['day'] + '_' + row['session']

            t1 = self.load_t1(self.pathT1[mouseName][row['day']])

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
                            imgT1, imgT2 = self.transform_img(imgDS, t2, t1=t1)
                            dataTrialTr += [imgT2]

                        dataImgTr = np.array(dataTrialTr)
                        dataRSP += [self.extract_channel_data(dataImgTr)]

                    except:
                        print('---warning, reading video failed, filling with NAN')
                        dataRSP += [np.full(dataRSP[-1].shape, np.nan)]

                with h5py.File(mouseName + '.h5', 'a') as h5file:
                    h5file['data'].create_dataset(sessionName, data=np.array(dataRSP))

    # Different mice have different Go/NoGo testures
    def tex_go_nogo_bymouse(self, mouseName):
        # return 'P100', 'P1200'
        if (mouseName == 'mou_5') or (mouseName == 'mou_7'):
            return 'P100', 'P1200'
        else:
            return 'P1200', 'P100'

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

    # Count number of trial types of each type. Ensure absent trials have 0 counts
    def count_trial_types(self, df):
        targetTypes = ['Hit', 'Miss', 'CR', 'FA']
        d = df['trialType'].value_counts()
        rezLst = [d[k] if k in d.keys() else 0 for k in targetTypes]
        return dict(zip(targetTypes, rezLst))

    # Read all structure files, process, save to H5. Also compute performance and save to H5
    def process_metadata_files(self):
        for mouseName, mouseRows in self.dataPaths.groupby(['mouse']):
            h5name = mouseName + '.h5'
            self._h5_append_group(h5name, 'metadata')
            self._h5_append_group(h5name, 'accuracy')
            self._h5_append_group(h5name, 'dprime')

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
                    ttDict = self.count_trial_types(dfTrialStruct)
                    acc = accuracy(ttDict['Hit'], ttDict['Miss'], ttDict['FA'], ttDict['CR'])
                    dp = d_prime(ttDict['Hit'], ttDict['Miss'], ttDict['FA'], ttDict['CR'])

                    with h5py.File(h5name, 'a') as h5f:
                        h5f['accuracy'].create_dataset(sessionName, data=acc)
                        h5f['dprime'].create_dataset(sessionName, data=dp)

    # For a given session, compute time of each timestep of each trial relative to start of session
    # Return as 2D array (nTrial, nTime)
    def get_pooled_data_rel_times(self, pwd, mouseName, session, FPS=20.0):
        fpath = os.path.join(pwd, mouseName + '.h5')
        with h5py.File(fpath) as f:
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

    # Fit polynomial to RSP data, return fit
    def polyfit_data_3D(self, times, dataRSP, ord, alpha):
        timesFlat = times.flatten()
        dataFlat = numpy_merge_dimensions(dataRSP, 0, 2)

        rez = np.zeros(dataRSP.shape)
        for iCh in range(dataRSP.shape[2]):
            # y = polyfit.poly_fit_transform(timesFlat, dataFlat[:, iCh], ord)
            y = natural_cubic_spline_fit_reg(timesFlat, dataFlat[:, iCh], dof=ord, alpha=alpha)
            rez[:, :, iCh] = y.reshape(times.shape)
        return rez

    # Plot data of a few channels throughout the whole session
    def example_poly_fit(self, pwd, mouseName, session, iCh=0, ord=2, alpha=0.01):
        times, dataRSP = self.get_pooled_data_rel_times(pwd, mouseName, session)
        timesFlat = times.flatten()
        dataFlat = numpy_merge_dimensions(dataRSP, 0, 2)

        print(times.shape, dataRSP.shape)

        nTrial, nTime, nChannel = dataRSP.shape
        plt.figure(figsize=(8, 4))
        # y = polyfit.poly_fit_transform(timesFlat, dataFlat[:, iCh], ord)
        y = natural_cubic_spline_fit_reg(timesFlat, dataFlat[:, iCh], dof=ord, alpha=alpha)

        for iTr in range(nTrial):
            plt.plot(times[iTr], dataRSP[iTr, :, iCh], color='orange')
        plt.plot(timesFlat, y)
        plt.show()

    # For each trial, compute DFF, store back to h5
    def baseline_subtraction_dff(self, pwd, iMin, iMax, skipExist=False):
        for mouseName, dfMouse in self.dataPaths.groupby(['mouse']):
            h5fname = mouseName + '.h5'

            self._h5_append_group(h5fname, 'bn_trial')

            for idx, row in dfMouse.iterrows():
                session = row['day'] + '_' + row['session']
                with h5py.File(h5fname, 'a') as h5f:
                    if session in h5f['bn_trial'].keys():
                        if skipExist:
                            del h5f['bn_trial'][session]
                        else:
                            print(mouseName, session, 'already exists, skipping')
                            continue

                print(mouseName, session)
                times, dataRSP = self.get_pooled_data_rel_times(pwd, mouseName, session)
                if len(times) != len(dataRSP):
                    print('-- trial mismatch', times.shape, dataRSP.shape)
                    continue

                dataBN = np.zeros(dataRSP.shape)

                for iTr in range(dataRSP.shape[0]):
                    mu = np.nanmean(dataRSP[iTr, iMin:iMax], axis=0)
                    dataBN[iTr] = dataRSP[iTr] / mu - 1

                with h5py.File(h5fname, 'a') as h5f:
                    h5f['bn_trial'].create_dataset(session, data=dataBN)

    # For each session: fit poly, do poly-DFF, store back to h5
    def baseline_subtraction_poly(self, pwd, ord=2, alpha=0.01, skipExist=False):
        for mouseName, dfMouse in self.dataPaths.groupby(['mouse']):
            h5fname = mouseName + '.h5'

            self._h5_append_group(h5fname, 'bn_session')
            self._h5_append_group(h5fname, 'bn_fit')
            self._h5_append_group(h5fname, 'raw')

            for idx, row in dfMouse.iterrows():
                session = row['day'] + '_' + row['session']
                with h5py.File(h5fname, 'r') as h5f:
                    if session in h5f['bn_session'].keys():
                        if skipExist:
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
                    continue

                dataRSPfit = self.polyfit_data_3D(times, dataRSP, ord, alpha)

                dataRaw = dataRSP - dataRSPfit
                dataBN = dataRaw / dataRSPfit

                with h5py.File(h5fname, 'a') as h5f:
                    h5f['raw'].create_dataset(session, data=dataRaw)
                    h5f['bn_fit'].create_dataset(session, data=dataRSPfit)
                    h5f['bn_session'].create_dataset(session, data=dataBN)
