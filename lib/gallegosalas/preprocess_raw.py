import json
import h5py
import dcimg
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pymatreader
from scipy.ndimage import affine_transform
import skimage.transform as skt

from mesostat.utils.pandas_helper import pd_append_row
from mesostat.utils.matlab_helper import loadmat
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
        rez = [img]
        if t1 is not None:
            at = skt.AffineTransform(t1.T)
            rez += [skt.warp(img, at.inverse)]

        A, B = t2
        pt = skt.PolynomialTransform(np.array([A, B]))
        rez += [skt.warp(rez[-1], pt)]

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
        imgT2 = imgT2[::-1].T / np.max(imgT2) + 0.3 * (self.allen == 0).astype(int)

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

            filesTmp = os.listdir(sessionPath)
            filesTmp = [f for f in filesTmp if f[:4] == day[:4]]

            filePath = os.path.join(sessionPath, filesTmp[0])
            data = dcimg.DCIMGFile(filePath)[:]

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
            print(vid.shape)
            vidPix = vid[:, idxs]
            print(vidPix.shape)
            rez[:, ikey] = np.mean(vidPix, axis=1)

        return rez

    # Read video files, extract channel data and save to HDF5
    def process_video_files(self, skipExisting=True):
        for mouseName, mouseRows in self.dataPaths.groupby(['mousename']):
            t2 = np.array(self.load_t2(self.pathT2[mouseName]))

            with h5py.File(mouseName + '.h5', 'a') as h5file:
                if 'data' not in h5file.keys():
                    h5file.create_group('data')

                for idx, row in mouseRows.iterrows():
                    sessionName = row['day'] + '_' + row['session']

                    t1 = self.load_t1(self.pathT1[mouseName][row['day']])

                    if (sessionName in h5file['data'].keys()) and skipExisting:
                        print('Have', sessionName, '; skipping')
                    else:
                        pathFiles = row['sessionPath']
                        filesTmp = os.listdir(pathFiles)
                        filesTmp = [f for f in filesTmp if row['day'] in f]

                        dataRSP = []
                        for vidfname in filesTmp:
                            vidfpath = os.path.join(pathFiles, vidfname)
                            dataTrial = dcimg.DCIMGFile(vidfpath)[:]

                            dataTrialTr = []
                            for img in dataTrial:
                                imgDS = skt.downscale_local_mean(img, (2, 2))
                                imgT1, imgT2 = self.transform_img(imgDS, t2, t1=t1)
                                dataTrialTr += [imgT2]

                            dataImgTr = np.array(dataTrialTr)
                            dataRSP += [self.extract_channel_data(dataImgTr)]

                        h5file['data'].create_dataset(sessionName, data=np.array(dataRSP))
