import os
import h5py
import dcimg
import numpy as np
import pymatreader
import matplotlib.pyplot as plt
import skimage.transform as skt
from scipy.spatial import ConvexHull

from mesostat.utils.matlab_helper import loadmat
from mesostat.utils.arrays import numpy_merge_dimensions
from mesostat.utils.signals.fit import natural_cubic_spline_fit_reg
from mesostat.utils.signals.resample import resample_kernel


#############################
# OS
#############################

# Test if folder is found
def _testdir(path, critical=True):
    rez = os.path.isdir(path)
    if not rez:
        if critical:
            raise ValueError('Not found path:', path)
        else:
            print('Not found path:', path)
    return rez


# Test if file is found
def _testfile(path, critical=True):
    rez = os.path.isfile(path)
    if not rez:
        if critical:
            raise ValueError('WARNING: Not found file:', path)
        else:
            print('WARNING: Not found file:', path)
    return rez

#############################
# H5PY
#############################

def _h5_append_group(h5path, group, overwrite=False):
    with h5py.File(h5path, 'a') as h5file:
        if group not in h5file.keys():
            h5file.create_group(group)
        elif overwrite:
            del h5file[group]
            h5file.create_group(group)


#############################
# Video processing and transform
#############################


# Load T1: Affine Transformation from file
def load_t1(path):
    d = pymatreader.read_mat(path)['tform']['tdata']
    t = d['T']
    tinv = d['Tinv']
    assert np.linalg.norm(t.dot(tinv) - np.eye(3)) < 1.0E-10
    return t


# Load T2: Polynomial Transformation from file
def load_t2(path):
    d = pymatreader.read_mat(path)['tform_str']
    assert d['Degree'] == 3
    assert d['Dimensionality'] == 2

    # Canonical Monads:  1, x, y, x2, xy, y2, x3, x2y, xy2, x3
    # Matlab Monads:     1, x, y, xy, x2, y2, x2y, xy2, x3, y3

    monadOrder = [0, 1, 2, 4, 3, 5, 8, 6, 7, 9]
    return d['A'][monadOrder], d['B'][monadOrder]


# Load reference image from path
def load_ref_img(path):
    return loadmat(path)['refImg']


# Plot the results of transforms
def plot_transforms(imgAllen, img, imgT2, imgT1=None):
    fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
    ax[0].imshow(img)
    ax[0].set_title('Raw')

    if imgT1 is not None:
        ax[1].imshow(imgT1)
        ax[1].set_title('T1')

    # Overlay Allen
    imgT2 = imgT2 / np.nanmax(imgT2) + 0.3 * (imgAllen == 0).astype(int)

    ax[2].imshow(imgT2)
    ax[2].set_title('T2')
    plt.show()


# Apply T1 and T2 to a flat 2D image. Return results of each transform
def transform_img(img, t2, t1=None):
    cval = 100 * np.nanmax(img)
    # print(cval)

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


# Apply transforms to reference files, overlay allen. Inspect by eye
def test_transform_ref(pathT2Dict, pathRefDict):
    for mousename, t2path in pathT2Dict.items():
        t2data = load_t2(t2path)
        refImg = load_ref_img(pathRefDict[mousename])
        transImg = transform_img(refImg, t2data)

        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))
        fig.set_title(mousename)
        ax[0].imshow(refImg)
        ax[1].imshow(transImg)
        plt.show()


# Apply transforms to time-averaged single trials
# Average result over examples from different days
#  1) T1 must make averages over days less blurry, as it aligns different days to same position
def test_transform_vids(pathsMouse, pathT1mouse, pathT2mouse, imgAllen, parse_video_paths):
    imgLst = []
    imgT1Lst = []
    imgT2Lst = []
    for day in set(pathsMouse['day']):
        print('--', day)
        pathsDay = pathsMouse[pathsMouse['day'] == day].reset_index()
        sessionPath = pathsDay['sessionPath'][0]

        filePaths = parse_video_paths(sessionPath, day)
        data = dcimg.DCIMGFile(filePaths[0])[:]

        A, B = load_t2(pathT2mouse)

        t1 = load_t1(pathT1mouse[day])

        imgMean = np.mean(data, axis=0)
        imgDS = skt.downscale_local_mean(imgMean, (2, 2))
        imgT1, imgT2 = transform_img(imgDS, np.array([A, B]), t1=t1)

        imgLst += [imgDS]
        imgT1Lst += [imgT1]
        imgT2Lst += [imgT2]

    imgLst = np.mean(imgLst, axis=0)
    imgT1Lst = np.mean(imgT1Lst, axis=0)
    imgT2Lst = np.mean(imgT2Lst, axis=0)
    plot_transforms(imgAllen, imgLst, imgT2Lst, imgT1Lst)


# Convert transformed video -> (nTime, nChannel)
def extract_channel_data(vid, imgAllen):
    keys = sorted(set(imgAllen.flatten()))[2:]
    assert len(keys) == 27

    rez = np.zeros((vid.shape[0], 27))
    for ikey, key in enumerate(keys):
        idxs = imgAllen == key
        # print(vid.shape)
        vidPix = vid[:, idxs]
        # print(vidPix.shape)
        rez[:, ikey] = np.nanmean(vidPix, axis=1)

    return rez


#############################
# Metadata
#############################

# Count number of trial types of each type. Ensure absent trials have 0 counts
def count_trial_types(df):
    targetTypes = ['Hit', 'Miss', 'CR', 'FA']
    d = df['TrialType'].value_counts()
    rezLst = [d[k] if k in d.keys() else 0 for k in targetTypes]
    return dict(zip(targetTypes, rezLst))


def parse_active_passive(dfTrialStruct, activePassivePath, mapCanon):
    dfTrialStruct['Activity'] = None

    activePassiveStruct = pymatreader.read_mat(activePassivePath)
    for tt, ttCanon in mapCanon.items():
        rezDict = {}

        for activity in ['delay_move', 'no_prior_move', 'noisy', 'prior_move', 'quiet_sens', 'quiet_then_move']:
            keyAct = 'tr_' + tt + '_' + activity
            if keyAct in activePassiveStruct:
                keys = activePassiveStruct[keyAct]
                if isinstance(keys, int):
                    keys = [keys]

                vals = [activity] * len(keys)
                rezDict = {**rezDict, **dict(zip(keys, vals))}

        print('--', tt, (dfTrialStruct['trialType'] == ttCanon).sum(), len(rezDict))

        iTT = 0
        for idx, row in dfTrialStruct.iterrows():
            if row['trialType'] == ttCanon:
                if iTT + 1 in rezDict:
                    dfTrialStruct.loc[idx, 'Activity'] = rezDict[iTT + 1]
                iTT += 1

    return dfTrialStruct


def calc_allen_shortest_distances(allenMap, allenIndices):
    pointsBoundary = {}
    for i in allenIndices:
        if i > 1:
            points = np.array(np.where(allenMap == i))
            pointsBoundary[i] = points[:, ConvexHull(points.T).vertices].T

    nRegion = len(pointsBoundary)
    minDist = np.zeros((nRegion, nRegion))
    for i, iPoints in enumerate(pointsBoundary.values()):
        for j, jPoints in enumerate(pointsBoundary.values()):
            if i < j:
                minDist[i][j] = 10000000.0
                for p1 in iPoints:
                    for p2 in jPoints:
                        minDist[i][j] = np.min([minDist[i][j], np.linalg.norm(p1 - p2)])
                minDist[j][i] = minDist[i][j]

    return minDist

############################
#  Baseline Subtraction
############################

# Fit polynomial to RSP data, return fit
def polyfit_data_3D(times, dataRSP, ord, alpha):
    timesFlat = times.flatten()
    dataFlat = numpy_merge_dimensions(dataRSP, 0, 2)

    rez = np.zeros(dataRSP.shape)
    for iCh in range(dataRSP.shape[2]):
        if np.any(np.isnan(dataFlat[:, iCh])):
            print(iCh, len(dataFlat), np.sum(np.isnan(dataFlat[:, iCh])))

        # y = polyfit.poly_fit_transform(timesFlat, dataFlat[:, iCh], ord)
        y = natural_cubic_spline_fit_reg(timesFlat, dataFlat[:, iCh], dof=ord, alpha=alpha)
        rez[:, :, iCh] = y.reshape(times.shape)
    return rez


# Plot data of a few channels throughout the whole session
def example_poly_fit(times, dataRSP, iCh=0, ord=2, alpha=0.01):
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
    plt.ylabel(str(iCh))
    plt.show()

############################
#  Behaviour
############################


def _behaviour_resample(matRS, nSampleSrc, srcFreq, trgFreq, sig2):
    # Figure out timings of source and target timesteps
    nSampleTrg = int(nSampleSrc / srcFreq * trgFreq)
    timesSrc = np.arange(nSampleSrc) / srcFreq
    timesTrg = np.arange(nSampleTrg) / trgFreq

    # Perform resample
    K = resample_kernel(timesSrc, timesTrg, sig2=sig2)
    rezRS = matRS.dot(K.T)
    return timesSrc, timesTrg, rezRS


def behaviour_tune_resample_kernel(pwd, sig2, trialType='Hit', trialIdx=0, srcFreq=30.0, trgFreq=20.0):
    # Read behaviour matrix
    matDict = pymatreader.read_mat(pwd)
    print(matDict.keys())

    matRS = matDict[trialType].T
    nTrialSrc, nSampleSrc = matRS.shape

    print(matRS.shape)
    timesSrc, timesTrg, rezRS = _behaviour_resample(matRS, nSampleSrc, srcFreq, trgFreq, sig2)

    # Plot original and resampled data
    plt.figure(figsize=(10,4))
    plt.plot(timesSrc, matRS[trialIdx], label='src')
    plt.plot(timesTrg, rezRS[trialIdx], label='trg')
    plt.legend()
    plt.show()


def read_resample_movement_data(pwdMove, trialTypeNames, nTrialData, nSampleData, trialNameMap, sig2, srcFreq=30.0, trgFreq=20.0):
    # Make array of NAN nTrial x nTime (as in data)
    movementRS = np.full((nTrialData, nSampleData), np.nan)

    # Read behaviour matrix
    matDict = pymatreader.read_mat(pwdMove)

    # for each trialType, resample 30Hz to 20Hz (3-step window), fill matrix
    for srcName, trgName in trialNameMap.items():
        if srcName in matDict.keys():
            movementRawRS = matDict[srcName].T  # Original is SR, so transpose

            if movementRawRS.ndim == 2:
                nTrialSrc, nSampleSrc = movementRawRS.shape
            elif movementRawRS.ndim == 1:
                nTrialSrc, nSampleSrc = 1, movementRawRS.shape[0]
                if nSampleSrc == 0:
                    nTrialSrc = 0
            else:
                raise IOError("Unexpected shape", movementRawRS.shape)

            # Test that the number of behavioural trials data trials of this type match
            # print(trgName, nTrialSrc, np.sum(trialTypeNames == trgName))

            if nTrialSrc == 0:
                print('--No trials for ', trgName, ', skipping')
                continue

            timesSrc, timesTrg, rezRS = _behaviour_resample(movementRawRS, nSampleSrc, srcFreq, trgFreq, sig2)

            # Find number of target trials and timesteps
            trialTypeIdxs = trialTypeNames == trgName
            nTrialTrg = np.sum(trialTypeIdxs)
            nSampleTrg = len(timesTrg)

            # Augment or crop number of trials
            if nTrialSrc != nTrialTrg:
                print('Warning: Trial mismatch:', trgName, nTrialSrc, np.sum(trialTypeNames == trgName), ': cropping')
                if nTrialSrc > nTrialTrg:
                    rezRS = rezRS[:nTrialTrg]
                else:
                    tmpRS = np.full((nTrialTrg, nSampleTrg), np.nan)
                    tmpRS[:nTrialSrc] = rezRS
                    rezRS = tmpRS

            # Augment or crop duration
            if nSampleTrg > nSampleData:
                print('too long, crop', nSampleTrg, nSampleData)
                movementRS[trialTypeIdxs] = rezRS[:, :nSampleData]  # If behaviour too long, crop it
            elif nSampleTrg > nSampleData:
                print('too short, pad', nSampleTrg, nSampleData)
                movementRS[trialTypeIdxs, :nSampleTrg] = rezRS  # If behaviour too short, pad it
            else:
                movementRS[trialTypeIdxs] = rezRS

    return movementRS
