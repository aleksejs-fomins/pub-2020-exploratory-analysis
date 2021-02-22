import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from scipy.stats import mannwhitneyu
from sklearn.linear_model import RidgeClassifier, LogisticRegression

from mesostat.utils.signals import zscore, resample
from mesostat.utils.pandas_helper import pd_query, pd_is_one_row
from mesostat.visualization.mpl_colorbar import imshow_add_color_bar
from mesostat.stat.classification import binary_classifier

from lib.sych.preprocess_raw import poly_fit_transform


def plot_session(dataDB, session):
    dataThis, tiThis, itiThis, fps, trialTypes = dataDB.get_data_raw(session)

    channelIdx = 0
    dataThisCh = dataThis[:, channelIdx]

    nTimePre = 2.0
    nTimePost = 8.0
    nTimeStep = len(dataThis)
    times = np.arange(nTimeStep) / fps

    yMin = np.min(dataThisCh)
    yMax = np.max(dataThisCh)

    # Trial boxes
    patchesPre = []
    patchesPost = []
    for idx in tiThis:
        patchesPre += [matplotlib.patches.Rectangle((idx / fps - nTimePre, yMin), nTimePre, yMax - yMin)]
        patchesPost += [matplotlib.patches.Rectangle((idx / fps, yMin), nTimePost, yMax - yMin)]

    # Background fitting
    nPointRes = 400
    paramRes = {'method': 'kernel', 'kind': 'gau', 'ker_sig2': (times[-1] / nPointRes) ** 2}
    tResampled = np.linspace(0, times[-1], nPointRes)
    dataFitted = poly_fit_transform(times, dataThisCh, 5)
    dataDiff = dataThisCh - dataFitted
    dataResampled = resample(times, dataThisCh, tResampled, param=paramRes)
    dataDiffResampled = resample(times, dataDiff, tResampled, param=paramRes)

    fig, ax = plt.subplots(nrows=2, figsize=(10, 4))
    ax[0].plot(times, dataThisCh)
    ax[0].plot(times, dataFitted)
    ax[0].plot(tResampled, dataResampled)
    ax[0].add_collection(PatchCollection(patchesPre, alpha=0.3, color='y', edgecolor='none'))
    ax[0].add_collection(PatchCollection(patchesPost, alpha=0.3, color='g', edgecolor='none'))
    ax[1].plot(times, dataDiff)
    ax[1].plot(tResampled, dataDiffResampled)
    ax[0].set_ylabel('RAW')
    ax[1].set_ylabel('BG-SUB')
    plt.show()


def prepare_data(dataDB, intervalsTrg, bgSub=True):
    trialTypesTrg = {'iGO', 'iNOGO'}

    dataIndexed = []
    dataDF = pd.DataFrame()
    for mousename in dataDB.mice:
        for session in dataDB.get_sessions(mousename):
            dataThis, tiThis, itiThis, fps, trialTypes = dataDB.get_data_raw(session)

            if bgSub:
                times = np.arange(len(dataThis))
                for iCh in range(dataThis.shape[1]):
                    dataFitted = poly_fit_transform(times, dataThis[:, iCh], 15)
                    dataThis[:, iCh] -= dataFitted

            for trialType in trialTypesTrg:
                trialIdxs = trialTypes == trialType
                nTrials = np.sum(trialIdxs)
                if nTrials < 50:
                    print('Too few trials =', nTrials, ' for', session, trialType, ': skipping')
                else:
                    tiTrials = tiThis[trialIdxs]

                    for intervName, intervRng in intervalsTrg.items():
                        dataInterv = []
                        for idx in tiTrials:
                            idxL = int(idx + intervRng[0] * fps)
                            idxR = int(idx + intervRng[1] * fps + 1)
                            dataInterv += [np.mean(dataThis[idxL:idxR, :48], axis=0)]

                        dataIndexed += [np.array(dataInterv)]
                        dataDF = dataDF.append({'mousename': mousename, 'session': session,
                                                'trialType': trialType, 'interval': intervName}, ignore_index=True)

    return dataIndexed, dataDF


def test_prediction(dataDB, prepData, prepDF, intervalsTrg):
    # classifier = LogisticRegression(max_iter=10000, C=1.0E-2, solver='lbfgs')
    classifier = RidgeClassifier(max_iter=10000, alpha=1.0E-2)

    for mousename in sorted(dataDB.mice):
        sessions = dataDB.get_sessions(mousename)

        nSessions = len(sessions)
        nIntervals = len(intervalsTrg)

        figTest, axTest = plt.subplots(ncols=3, figsize=(10, 5))
        figClass, axClass = plt.subplots(ncols=3, figsize=(10, 5))
        figTest.suptitle(mousename)
        figClass.suptitle(mousename)

        for iInterv, (intervName, intervRng) in enumerate(intervalsTrg.items()):
            testMat = np.zeros((48, nSessions))
            accLst = []

            for iSession, session in enumerate(sessions):
                print(intervName, session)

                queryDict = {'mousename': mousename, 'session': session, 'interval': intervName}
                rowGo = pd_query(prepDF, {**queryDict, **{'trialType': 'iGO'}})
                rowNogo = pd_query(prepDF, {**queryDict, **{'trialType': 'iNOGO'}})

                if (len(rowGo) == 0) or (len(rowNogo) == 0):
                    print('Skipping session', session, 'because too few trials')
                    testMat[:, iSession] = np.nan
                    accLst += [{'accTrain': np.nan, 'accTest': np.nan}]
                else:
                    idxRowGO, _ = pd_is_one_row(rowGo)
                    idxRowNOGO, _ = pd_is_one_row(rowNogo)
                    dataGO = prepData[idxRowGO]
                    dataNOGO = prepData[idxRowNOGO]

                    # Doing pairwise testing on individual channels
                    for iCh in range(48):
                        p = mannwhitneyu(dataGO[:, iCh], dataNOGO[:, iCh], alternative='two-sided')[1]
                        testMat[iCh, iSession] = -np.log10(p)

                    # Doing classification
                    accLst += [binary_classifier(dataGO, dataNOGO, classifier,
                                                 method="looc", balancing=False)]

            # Plot test
            axTest[iInterv].set_title(intervName)
            img = axTest[iInterv].imshow(testMat, vmin=0, vmax=10)
            imshow_add_color_bar(figTest, axTest[iInterv], img)

            # Plot classification
            axClass[iInterv].set_title(intervName)
            axClass[iInterv].plot([l['accTrain'] for l in accLst], label='train')
            axClass[iInterv].plot([l['accTest'] for l in accLst], label='test')
            axClass[iInterv].axhline(y=0.5, linestyle='--', color='pink')
            axClass[iInterv].set_xlim(0, len(sessions))
            axClass[iInterv].set_ylim(0, 1)
            axClass[iInterv].legend()

        plt.show()
