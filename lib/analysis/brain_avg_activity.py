import numpy as np
import matplotlib.pyplot as plt

from mesostat.utils.arrays import numpy_merge_dimensions
from mesostat.utils.signals.resample import resample_kernel_same_interv
from sklearn.decomposition import PCA


def _resample_delay(dataRSP, tStart, tStop, FPS=20.0, padReward=False):
    iStart = int(tStart * FPS)
    iStop = int(tStop * FPS)
    nDelTrg = int(2.0 * FPS)
    nRewTrg = int(1.0 * FPS)

    dataPRE = dataRSP[:, :iStart]
    dataDEL = dataRSP[:, iStart:iStop]
    dataREW = dataRSP[:, iStop:]

    nDel = dataDEL.shape[1]
    nRew = dataREW.shape[1]

    # Resample delay to 2s
    W = resample_kernel_same_interv(nDel, nDelTrg)
    dataDEL = np.einsum('lj,ijk->ilk', W, dataDEL)

    if padReward:
        if nRew > nRewTrg:
            # Crop reward to 1 second if it exceeds
            dataREW = dataREW[:, :nRewTrg]
        elif nRew < nRewTrg:
            # Pad reward with NAN if it is too short
            tmp = np.full((dataREW.shape[0], nRewTrg, dataREW.shape[2]), np.nan)
            tmp[:, :nRew] = dataREW
            dataREW = tmp

    return np.concatenate([dataPRE, dataDEL, dataREW], axis=1)


def _get_data_concatendated_sessions(dataDB, haveDelay, mousename, datatype, trialType):
    if not haveDelay:
        dataRSPLst = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype, trialType=trialType)
    else:
        dataRSPLst = []
        for session in dataDB.get_sessions(mousename):
            dataRSP = dataDB.get_neuro_data({'session': session}, datatype=datatype, trialType=trialType)[0]
            delayStart = dataDB.get_interval_times(session, mousename, 'DEL')[0][0]
            delayEnd = delayStart + dataDB.get_delay_length(mousename, session)
            dataRSPLst += [_resample_delay(dataRSP, delayStart, delayEnd, FPS=dataDB.targetFreq, padReward=True)]

    # Stack sessions into trials
    return np.concatenate(dataRSPLst, axis=0)


def plot_pca1_session(dataDB, mousename, session, trialTypesSelected=('Hit', 'CR')):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    for iDataType, datatype in enumerate(['bn_trial', 'bn_session']):
        timesRS = dataDB.get_absolute_times(mousename, session)
        dataRSP = dataDB.get_neuro_data({'session': session}, datatype=datatype)[0]

        # Train PCA on whole session, but only trial based timesteps
        dataSP = numpy_merge_dimensions(dataRSP, 0, 2)
        timesS = numpy_merge_dimensions(timesRS, 0, 2)
        pca = PCA(n_components=1)
        dataPCA1 = pca.fit_transform(dataSP)[:, 0]
        pcaSig = np.sign(np.mean(pca.components_))
        pcaTransform = lambda x: pca.transform(x)[:, 0] * pcaSig

        # Compute 1st PCA during trial-time
        # Note: it is irrelevant whether averaging or PCA-transform comes first
        trialTypes = dataDB.get_trial_types(session, mousename)
        timesTrial = dataDB.get_times(dataRSP.shape[1])

        for tt in trialTypesSelected:
            dataAvgTTSP = np.mean(dataRSP[trialTypes == tt], axis=0)
            dataAvgTTPCA = pcaTransform(dataAvgTTSP)
            dataAvgTTAvg = np.mean(dataAvgTTSP, axis=1)

            ax[iDataType, 1].plot(timesTrial, dataAvgTTAvg, label=tt)
            ax[iDataType, 2].plot(timesTrial, dataAvgTTPCA, label=tt)

        ax[iDataType, 0].set_ylabel(datatype)
        ax[iDataType, 0].plot(timesS, dataPCA1)
        ax[iDataType, 0].set_title('1st PCA during session')

        ax[iDataType, 1].set_title('Trial-average activity')
        ax[iDataType, 1].legend()

        ax[iDataType, 2].set_title('1st PCA trial-average')
        ax[iDataType, 2].legend()

        dataDB.label_plot_timestamps(ax[iDataType, 1])
        dataDB.label_plot_timestamps(ax[iDataType, 2])
    plt.show()


def plot_pca1_mouse(dataDB, trialTypesSelected=('Hit', 'CR'), skipReward=None):
    haveDelay = 'DEL' in dataDB.get_interval_names()
    nMice = len(dataDB.mice)

    for iDataType, datatype in enumerate(['bn_trial', 'bn_session']):
        fig, ax = plt.subplots(nrows=2, ncols=nMice, figsize=(4*nMice, 8))
        fig.suptitle(datatype)

        for iMouse, mousename in enumerate(sorted(dataDB.mice)):
            # Train PCA on whole dataset, but only trial based timesteps
            dataRSP = _get_data_concatendated_sessions(dataDB, haveDelay, mousename, datatype, None)
            timesTrial = dataDB.get_times(dataRSP.shape[1])

            dataSP = numpy_merge_dimensions(dataRSP, 0, 2)
            dataSP = dataSP[~np.any(np.isnan(dataSP), axis=1)]

            pca = PCA(n_components=1)
            pca.fit(dataSP)
            pcaSig = np.sign(np.mean(pca.components_))
            pcaTransform = lambda x: pca.transform(x)[:, 0] * pcaSig

            for trialType in trialTypesSelected:
                # Evaluate on individual trials
                dataRSP = _get_data_concatendated_sessions(dataDB, haveDelay, mousename, datatype, trialType)
                dataSP = np.nanmean(dataRSP, axis=0)
                dataPCA = pcaTransform(dataSP)
                dataAvg = np.mean(dataSP, axis=1)
                dataStd = np.mean(np.nanstd(dataRSP, axis=2), axis=0) / np.sqrt(dataRSP.shape[0])

                ax[0, iMouse].plot(timesTrial, dataAvg, label=trialType)
                ax[1, iMouse].plot(timesTrial, dataPCA, label=trialType)
                ax[0, iMouse].fill_between(timesTrial, dataAvg-dataStd, dataAvg+dataStd, alpha=0.2)

            dataDB.label_plot_timestamps(ax[0, iMouse])
            dataDB.label_plot_timestamps(ax[1, iMouse])
            # ax[0, iMouse].legend()
            # ax[1, iMouse].legend()
            ax[0, iMouse].set_title(mousename)

        ax[0, 0].set_ylabel('Trial-average activity')
        ax[1, 0].set_ylabel('1st PCA trial-average')
        plt.show()
