import numpy as np
import matplotlib.pyplot as plt

from mesostat.utils.arrays import numpy_merge_dimensions
from sklearn.decomposition import PCA


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
