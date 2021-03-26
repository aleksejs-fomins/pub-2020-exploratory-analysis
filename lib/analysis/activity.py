import numpy as np
import matplotlib.pyplot as plt

from mesostat.utils.pandas_helper import pd_query
from scipy.stats import mannwhitneyu, wilcoxon
from mesostat.visualization.mpl_matrix import imshow


def compute_mean_interval(dataDB, ds, trialTypesTrg, intervDict):
    for iMouse, mousename in enumerate(sorted(dataDB.mice)):
        for datatype in dataDB.get_data_types():
            for trialType in trialTypesTrg:
                for intervName, interv in intervDict.items():
                    print(mousename, datatype, trialType, intervName)

                    for session in dataDB.get_sessions(mousename, datatype=datatype):
                        dataRSP = dataDB.get_neuro_data({'session': session}, datatype=datatype,
                                                        cropTime=interv, trialType=trialType)[0]

                        dataRP = np.mean(dataRSP, axis=1)

                        attrsDict = {
                            'datatype': datatype,
                            'session': session,
                            'trialType': trialType,
                            'interv': intervName
                        }

                        ds.save_data('mean', dataRP, attrsDict)


def plot_consistency_significant_activity_byaction(dataDB, ds):
    rows = ds.list_dsets_pd()
    rows['mousename'] = [dataDB.find_mouse_by_session(session) for session in rows['session']]

    for (datatype, intervName), rowsMouse in rows.groupby(['datatype', 'interv']):
        pSigDict = {}
        for mousename, rowsSession in rowsMouse.groupby(['mousename']):
            pSig = []
            for session, rowsTrial in rowsSession.groupby(['session']):
                print(session, datatype, intervName)

                dataThis = []
                for idx, row in rowsTrial.iterrows():
                    dataThis += [ds.get_data(row['dset'])]

                nTrials1 = dataThis[0].shape[0]
                nTrials2 = dataThis[1].shape[0]
                if (nTrials1 != 0) and (nTrials2 != 0):
                    nChannels = dataThis[0].shape[1]
                    pvals = [mannwhitneyu(dataThis[0][:, iCh], dataThis[1][:, iCh], alternative='two-sided')[1]
                             for iCh in range(nChannels)]
                    #             pSig += [(np.array(pvals) < 0.01).astype(int)]
                    pSig += [-np.log10(np.array(pvals))]
            #         pSigDict[mousename] = np.sum(pSig, axis=0)
            pSigDict[mousename] = np.mean(pSig, axis=0)

        mice = sorted(dataDB.mice)
        nMice = len(mice)
        fig1, ax1 = plt.subplots(nrows=nMice, ncols=nMice, figsize=(2 * nMice, 2 * nMice), tight_layout=True)

        corrCoef = np.zeros((nMice, nMice))
        for iMouse, iName in enumerate(mice):
            ax1[iMouse][0].set_ylabel(iName)
            ax1[0][iMouse].set_title(iName)

            for jMouse, jName in enumerate(mice):
                ax1[iMouse][jMouse].plot(pSigDict[jName], pSigDict[iName], '.')  # Coefficient flip intended
                corrCoef[iMouse, jMouse] = np.corrcoef(pSigDict[iName], pSigDict[jName])[0, 1]

        plt.savefig('pics/consistency_significant_activity_bymouse_' + datatype + '_' + intervName + '.png')
        plt.close()

        fig2, ax2 = plt.subplots()
        ax2.imshow(corrCoef, vmin=0, vmax=1)
        imshow(fig2, ax2, corrCoef, title='Significance Correlation', haveColorBar=True, limits=[0, 1],
               xTicks=mice, yTicks=mice)

        plt.savefig('pics/consistency_significant_activity_bymouse_corr_' + datatype + '_' + intervName + '.png')
        plt.close()


def plot_consistency_significant_activity_byphase(dataDB, ds):
    rows = ds.list_dsets_pd()
    rows['mousename'] = [dataDB.find_mouse_by_session(session) for session in rows['session']]

    for (datatype, trialType), rowsMouse in rows.groupby(['datatype', 'trialType']):
        pSigDict = {}
        for mousename, rowsSession in rowsMouse.groupby(['mousename']):
            pSig = []
            for session, rowsTrial in rowsSession.groupby(['session']):
                print(session, datatype, trialType)

                dataThis = []
                for idx, row in rowsTrial.iterrows():
                    if row['interv'] != 'PRE':
                        dataThis += [ds.get_data(row['dset'])]

                nTrials1 = dataThis[0].shape[0]
                nTrials2 = dataThis[1].shape[0]
                if (nTrials1 != 0) and (nTrials2 != 0):
                    nChannels = dataThis[0].shape[1]
                    pvals = [wilcoxon(dataThis[0][:, iCh], dataThis[1][:, iCh], alternative='two-sided')[1]
                             for iCh in range(nChannels)]
                    #             pSig += [(np.array(pvals) < 0.01).astype(int)]
                    pSig += [-np.log10(np.array(pvals))]
            #         pSigDict[mousename] = np.sum(pSig, axis=0)
            pSigDict[mousename] = np.mean(pSig, axis=0)

        mice = sorted(dataDB.mice)
        nMice = len(mice)
        fig1, ax1 = plt.subplots(nrows=nMice, ncols=nMice, figsize=(2 * nMice, 2 * nMice), tight_layout=True)

        corrCoef = np.zeros((nMice, nMice))
        for iMouse, iName in enumerate(mice):
            ax1[iMouse][0].set_ylabel(iName)
            ax1[0][iMouse].set_title(iName)

            for jMouse, jName in enumerate(mice):
                ax1[iMouse][jMouse].plot(pSigDict[jName], pSigDict[iName], '.')  # Coefficient flip intended
                corrCoef[iMouse, jMouse] = np.corrcoef(pSigDict[iName], pSigDict[jName])[0, 1]

        plt.savefig('pics/consistency_significant_activity_byphase_' + datatype + '_' + trialType + '.png')
        plt.close()

        fig2, ax2 = plt.subplots()
        ax2.imshow(corrCoef, vmin=0, vmax=1)
        imshow(fig2, ax2, corrCoef, title='Significance Correlation', haveColorBar=True, limits=[0, 1],
               xTicks=mice, yTicks=mice)

        plt.savefig('pics/consistency_significant_activity_byphase_corr_' + datatype + '_' + trialType + '.png')
        plt.close()
