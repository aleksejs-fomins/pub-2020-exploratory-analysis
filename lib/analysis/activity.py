import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon, combine_pvalues

from mesostat.utils.pandas_helper import pd_append_row, pd_pivot
from mesostat.visualization.mpl_matrix import imshow
from mesostat.stat.connectomics import offdiag_1D


def compute_mean_interval(dataDB, ds, trialTypesTrg, intervDict, skipExisting=False):
    dataName = 'mean'

    for iMouse, mousename in enumerate(sorted(dataDB.mice)):
        for datatype in dataDB.get_data_types():
            for trialType in trialTypesTrg:
                for intervName, interv in intervDict.items():
                    print(mousename, datatype, trialType, intervName)

                    for session in dataDB.get_sessions(mousename, datatype=datatype):
                        attrsDict = {
                            'datatype': datatype,
                            'session': session,
                            'trialType': trialType,
                            'interv': intervName
                        }

                        dsDataLabels = ds.ping_data(dataName, attrsDict)
                        if not skipExisting and len(dsDataLabels) > 0:
                            dsuffix = dataName + '_' + '_'.join(attrsDict.values())
                            print('Skipping existing', dsuffix)
                        else:
                            dataRSP = dataDB.get_neuro_data({'session': session}, datatype=datatype,
                                                            cropTime=interv, trialType=trialType)[0]

                            dataRP = np.mean(dataRSP, axis=1)

                            ds.delete_rows(dsDataLabels, verbose=False)
                            ds.save_data(dataName, dataRP, attrsDict)


def plot_consistency_significant_activity_byaction(dataDB, ds, minTrials=10, performance=None, dropChannels=None):
    rows = ds.list_dsets_pd()
    rows['mousename'] = [dataDB.find_mouse_by_session(session) for session in rows['session']]

    dfColumns = ['datatype', 'phase', 'consistency']
    dfConsistency = pd.DataFrame(columns=dfColumns)

    for (datatype, intervName), rowsMouse in rows.groupby(['datatype', 'interv']):
        pSigDict = {}
        for mousename, rowsSession in rowsMouse.groupby(['mousename']):
            pSig = []
            for session, rowsTrial in rowsSession.groupby(['session']):
                if (performance is None) or dataDB.is_matching_performance(session, performance, mousename=mousename):
                    dataThis = []
                    for idx, row in rowsTrial.iterrows():
                        dataThis += [ds.get_data(row['dset'])]

                    nTrials1 = dataThis[0].shape[0]
                    nTrials2 = dataThis[1].shape[0]

                    if (nTrials1 < minTrials) or (nTrials2 < minTrials):
                        print(session, datatype, intervName, 'too few trials', nTrials1, nTrials2, ';; skipping')
                    else:
                        nChannels = dataThis[0].shape[1]

                        if dropChannels is not None:
                            channelMask = np.ones(nChannels).astype(bool)
                            channelMask[dropChannels] = 0
                            dataThis[0] = dataThis[0][:, channelMask]
                            dataThis[1] = dataThis[1][:, channelMask]
                            nChannels = nChannels - len(dropChannels)

                        pvals = [mannwhitneyu(dataThis[0][:, iCh], dataThis[1][:, iCh], alternative='two-sided')[1]
                                 for iCh in range(nChannels)]
                        # pSig += [(np.array(pvals) < 0.01).astype(int)]
                        pSig += [-np.log10(np.array(pvals))]
            # pSigDict[mousename] = np.sum(pSig, axis=0)
            pSigDict[mousename] = np.mean(pSig, axis=0)

        mice = sorted(dataDB.mice)
        nMice = len(mice)
        corrCoef = np.zeros((nMice, nMice))
        for iMouse, iName in enumerate(mice):
            for jMouse, jName in enumerate(mice):
                corrCoef[iMouse, jMouse] = np.corrcoef(pSigDict[iName], pSigDict[jName])[0, 1]

        plotSuffix = '_'.join([datatype, str(performance), intervName])

        sns.pairplot(data=pd.DataFrame(pSigDict), vars=mice)
        plt.savefig('pics/consistency_significant_activity_bymouse_' + plotSuffix + '.png')
        plt.close()

        fig2, ax2 = plt.subplots()
        ax2.imshow(corrCoef, vmin=0, vmax=1)
        imshow(fig2, ax2, corrCoef, title='Significance Correlation', haveColorBar=True, limits=[0, 1],
               xTicks=mice, yTicks=mice)

        plt.savefig('pics/consistency_significant_activity_bymouse_corr_' + plotSuffix + '.png')
        plt.close()

        avgConsistency = np.round(np.mean(offdiag_1D(corrCoef)), 2)
        dfConsistency = pd_append_row(dfConsistency, [datatype, intervName, avgConsistency])

    fig, ax = plt.subplots()
    dfPivot = pd_pivot(dfConsistency, *dfColumns)
    sns.heatmap(data=dfPivot, ax=ax, annot=True, vmax=1, cmap='jet')
    fig.savefig('consistency_significant_activity_action_bymouse_metric_' + str(performance) + '.png')
    plt.close()


def plot_consistency_significant_activity_byphase(dataDB, ds, minTrials=10, performance=None, dropChannels=None):
    rows = ds.list_dsets_pd()
    rows['mousename'] = [dataDB.find_mouse_by_session(session) for session in rows['session']]

    dfColumns = ['datatype', 'trialType', 'consistency']
    dfConsistency = pd.DataFrame(columns=dfColumns)

    for (datatype, trialType), rowsMouse in rows.groupby(['datatype', 'trialType']):
        pSigDict = {}
        for mousename, rowsSession in rowsMouse.groupby(['mousename']):
            pSig = []
            for session, rowsTrial in rowsSession.groupby(['session']):
                if (performance is None) or dataDB.is_matching_performance(session, performance, mousename=mousename):
                    dataThis = []
                    for idx, row in rowsTrial.iterrows():
                        if row['interv'] != 'PRE':
                            dataThis += [ds.get_data(row['dset'])]

                    nTrials1 = dataThis[0].shape[0]
                    nTrials2 = dataThis[1].shape[0]
                    if (nTrials1 < minTrials) or (nTrials2 < minTrials):
                        print(session, datatype, trialType, 'too few trials', nTrials1, nTrials2, ';; skipping')
                    else:
                        nChannels = dataThis[0].shape[1]
                        if dropChannels is not None:
                            channelMask = np.ones(nChannels).astype(bool)
                            channelMask[dropChannels] = 0
                            dataThis[0] = dataThis[0][:, channelMask]
                            dataThis[1] = dataThis[1][:, channelMask]
                            nChannels = nChannels - len(dropChannels)

                        pvals = [wilcoxon(dataThis[0][:, iCh], dataThis[1][:, iCh], alternative='two-sided')[1]
                                 for iCh in range(nChannels)]
                        # pSig += [(np.array(pvals) < 0.01).astype(int)]
                        pSig += [-np.log10(np.array(pvals))]
            # pSigDict[mousename] = np.sum(pSig, axis=0)
            pSigDict[mousename] = np.mean(pSig, axis=0)

        mice = sorted(dataDB.mice)
        nMice = len(mice)
        corrCoef = np.zeros((nMice, nMice))
        for iMouse, iName in enumerate(mice):
            for jMouse, jName in enumerate(mice):
                corrCoef[iMouse, jMouse] = np.corrcoef(pSigDict[iName], pSigDict[jName])[0, 1]

        sns.pairplot(data=pd.DataFrame(pSigDict), vars=mice)
        plt.savefig('pics/consistency_significant_activity_byphase_' + datatype + '_' + trialType + '.png')
        plt.close()

        fig2, ax2 = plt.subplots()
        ax2.imshow(corrCoef, vmin=0, vmax=1)
        imshow(fig2, ax2, corrCoef, title='Significance Correlation', haveColorBar=True, limits=[0, 1],
               xTicks=mice, yTicks=mice)

        plt.savefig('pics/consistency_significant_activity_byphase_corr_' + datatype + '_' + trialType + '.png')
        plt.close()

        avgConsistency = np.round(np.mean(offdiag_1D(corrCoef)), 2)
        dfConsistency = pd_append_row(dfConsistency, [datatype, trialType, avgConsistency])

    fig, ax = plt.subplots()
    dfPivot = pd_pivot(dfConsistency, *dfColumns)
    sns.heatmap(data=dfPivot, ax=ax, annot=True, vmax=1, cmap='jet')
    fig.savefig('consistency_significant_activity_phase_bymouse_metric_' + str(performance) + '.png')
    plt.close()
