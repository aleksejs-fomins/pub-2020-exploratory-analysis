import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon, combine_pvalues

from mesostat.utils.pandas_helper import pd_append_row, pd_pivot, outer_product_df, drop_rows_byquery, pd_is_one_row, pd_query
from mesostat.visualization.mpl_matrix import imshow
from mesostat.stat.connectomics import offdiag_1D
from mesostat.stat.testing.htests import classification_accuracy_weighted, rstest_twosided


def _get_test_metric(metricName):
    if metricName == 'accuracy':
        return classification_accuracy_weighted
    elif metricName == 'nlog_pval':
        return lambda x,y: -np.log10(rstest_twosided(x, y))[1]
    else:
        raise ValueError('Unexpected metric name', metricName)


def compute_mean_interval(dataDB, ds, trialTypesTrg, intervNames=None, skipExisting=False, exclQueryLst=None):
    dataName = 'mean'

    argSweepDict = {
        'mousename': sorted(list(dataDB.mice)),
        'intervName': intervNames if intervNames is not None else dataDB.get_interval_names(),
        'datatype': dataDB.get_data_types(),
        'trialType': trialTypesTrg
    }

    sweepDF = outer_product_df(argSweepDict)
    if exclQueryLst is not None:
        sweepDF = drop_rows_byquery(sweepDF, exclQueryLst)

    for idx, row in sweepDF.iterrows():
        print(list(row))

        for session in dataDB.get_sessions(row['mousename'], datatype=row['datatype']):
            attrsDict = {**{'session': session}, **dict(row)}

            dsDataLabels = ds.ping_data(dataName, attrsDict)
            if not skipExisting and len(dsDataLabels) > 0:
                dsuffix = dataName + '_' + '_'.join(attrsDict.values())
                print('Skipping existing', dsuffix)
            else:
                dataRSP = dataDB.get_neuro_data({'session': session}, datatype=row['datatype'],
                                                intervName=row['intervName'], trialType=row['trialType'])[0]

                dataRP = np.mean(dataRSP, axis=1)

                ds.delete_rows(dsDataLabels, verbose=False)
                ds.save_data(dataName, dataRP, attrsDict)


def plot_channel_significance_byaction(dataDB, ds, performance=None, metric='accuracy', minTrials=10, limits=(0.5, 1.0)):
    testFunc = _get_test_metric(metric)

    rows = ds.list_dsets_pd()
    rows['mousename'] = [dataDB.find_mouse_by_session(session) for session in rows['session']]

    for (datatype, intervName), rowsMouse in rows.groupby(['datatype', 'interv']):
        for mousename, rowsSession in rowsMouse.groupby(['mousename']):
            pSig = []
            for session, rowsTrial in rowsSession.groupby(['session']):
                if (performance is None) or dataDB.is_matching_performance(session, performance, mousename=mousename):
                    dataThis = []
                    for idx, row in rowsTrial.iterrows():
                        dataThis += [ds.get_data(row['dset'])]

                    nChannels = dataThis[0].shape[1]
                    nTrials1 = dataThis[0].shape[0]
                    nTrials2 = dataThis[1].shape[0]

                    if (nTrials1 < minTrials) or (nTrials2 < minTrials):
                        print(session, datatype, intervName, 'too few trials', nTrials1, nTrials2, ';; skipping')
                    else:
                        pSig += [[testFunc(dataThis[0][:, iCh], dataThis[1][:, iCh]) for iCh in range(nChannels)]]
            # pSigDict[mousename] = np.sum(pSig, axis=0)
            pSigAvg = np.mean(pSig, axis=0)

            fig, ax = dataDB.plot_area_values(pSigAvg, vmin=limits[0], vmax=limits[1], cmap='jet')
            plotSuffix = '_'.join([mousename, datatype, str(performance), intervName])
            fig.savefig('significance_'+plotSuffix+'_brainmap.png')
            plt.close()


def plot_consistency_significant_activity_byaction(dataDB, ds, minTrials=10, performance=None, dropChannels=None,
                                                   metric='accuracy', limits=None):
    testFunc = _get_test_metric(metric)

    rows = ds.list_dsets_pd()
    rows['mousename'] = [dataDB.find_mouse_by_session(session) for session in rows['session']]

    dfColumns = ['datatype', 'phase', 'consistency']
    dfConsistency = pd.DataFrame(columns=dfColumns)

    for (datatype, intervName), rowsMouse in rows.groupby(['datatype', 'intervName']):
        pSigDict = {}
        for mousename, rowsSession in rowsMouse.groupby(['mousename']):
            pSig = []
            for session, rowsTrial in rowsSession.groupby(['session']):
                if (performance is None) or dataDB.is_matching_performance(session, performance, mousename=mousename):
                    if len(rowsTrial) != 2:
                        print(mousename, session, rowsTrial)
                        raise ValueError('Expected exactly 2 rows')

                    dsetLabels = list(rowsTrial['dset'])
                    data1 = ds.get_data(dsetLabels[0])
                    data2 = ds.get_data(dsetLabels[1])
                    nTrials1 = data1.shape[0]
                    nTrials2 = data2.shape[1]

                    if (nTrials1 < minTrials) or (nTrials2 < minTrials):
                        print(session, datatype, intervName, 'too few trials', nTrials1, nTrials2, ';; skipping')
                    else:
                        nChannels = data1.shape[1]

                        if dropChannels is not None:
                            channelMask = np.ones(nChannels).astype(bool)
                            channelMask[dropChannels] = 0
                            data1 = data1[:, channelMask]
                            data2 = data2[:, channelMask]
                            nChannels = nChannels - len(dropChannels)

                        pvals = [testFunc(data1[:, iCh], data2[:, iCh]) for iCh in range(nChannels)]

                        # pSig += [(np.array(pvals) < 0.01).astype(int)]
                        pSig += [-np.log10(np.array(pvals))]
            # pSigDict[mousename] = np.sum(pSig, axis=0)
            pSigDict[mousename] = np.mean(pSig, axis=0)

        mice = sorted(pSigDict.keys())
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


def plot_consistency_significant_activity_byphase(dataDB, ds, intervals, minTrials=10, performance=None, dropChannels=None):
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
                    assert intervals[0] in list(rowsTrial['intervName'])
                    assert intervals[1] in list(rowsTrial['intervName'])
                    dsetLabel1 = pd_is_one_row(pd_query(rowsTrial, {'intervName': intervals[0]}))[1]['dset']
                    dsetLabel2 = pd_is_one_row(pd_query(rowsTrial, {'intervName': intervals[1]}))[1]['dset']
                    data1 = ds.get_data(dsetLabel1)
                    data2 = ds.get_data(dsetLabel2)
                    nTrials1 = data1.shape[0]
                    nTrials2 = data2.shape[1]

                    if (nTrials1 < minTrials) or (nTrials2 < minTrials):
                        print(session, datatype, trialType, 'too few trials', nTrials1, nTrials2, ';; skipping')
                    else:
                        nChannels = data1.shape[1]
                        if dropChannels is not None:
                            channelMask = np.ones(nChannels).astype(bool)
                            channelMask[dropChannels] = 0
                            data1 = data1[:, channelMask]
                            data2 = data2[:, channelMask]
                            nChannels = nChannels - len(dropChannels)

                        pvals = [wilcoxon(data1[:, iCh], data2[:, iCh], alternative='two-sided')[1]
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


def plot_activity_bychannel(dataDB, trialType, vmin=None, vmax=None, drop6=False):
    for datatype in ['bn_trial', 'bn_session']:
        mice = sorted(dataDB.mice)
        intervals = dataDB.get_interval_names()

        fig, ax = plt.subplots(nrows=len(mice), ncols=len(intervals), figsize=(4 * len(intervals), 4 * len(mice)))

        for iMouse, mousename in enumerate(mice):
            ax[iMouse][0].set_ylabel(mousename)
            for iInterv, intervName in enumerate(intervals):
                ax[0][iInterv].set_xlabel(intervName)
                if (~drop6) or (intervName != 'REW') or (mousename != 'mou_6'):
                    print(datatype, mousename, intervName)

                    dataLst = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype, intervName=intervName, trialType=trialType)
                    dataRSP = np.concatenate(dataLst, axis=0)
                    dataP = np.mean(dataRSP, axis=(0,1))

                    dataDB.plot_area_values(fig, ax[iMouse][iInterv], dataP, vmin=vmin, vmax=vmax, cmap='jet')

        plt.show()


def plot_classification_accuracy_bychannel(dataDB, drop6=False):
    for datatype in ['bn_trial', 'bn_session']:
        mice = sorted(dataDB.mice)
        intervals = dataDB.get_interval_names()

        fig, ax = plt.subplots(nrows=len(mice), ncols=len(intervals), figsize=(4 * len(intervals), 4 * len(mice)))

        for iMouse, mousename in enumerate(mice):
            ax[iMouse][0].set_ylabel(mousename)
            for iInterv, intervName in enumerate(intervals):
                ax[0][iInterv].set_xlabel(intervName)
                if (~drop6) or (intervName != 'REW') or (mousename != 'mou_6'):
                    print(datatype, mousename, intervName)

                    dataLst = [
                        dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype,
                                              intervName=intervName, trialType=trialType)
                        for trialType in ['Hit', 'FA', 'CR', 'Miss']
                    ]

                    # Stitch all sessions
                    dataLst = [np.concatenate(data, axis=0) for data in dataLst]

                    # Average out time
                    dataLst = [np.mean(data, axis=1) for data in dataLst]

                    # Split two textures
                    dataT1 = np.concatenate([dataLst[0], dataLst[1]])
                    dataT2 = np.concatenate([dataLst[2], dataLst[3]])

                    svcAcc = [classification_accuracy_weighted(x[:, None], y[:, None]) for x, y in zip(dataT1.T, dataT2.T)]

                    dataDB.plot_area_values(fig, ax[iMouse][iInterv], svcAcc, vmin=0.5, vmax=1.0, cmap='jet')

        plt.show()