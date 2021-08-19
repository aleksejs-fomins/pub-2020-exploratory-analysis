import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, wilcoxon, combine_pvalues
from IPython.display import display
from ipywidgets import IntProgress

from mesostat.utils.pandas_helper import pd_append_row, pd_pivot, outer_product_df, drop_rows_byquery, pd_is_one_row, pd_query
from mesostat.visualization.mpl_matrix import imshow

from mesostat.stat.connectomics import offdiag_1D
from mesostat.stat.testing.htests import classification_accuracy_weighted, rstest_twosided

from lib.analysis.metric_helper import get_data_list


def subset_dict(d1, d2):
    return d1.items() <= d2.items()


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


def significance_brainplot_mousephase_byaction(dataDB, ds, performance=None, #exclQueryLst=None,
                                               metric='accuracy', minTrials=10, limits=(0.5, 1.0), fontsize=20):
    testFunc = _get_test_metric(metric)

    rows = ds.list_dsets_pd()
    rows['mousename'] = [dataDB.find_mouse_by_session(session) for session in rows['session']]

    intervNames = dataDB.get_interval_names()
    mice = sorted(dataDB.mice)
    nInterv = len(intervNames)
    nMice = len(mice)

    for datatype, dfDataType in rows.groupby(['datatype']):
        fig, ax = plt.subplots(nrows=nMice, ncols=nInterv,
                               figsize=(4 * nInterv, 4 * nMice), tight_layout=True)

        for iInterv, intervName in enumerate(intervNames):
            ax[0][iInterv].set_title(intervName, fontsize=fontsize)
            for iMouse, mousename in enumerate(mice):
                ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

                pSig = []
                queryDict = {'mousename': mousename, 'intervName': intervName}

                # if (exclQueryLst is None) or all([not subset_dict(queryDict, d) for d in exclQueryLst]) :
                rowsSession = pd_query(dfDataType, queryDict)

                if len(rowsSession) > 0:
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
                    print(intervName, mousename, np.array(pSig).shape)

                    pSigAvg = np.mean(pSig, axis=0)

                    dataDB.plot_area_values(fig, ax[iMouse][iInterv], pSigAvg,
                                            vmin=limits[0], vmax=limits[1], cmap='jet',
                                            haveColorBar=iInterv==nInterv-1)

        plotSuffix = '_'.join([datatype, str(performance), metric])
        fig.savefig('significance_brainplot_'+plotSuffix+'.png')
        plt.close()


def activity_brainplot_mousephase(dataDB, trialTypes, vmin=None, vmax=None, drop6=False, fontsize=20):
    mice = sorted(dataDB.mice)
    intervals = dataDB.get_interval_names()

    for datatype in ['bn_trial', 'bn_session']:
        for trialType in trialTypes:
            fig, ax = plt.subplots(nrows=len(mice), ncols=len(intervals),
                                   figsize=(4 * len(intervals), 4 * len(mice)), tight_layout=True)

            for iMouse, mousename in enumerate(mice):
                ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
                for iInterv, intervName in enumerate(intervals):
                    ax[0][iInterv].set_title(intervName, fontsize=fontsize)
                    if (not drop6) or (intervName != 'REW') or (mousename != 'mou_6'):
                        print(datatype, mousename, intervName, drop6)
                        dataLst = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype, intervName=intervName,
                                                        trialType=trialType)
                        dataRSP = np.concatenate(dataLst, axis=0)
                        dataP = np.mean(dataRSP, axis=(0,1))

                        haveColorBar = iInterv == len(intervals)-1
                        dataDB.plot_area_values(fig, ax[iMouse][iInterv], dataP, vmin=vmin, vmax=vmax, cmap='jet',
                                                haveColorBar=haveColorBar)

            plt.savefig('activity_brainplot_mousephase_' + '_'.join([datatype, trialType]) + '.png')
            plt.close()


def activity_brainplot_mousetrialtype(dataDB, trialTypes, vmin=None, vmax=None, drop6=False, fontsize=20):
    mice = sorted(dataDB.mice)
    intervals = dataDB.get_interval_names()

    for datatype in ['bn_trial', 'bn_session']:
        for intervName in intervals:
            fig, ax = plt.subplots(nrows=len(mice), ncols=len(trialTypes),
                                   figsize=(4 * len(trialTypes), 4 * len(mice)), tight_layout=True)

            for iMouse, mousename in enumerate(mice):
                ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
                if (not drop6) or (intervName != 'REW') or (mousename != 'mou_6'):
                    for iTT, trialType in enumerate(trialTypes):
                        ax[0][iTT].set_title(trialType, fontsize=fontsize)
                        print(datatype, mousename, intervName, drop6)
                        dataLst = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype,
                                                        intervName=intervName, trialType=trialType)
                        dataRSP = np.concatenate(dataLst, axis=0)
                        dataP = np.mean(dataRSP, axis=(0,1))

                        haveColorBar = iTT == len(trialTypes)-1
                        dataDB.plot_area_values(fig, ax[iMouse][iTT], dataP, vmin=vmin, vmax=vmax, cmap='jet',
                                                haveColorBar=haveColorBar)

            plt.savefig('activity_brainplot_mousetrialtype_' + '_'.join([datatype, intervName]) + '.png')
            plt.close()


def activity_brainplot_mousephase_subpre(dataDB, trialTypes, vmin=None, vmax=None, drop6=False, fontsize=20):
    def _get_data_(dataDB, mousename, datatype, intervName, trialType):
        dataLst = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype, intervName=intervName,
                                        trialType=trialType)
        dataRSP = np.concatenate(dataLst, axis=0)
        return np.mean(dataRSP, axis=(0, 1))

    datatype = 'bn_session'
    mice = sorted(dataDB.mice)
    intervals = dataDB.get_interval_names()

    for trialType in trialTypes:
        fig, ax = plt.subplots(nrows=len(mice), ncols=len(intervals),
                               figsize=(4 * len(intervals), 4 * len(mice)), tight_layout=True)

        for iMouse, mousename in enumerate(mice):
            ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

            dataPPre = _get_data_(dataDB, mousename, datatype, 'PRE', trialType)

            for iInterv, intervName in enumerate(intervals):
                if intervName != 'PRE':
                    ax[0][iInterv].set_title(intervName, fontsize=fontsize)
                    if (not drop6) or (intervName != 'REW') or (mousename != 'mou_6'):
                        print(datatype, mousename, intervName, drop6)
                        dataP = _get_data_(dataDB, mousename, datatype, intervName, trialType)

                        dataPDelta = dataP - dataPPre

                        haveColorBar = iInterv == len(intervals)-1
                        dataDB.plot_area_values(fig, ax[iMouse][iInterv], dataPDelta, vmin=vmin, vmax=vmax, cmap='jet',
                                                haveColorBar=haveColorBar)

        plt.savefig('activity_brainplot_mousephase_subpre_' + '_'.join([datatype, trialType]) + '.png')
        plt.close()


def activity_brainplot_mousephase_submouse(dataDB, trialTypes, vmin=None, vmax=None, drop6=False, fontsize=20):
    def _get_data_(dataDB, mousename, datatype, intervName, trialType):
        dataLst = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype, intervName=intervName,
                                        trialType=trialType)
        dataRSP = np.concatenate(dataLst, axis=0)
        return np.mean(dataRSP, axis=(0, 1))

    datatype = 'bn_session'
    mice = sorted(dataDB.mice)
    intervals = dataDB.get_interval_names()

    for trialType in trialTypes:
        fig, ax = plt.subplots(nrows=len(mice), ncols=len(intervals),
                               figsize=(4 * len(intervals), 4 * len(mice)), tight_layout=True)

        for iInterv, intervName in enumerate(intervals):
            ax[0][iInterv].set_title(intervName, fontsize=fontsize)

            rezDict = {}

            for iMouse, mousename in enumerate(mice):
                print(datatype, mousename, intervName, drop6)
                ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

                if (not drop6) or (intervName != 'REW') or (mousename != 'mou_6'):
                    rezDict[mousename] = _get_data_(dataDB, mousename, datatype, intervName, trialType)

            dataPsub = np.mean(list(rezDict.values()), axis=0)
            for iMouse, mousename in enumerate(mice):
                if (not drop6) or (intervName != 'REW') or (mousename != 'mou_6'):
                    dataPDelta = rezDict[mousename] - dataPsub

                    haveColorBar = iInterv == len(intervals)-1
                    dataDB.plot_area_values(fig, ax[iMouse][iInterv], dataPDelta, vmin=vmin, vmax=vmax, cmap='jet',
                                            haveColorBar=haveColorBar)

        plt.savefig('activity_brainplot_mousephase_submouse_' + '_'.join([datatype, trialType]) + '.png')
        plt.close()


def activity_brainplot_mouse_2DF(dbDict, intervNameMap, intervOrdMap, trialTypes, vmin, vmax, drop6=False, fontsize=20):
    dbTmp = list(dbDict.values())[0]

    mice = sorted(dbTmp.mice)
    intervals = dbTmp.get_interval_names()

    for datatype in ['bn_trial', 'bn_session']:
        for trialType in trialTypes:
            for intervName in intervals:
                intervLabel = intervName if intervName not in intervNameMap else intervNameMap[intervName]

                fig, ax = plt.subplots(nrows=2, ncols=len(mice),
                                       figsize=(4 * len(mice), 4 * 2), tight_layout=True)

                for iDB, (dbName, dataDB) in enumerate(dbDict.items()):
                    ax[iDB][0].set_ylabel(dbName, fontsize=fontsize)
                    intervEffName = intervName if (dbName, intervName) not in intervOrdMap else intervOrdMap[(dbName, intervName)]

                    for iMouse, mousename in enumerate(mice):
                        ax[0][iMouse].set_title(mousename, fontsize=fontsize)
                        if (not drop6) or (intervEffName != 'REW') or (mousename != 'mou_6'):
                            print(datatype, intervEffName, dbName, mousename, drop6)
                            dataLst = dataDB.get_neuro_data({'mousename': mousename},
                                                            datatype=datatype, intervName=intervEffName,
                                                            trialType=trialType)
                            dataRSP = np.concatenate(dataLst, axis=0)
                            dataP = np.mean(dataRSP, axis=(0, 1))

                            haveColorBar = iMouse == len(mice)-1
                            dataDB.plot_area_values(fig, ax[iDB][iMouse], dataP, vmin=vmin, vmax=vmax, cmap='jet',
                                                    haveColorBar=haveColorBar)

                plt.savefig('activity_brainplot_mouse_2df_' + '_'.join([datatype, trialType, intervLabel]) + '.png')
                plt.close()


def classification_accuracy_brainplot_mousephase(dataDB, drop6=False, fontsize=20):
    for datatype in ['bn_trial', 'bn_session']:
        mice = sorted(dataDB.mice)
        intervals = dataDB.get_interval_names()

        fig, ax = plt.subplots(nrows=len(mice), ncols=len(intervals), figsize=(4 * len(intervals), 4 * len(mice)))

        for iMouse, mousename in enumerate(mice):
            ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
            for iInterv, intervName in enumerate(intervals):
                ax[0][iInterv].set_title(intervName, fontsize=fontsize)
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

        plt.savefig('classification_accuracy_brainplot_mousephase_' + '_'.join([datatype]) + '.png')
        plt.close()


################################
#  Consistency
################################


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


#############################
# Movies
#############################

def activity_brainplot_movie_mousetrialtype(dataDB, trialTypes, vmin=None, vmax=None, haveDelay=False, fontsize=20):
    mice = sorted(dataDB.mice)

    for datatype in ['bn_trial', 'bn_session']:
        # Store all preprocessed data first
        dataDict = {}
        for iMouse, mousename in enumerate(mice):
            for iTT, trialType in enumerate(trialTypes):
                print('Reading data, ', datatype, mousename, trialType)

                dataLst = get_data_list(dataDB, haveDelay, mousename, datatype=datatype, trialType=trialType)
                dataRSP = np.concatenate(dataLst, axis=0)
                dataSP = np.nanmean(dataRSP, axis=0)
                dataDict[(mousename, trialType)] = dataSP

        # Test that all datasets have the same duration
        shapeSet = set([v.shape for v in dataDict.values()])
        assert len(shapeSet) == 1
        nTimes = shapeSet.pop()[0]

        progBar = IntProgress(min=0, max=nTimes, description=datatype)
        display(progBar)  # display the bar
        for iTime in range(nTimes):
            fig, ax = plt.subplots(nrows=len(mice), ncols=len(trialTypes),
                                   figsize=(4 * len(trialTypes), 4 * len(mice)), tight_layout=True)

            for iMouse, mousename in enumerate(mice):
                ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
                for iTT, trialType in enumerate(trialTypes):
                    ax[0][iTT].set_title(trialType, fontsize=fontsize)
                    # print(datatype, mousename)

                    dataP = dataDict[(mousename, trialType)][iTime]

                    haveColorBar = iTT == len(trialTypes)-1
                    dataDB.plot_area_values(fig, ax[iMouse][iTT], dataP, vmin=vmin, vmax=vmax, cmap='jet',
                                            haveColorBar=haveColorBar)

            plt.savefig('activity_brainplot_mousetrialtype_' + '_'.join([datatype, str(iTime)]) + '.png')
            plt.close()
            progBar.value += 1
