# import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon #, mannwhitneyu, combine_pvalues
# from IPython.display import display
# from ipywidgets import IntProgress

from mesostat.utils.system import make_path
from mesostat.utils.pandas_helper import pd_append_row, pd_pivot, pd_is_one_row, pd_query, pd_first_row
from mesostat.visualization.mpl_matrix import imshow
# from mesostat.visualization.mpl_timescale_bar import add_timescale_bar

from mesostat.utils.matrix import offdiag_1D
from mesostat.stat.testing.htests import classification_accuracy_weighted

from lib.common.datawrapper import get_data_list
from lib.common.stat import test_metric_by_name
from lib.common.param_sweep import DataParameterSweep, param_vals_to_suffix, pd_row_to_kwargs
from lib.common.visualization import movie_mouse_trialtype


def subset_dict(d1, d2):
    return d1.items() <= d2.items()


def get_data_avg(dataDB, mousename, avgAxes=(0, 1), **kwargs):
    dataLst = dataDB.get_neuro_data({'mousename': mousename}, **kwargs)
    dataRSP = np.concatenate(dataLst, axis=0)
    return np.mean(dataRSP, axis=avgAxes)


def compute_mean_interval(dataDB, ds, trialTypeTrg, skipExisting=False, exclQueryLst=None, **kwargs):  # intervName=None,
    dataName = 'mean'

    dps = DataParameterSweep(dataDB, exclQueryLst, mousename='auto', trialType=trialTypeTrg, **kwargs)
    for idx, row in dps.sweepDF.iterrows():
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


def activity_brainplot_mouse(dataDB, xParamName, exclQueryLst=None, vmin=None, vmax=None, fontsize=20, dpi=200, **kwargs):
    assert xParamName in kwargs.keys(), 'Requires ' + xParamName
    dps = DataParameterSweep(dataDB, exclQueryLst, mousename='auto', **kwargs)
    nMice = dps.param_size('mousename')
    nXParam = dps.param_size(xParamName)

    for paramVals, dfTmp in dps.sweepDF.groupby(dps.invert_param(['mousename', xParamName])):
        plotSuffix = param_vals_to_suffix(paramVals)
        print(plotSuffix)

        fig, ax = plt.subplots(nrows=nMice, ncols=nXParam, figsize=(4*nXParam, 4*nMice), tight_layout=True)

        for mousename, dfMouse in dfTmp.groupby(['mousename']):
            iMouse = dps.param_index('mousename', mousename)

            ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
            for idx, row in dfMouse.iterrows():
                xParamVal = row[xParamName]
                iXParam = dps.param_index(xParamName, xParamVal)

                ax[0][iXParam].set_title(xParamVal, fontsize=fontsize)

                kwargsThis = pd_row_to_kwargs(row, parseNone=True, dropKeys=['mousename'])
                dataLst = dataDB.get_neuro_data({'mousename': mousename}, **kwargsThis)
                dataRSP = np.concatenate(dataLst, axis=0)
                dataP = np.mean(dataRSP, axis=(0,1))

                haveColorBar = iXParam == nXParam - 1
                dataDB.plot_area_values(fig, ax[iMouse][iXParam], dataP, vmin=vmin, vmax=vmax, cmap='jet',
                                        haveColorBar=haveColorBar)

        prefixPath = 'pics/activity/brainplot_mousephase/'
        make_path(prefixPath)
        plt.savefig(prefixPath + plotSuffix + '.png', dpi=dpi)
        plt.close()


def activity_brainplot_mousephase_subpre(dataDB, exclQueryLst=None, vmin=None, vmax=None, fontsize=20, dpi=200, **kwargs):
    assert 'intervName' in kwargs.keys(), 'Requires phases'
    dps = DataParameterSweep(dataDB, exclQueryLst, mousename='auto', datatype=['bn_session'], **kwargs)
    nMice = dps.param_size('mousename')
    nInterv = dps.param_size('intervName')

    for paramVals, dfTmp in dps.sweepDF.groupby(dps.invert_param(['mousename', 'intervName'])):
        plotSuffix = param_vals_to_suffix(paramVals)
        print(plotSuffix)

        fig, ax = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4 * nInterv, 4 * nMice), tight_layout=True)
        for mousename, dfMouse in dfTmp.groupby(['mousename']):
            iMouse = dps.param_index('mousename', mousename)

            ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

            kwargsPre = pd_row_to_kwargs(pd_first_row(dfMouse)[1], parseNone=True, dropKeys=['mousename', 'intervName'])
            kwargsPre['intervName'] = 'PRE'
            dataPPre = get_data_avg(dataDB, mousename, avgAxes=(0, 1), **kwargsPre)

            for idx, row in dfMouse.iterrows():
                intervName = row['intervName']
                iInterv = dps.param_index('intervName', intervName)

                if intervName != 'PRE':
                    ax[0][iInterv].set_title(intervName, fontsize=fontsize)

                    kwargsThis = pd_row_to_kwargs(row, parseNone=True, dropKeys=['mousename'])
                    dataP = get_data_avg(dataDB, mousename, avgAxes=(0, 1), **kwargsThis)

                    dataPDelta = dataP - dataPPre

                    haveColorBar = iInterv == nInterv - 1
                    dataDB.plot_area_values(fig, ax[iMouse][iInterv], dataPDelta, vmin=vmin, vmax=vmax, cmap='jet',
                                            haveColorBar=haveColorBar)

        prefixPath = 'pics/activity/brainplot_mousephase/subpre/'
        make_path(prefixPath)
        plt.savefig(prefixPath + plotSuffix + '.png', dpi=dpi)
        plt.close()


def activity_brainplot_mousephase_submouse(dataDB, exclQueryLst=None, vmin=None, vmax=None, fontsize=20, dpi=200, **kwargs):
    assert 'intervName' in kwargs.keys(), 'Requires phases'
    dps = DataParameterSweep(dataDB, exclQueryLst, mousename='auto', **kwargs)
    nMice = dps.param_size('mousename')
    nInterv = dps.param_size('intervName')

    for paramVals, dfTmp in dps.sweepDF.groupby(dps.invert_param(['mousename', 'intervName'])):
        plotSuffix = param_vals_to_suffix(paramVals)
        print(plotSuffix)

        fig, ax = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4*nInterv, 4*nMice), tight_layout=True)

        for intervName, dfInterv in dfTmp.groupby(['intervName']):
            iInterv = dps.param_index('intervName', intervName)

            ax[0][iInterv].set_title(intervName, fontsize=fontsize)

            rezDict = {}
            for idx, row in dfInterv.iterrows():
                mousename = row['mousename']
                kwargsThis = pd_row_to_kwargs(row, parseNone=True, dropKeys=['mousename'])
                rezDict[mousename] = get_data_avg(dataDB, mousename, avgAxes=(0, 1), **kwargsThis)

            dataPsub = np.mean(list(rezDict.values()), axis=0)
            for idx, row in dfInterv.iterrows():
                mousename = row['mousename']
                iMouse = dps.param_index('mousename', mousename)
                ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
                dataPDelta = rezDict[mousename] - dataPsub

                haveColorBar = iInterv == nInterv - 1
                dataDB.plot_area_values(fig, ax[iMouse][iInterv], dataPDelta, vmin=vmin, vmax=vmax, cmap='jet',
                                        haveColorBar=haveColorBar)

        prefixPath = 'pics/activity/brainplot_mousephase/submouse/'
        make_path(prefixPath)
        plt.savefig(prefixPath + plotSuffix + '.png', dpi=dpi)
        plt.close()


def activity_brainplot_mouse_2DF(dbDict, intervNameMap, intervOrdMap, trialTypes, vmin, vmax, drop6=False, dpi=200, fontsize=20):
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

                prefixPath = 'pics/activity/brainplot_mousephase/2df/'
                make_path(prefixPath)
                plt.savefig(prefixPath + '_'.join([datatype, trialType, intervLabel]) + '.png', dpi=dpi)
                plt.close()


def significance_brainplot_mousephase_byaction(dataDB, ds, performance=None, #exclQueryLst=None,
                                               metric='accuracy', minTrials=10, limits=(0.5, 1.0), fontsize=20):
    testFunc = test_metric_by_name(metric)

    rows = ds.list_dsets_pd()
    rows['mousename'] = [dataDB.find_mouse_by_session(session) for session in rows['session']]

    intervNames = dataDB.get_interval_names()
    mice = sorted(dataDB.mice)
    nInterv = len(intervNames)
    nMice = len(mice)

    for datatype, dfDataType in rows.groupby(['datatype']):
        fig, ax = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4 * nInterv, 4 * nMice), tight_layout=True)

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
        prefixPath = 'pics/significance/brainplot_mousephase/byaction/'
        make_path(prefixPath)
        fig.savefig(prefixPath + plotSuffix+'.png')
        plt.close()


def classification_accuracy_brainplot_mousephase(dataDB, exclQueryLst, fontsize=20, trialType='auto', **kwargs):
    assert 'intervName' in kwargs.keys(), 'Requires phases'
    dps = DataParameterSweep(dataDB, exclQueryLst, mousename='auto', **kwargs)
    nMice = dps.param_size('mousename')
    nInterv = dps.param_size('intervName')

    trialType = trialType if trialType != 'auto' else dataDB.get_trial_type_names()

    for paramVals, dfTmp in dps.sweepDF.groupby(dps.invert_param(['mousename', 'intervName'])):
        plotSuffix = param_vals_to_suffix(paramVals)
        print(plotSuffix)

        fig, ax = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4 * nInterv, 4 * nMice))

        for mousename, dfMouse in dfTmp.groupby(['mousename']):
            iMouse = dps.param_index('mousename', mousename)

            ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
            for idx, row in dfMouse.iterrows():
                intervName = row['intervName']
                iInterv = dps.param_index('intervName', intervName)
                ax[0][iInterv].set_title(intervName, fontsize=fontsize)

                kwargsThis = pd_row_to_kwargs(row, parseNone=True, dropKeys=['mousename'])
                dataRPLst = [get_data_avg(dataDB, mousename, avgAxes=1, **kwargsThis) for tt in trialType]

                # Split two textures
                dataT1 = np.concatenate([dataRPLst[0], dataRPLst[1]])
                dataT2 = np.concatenate([dataRPLst[2], dataRPLst[3]])

                svcAcc = [classification_accuracy_weighted(x[:, None], y[:, None]) for x, y in zip(dataT1.T, dataT2.T)]

                dataDB.plot_area_values(fig, ax[iMouse][iInterv], svcAcc, vmin=0.5, vmax=1.0, cmap='jet')

        prefixPath = 'pics/classification_accuracy/brainplot_mousephase/'
        make_path(prefixPath)
        plt.savefig(prefixPath + plotSuffix + '.png')
        plt.close()


################################
#  Consistency
################################


def plot_consistency_significant_activity_byaction(dataDB, ds, minTrials=10, performance=None, dropChannels=None,
                                                   metric='accuracy', limits=None):
    testFunc = test_metric_by_name(metric)

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

        prefixPath = 'pics/consistency/significant_activity/byaction/bymouse/'
        make_path(prefixPath)
        plt.savefig(prefixPath + plotSuffix + '.png')
        plt.close()

        fig2, ax2 = plt.subplots()
        ax2.imshow(corrCoef, vmin=0, vmax=1)
        imshow(fig2, ax2, corrCoef, title='Significance Correlation', haveColorBar=True, limits=[0, 1],
               xTicks=mice, yTicks=mice)

        prefixPath = 'pics/consistency/significant_activity/byaction/bymouse_corr/'
        make_path(prefixPath)
        plt.savefig(prefixPath + plotSuffix + '.png')
        plt.close()

        avgConsistency = np.round(np.mean(offdiag_1D(corrCoef)), 2)
        dfConsistency = pd_append_row(dfConsistency, [datatype, intervName, avgConsistency])

    fig, ax = plt.subplots()
    dfPivot = pd_pivot(dfConsistency, *dfColumns)
    sns.heatmap(data=dfPivot, ax=ax, annot=True, vmax=1, cmap='jet')

    prefixPath = 'pics/consistency/significant_activity/byaction/'
    make_path(prefixPath)
    fig.savefig(prefixPath + 'consistency_' + str(performance) + '.png')
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

        prefixPath = 'pics/consistency/significant_activity/byphase/bymouse/'
        make_path(prefixPath)
        plt.savefig(prefixPath + datatype + '_' + trialType + '.png')
        plt.close()

        fig2, ax2 = plt.subplots()
        ax2.imshow(corrCoef, vmin=0, vmax=1)
        imshow(fig2, ax2, corrCoef, title='Significance Correlation', haveColorBar=True, limits=[0, 1],
               xTicks=mice, yTicks=mice)

        prefixPath = 'pics/consistency/significant_activity/byphase/bymouse_corr/'
        make_path(prefixPath)
        plt.savefig(prefixPath + datatype + '_' + trialType + '.png')
        plt.close()

        avgConsistency = np.round(np.mean(offdiag_1D(corrCoef)), 2)
        dfConsistency = pd_append_row(dfConsistency, [datatype, trialType, avgConsistency])

    fig, ax = plt.subplots()
    dfPivot = pd_pivot(dfConsistency, *dfColumns)
    sns.heatmap(data=dfPivot, ax=ax, annot=True, vmax=1, cmap='jet')

    prefixPath = 'pics/consistency/significant_activity/byphase/'
    make_path(prefixPath)
    fig.savefig(prefixPath + 'consistency_' + str(performance) + '.png')
    plt.close()


#############################
# Movies
#############################

def calc_mean_sp(dataDB, mousename, calcKWArgs, haveDelay=False, **kwargsData):
    dataLst = get_data_list(dataDB, haveDelay, mousename, **kwargsData)
    dataRSP = np.concatenate(dataLst, axis=0)
    dataSP = np.nanmean(dataRSP, axis=0)
    return dataSP


def brainplot_mean(dataDB, fig, ax, data, **plotKWArgs):  # vmin, vmax
    if 'cmap' not in plotKWArgs.keys():
        plotKWArgs['cmap'] = 'jet'

    dataDB.plot_area_values(fig, ax, data, **plotKWArgs)


def activity_brainplot_movie_mousetrialtype(dataDB, dataKWArgs, plotKWArgs, exclQueryLst=None, haveDelay=False,
                                            fontsize=20, tTrgDelay=2.0, tTrgRew=2.0):

    prefixPath = 'pics/activity/brainplot_mousetrialtype/movies/'
    movie_mouse_trialtype(dataDB, dataKWArgs, {}, plotKWArgs, calc_mean_sp, brainplot_mean,
                          prefixPath=prefixPath, exclQueryLst=exclQueryLst, haveDelay=haveDelay, fontsize=fontsize,
                          tTrgDelay=tTrgDelay, tTrgRew=tTrgRew)
