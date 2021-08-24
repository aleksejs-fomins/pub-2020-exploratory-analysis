import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from IPython.display import display
from ipywidgets import IntProgress

from mesostat.utils.signals.filter import zscore, drop_PCA, drop_PCA_3D
import mesostat.stat.consistency.pca as pca
from mesostat.utils.matrix import offdiag_1D, tril_1D
from mesostat.utils.pandas_helper import pd_append_row, pd_pivot #, outer_product_df, drop_rows_byquery, pd_is_one_row, pd_query
from mesostat.visualization.mpl_matrix import imshow
from mesostat.visualization.mpl_colorbar import imshow_add_color_bar
from mesostat.stat.clustering import cluster_dist_matrix_max, cluster_plot

from lib.common.datawrapper import get_data_list
from lib.common.visualization import cluster_brain_plot
from lib.common.param_sweep import DataParameterSweep, param_vals_to_suffix, pd_row_to_kwargs


###############################
# Correlation Plots
###############################


def subset_dict(d1, d2):
    return d1.items() <= d2.items()


# Plot channels by their average correlation
def _plot_corr_1D(fig, ax, channelLabels, rez2D, thrMono):
    np.fill_diagonal(rez2D, np.nan)
    rez1D = np.nanmean(rez2D, axis=0)
    idxsPlot = rez1D < thrMono

    if np.sum(idxsPlot) == 0:
        print('--No channels with avgcorr < ', thrMono)
    else:
        vals = rez1D[idxsPlot]
        chLabels = np.array(channelLabels)[idxsPlot]
        idxsValsSort = np.argsort(vals)
        vals = vals[idxsValsSort]
        chLabels = chLabels[idxsValsSort]

        g = sns.barplot(ax=ax, x=chLabels, y=vals)
        g.set_xticklabels(chLabels, rotation=90)
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color='pink', linestyle='--')


def calc_corr_mouse(dataDB, mc, mousename, nDropPCA=1, dropChannels=None, strategy='mean', estimator='corr', **kwargs):
    # NOTE: zscore channels for each session to avoid session-wise effects
    dataRSPLst = dataDB.get_neuro_data({'mousename': mousename}, zscoreDim='rs', **kwargs)
    dataRSP = np.concatenate(dataRSPLst, axis=0)
    channelLabels = np.array(dataDB.get_channel_labels())

    if strategy == 'mean':
        dataRP = np.mean(dataRSP, axis=1)
        if nDropPCA is not None:
            dataRP = drop_PCA(dataRP, nDropPCA)

        if dropChannels is not None:
            nChannels = dataRP.shape[1]
            channelMask = np.ones(nChannels).astype(bool)
            channelMask[dropChannels] = 0
            dataRP = dataRP[:, channelMask]
            channelLabels = channelLabels[channelMask]

        mc.set_data(dataRP, 'rp')
        metricSettings = {'havePVal': False, 'estimator': estimator}
        rez2D = mc.metric3D('corr', '', metricSettings=metricSettings)
        return channelLabels, rez2D
    elif strategy == 'sweep':
        if nDropPCA is not None:
            dataRSP = drop_PCA_3D(dataRSP, nDropPCA)

        if dropChannels is not None:
            nChannels = dataRSP.shape[2]
            channelMask = np.ones(nChannels).astype(bool)
            channelMask[dropChannels] = 0
            dataRSP = dataRSP[:, :, channelMask]
            channelLabels = channelLabels[channelMask]

        mc.set_data(dataRSP, 'rsp')
        metricSettings = {'havePVal': False, 'estimator': estimator}
        rezS2D = mc.metric3D('corr', 's', metricSettings=metricSettings)
        rez2D = np.mean(rezS2D, axis=0)

        return channelLabels, rez2D
    else:
        raise ValueError('Unexpected strategy', strategy)


def plot_corr_mouse(dataDB, mc, estimator, xParamName, nDropPCA=None, dropChannels=None, haveBrain=False, haveMono=True,
                    corrStrategy='mean', exclQueryLst=None, thrMono=0.4, clusterParam=-10, fontsize=20, **kwargs):

    assert xParamName in ['intervName', 'trialType'], 'Unexpected parameter'
    assert xParamName in kwargs.keys(), 'Requires ' + xParamName
    dps = DataParameterSweep(dataDB, exclQueryLst, mousename='auto', **kwargs)
    nMice = dps.param_size('mousename')
    nXParam = dps.param_size(xParamName)

    for paramVals, dfTmp in dps.sweepDF.groupby(dps.invert_param(['mousename', xParamName])):
        plotSuffix = param_vals_to_suffix(paramVals)
        print(plotSuffix)

        figCorr, axCorr = plt.subplots(nrows=nMice, ncols=nXParam, figsize=(4*nXParam, 4*nMice), tight_layout=True)
        figClust, axClust = plt.subplots(nrows=nMice, ncols=nXParam, figsize=(4 * nXParam, 4 * nMice), tight_layout=True)
        if haveBrain:
            figBrain, axBrain = plt.subplots(nrows=nMice, ncols=nXParam, figsize=(4 * nXParam, 4 * nMice), tight_layout=True)
        if haveMono:
            figMono, axMono = plt.subplots(nrows=nMice, ncols=nXParam, figsize=(4 * nXParam, 4 * nMice), tight_layout=True)

        for mousename, dfMouse in dfTmp.groupby(['mousename']):
            iMouse = dps.param_index('mousename', mousename)

            axCorr[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
            axClust[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

            if haveBrain:
                axBrain[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
            if haveMono:
                axMono[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

            for idx, row in dfMouse.iterrows():
                xParamVal = row[xParamName]
                iXParam = dps.param_index(xParamName, xParamVal)

                axCorr[0][iXParam].set_title(xParamVal, fontsize=fontsize)
                axClust[0][iXParam].set_title(xParamVal, fontsize=fontsize)

                if haveBrain:
                    axBrain[0][iXParam].set_title(xParamVal, fontsize=fontsize)
                if haveMono:
                    axMono[0][iXParam].set_title(xParamVal, fontsize=fontsize)

                kwargsThis = pd_row_to_kwargs(row, parseNone=True, dropKeys=['mousename'])
                channelLabels, rez2D = calc_corr_mouse(dataDB, mc, mousename, strategy=corrStrategy,
                                                       nDropPCA=nDropPCA, dropChannels=dropChannels,
                                                       estimator=estimator, **kwargsThis)

                haveColorBar = iXParam == nXParam - 1

                # Plot correlations
                imshow(figCorr, axCorr[iMouse][iXParam], rez2D, title='corr', limits=[-1,1], cmap='jet',
                       haveColorBar=haveColorBar)

                # Plot clustering
                clusters = cluster_dist_matrix_max(rez2D, clusterParam, method='Affinity')
                cluster_plot(figClust, axClust[iMouse][iXParam], rez2D, clusters, channelLabels, limits=[-1,1],
                             cmap='jet', haveColorBar=haveColorBar)

                if haveBrain:
                    cluster_brain_plot(figBrain, axBrain[iMouse][iXParam], dataDB, clusters,
                                        dropChannels=dropChannels, haveColorBar=haveColorBar)

                if haveMono:
                    _plot_corr_1D(figMono, axMono[iMouse][iXParam], channelLabels, rez2D, thrMono)

        # Save image
        plotPrefix = 'corr_mouse' + xParamName + '_dropPCA_' + str(nDropPCA)

        figCorr.savefig(plotPrefix + '_' + plotSuffix + '.png')
        plt.close(figCorr)
        figClust.savefig(plotPrefix + '_clust_' + plotSuffix + '.png')
        plt.close(figClust)
        if haveBrain:
            figBrain.savefig(plotPrefix + '_clust_brainplot_' + plotSuffix + '.png')
            plt.close(figBrain)
        if haveMono:
            figMono.savefig(plotPrefix + '_1D_' + plotSuffix + '.png')
            plt.close(figMono)


def plot_corr_mousephase_subpre(dataDB, mc, estimator, nDropPCA=None, dropChannels=None, exclQueryLst=None, fontsize=20,
                                corrStrategy='mean', **kwargs):

    assert 'intervName' in kwargs.keys(), 'Requires phases'
    dps = DataParameterSweep(dataDB, exclQueryLst, mousename='auto', datatype=['bn_session'], **kwargs)
    nMice = dps.param_size('mousename')
    nInterv = dps.param_size('intervName')

    for paramVals, dfTmp in dps.sweepDF.groupby(dps.invert_param(['mousename', 'intervName'])):
        plotSuffix = param_vals_to_suffix(paramVals)
        print(plotSuffix)

        figCorr, axCorr = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4*nInterv, 4*nMice), tight_layout=True)

        for mousename, dfMouse in dfTmp.groupby(['mousename']):
            iMouse = dps.param_index('mousename', mousename)

            axCorr[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
            rezDict = {}

            for idx, row in dfMouse.iterrows():
                intervName = row['intervName']

                kwargsThis = pd_row_to_kwargs(row, parseNone=True, dropKeys=['mousename'])
                channelLabels, rez2D = calc_corr_mouse(dataDB, mc, mousename, strategy=corrStrategy,
                                                       nDropPCA=nDropPCA, dropChannels=dropChannels,
                                                       estimator=estimator, **kwargsThis)

                rezDict[intervName] = rez2D

            # Plot correlations
            for intervName, rezInterv in rezDict.items():
                iInterv = dps.param_index('intervName', intervName)

                axCorr[0][iInterv].set_title(intervName, fontsize=fontsize)
                if (intervName in rezDict.keys()) and (intervName != 'PRE'):
                    haveColorBar = iInterv == nInterv - 1
                    imshow(figCorr, axCorr[iMouse][iInterv], rezInterv - rezDict['PRE'],
                           title='corr', haveColorBar=haveColorBar, limits=[-1, 1], cmap='RdBu_r')

        # Save image
        figCorr.savefig('corr_subpre_dropPCA_' + str(nDropPCA) + '_' + plotSuffix + '.png')
        plt.close()


def plot_corr_mousephase_submouse(dataDB, mc, estimator, nDropPCA=None, dropChannels=None, exclQueryLst=None,
                                  corrStrategy='mean', fontsize=20, **kwargs):

    assert 'intervName' in kwargs.keys(), 'Requires phases'
    dps = DataParameterSweep(dataDB, exclQueryLst, mousename='auto', **kwargs)
    nMice = dps.param_size('mousename')
    nInterv = dps.param_size('intervName')

    for paramVals, dfTmp in dps.sweepDF.groupby(dps.invert_param(['mousename', 'intervName'])):
        plotSuffix = param_vals_to_suffix(paramVals)
        print(plotSuffix)

        figCorr, axCorr = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4*nInterv, 4*nMice))

        for intervName, dfInterv in dfTmp.groupby(['intervName']):
            iInterv = dps.param_index('intervName', intervName)

            axCorr[0][iInterv].set_title(intervName, fontsize=fontsize)

            rezDict = {}
            for idx, row in dfInterv.iterrows():
                mousename = row['mousename']

                kwargsThis = pd_row_to_kwargs(row, parseNone=True, dropKeys=['mousename'])
                channelLabels, rez2D = calc_corr_mouse(dataDB, mc, mousename, strategy=corrStrategy,
                                                       nDropPCA=nDropPCA, dropChannels=dropChannels,
                                                       estimator=estimator, **kwargsThis)

                rezDict[mousename] = rez2D

            # Plot correlations
            rezMean = np.mean(list(rezDict.values()), axis=0)

            for mousename, rezMouse in rezDict.items():
                iMouse = dps.param_index('mousename', mousename)
                axCorr[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

                if mousename in rezDict.keys():
                    haveColorBar = iInterv == nInterv - 1
                    imshow(figCorr, axCorr[iMouse][iInterv], rezMouse - rezMean,
                           title='corr', haveColorBar=haveColorBar, limits=[-1,1], cmap='RdBu_r')

        # Save image
        figCorr.savefig('corr_submouse__dropPCA_' + str(nDropPCA) + '_' + plotSuffix + '.png')
        plt.close()


def plot_corr_mouse_2DF(dfDict, mc, estimator, intervNameMap, intervOrdMap, corrStrategy='mean',
                        nDropPCA=None, dropChannels=None, exclQueryLst=None):
    dataDBTmp = list(dfDict.values())[0]

    mice = sorted(dataDBTmp.mice)
    nMice = len(mice)
    intervNames = dataDBTmp.get_interval_names()
    trialTypes = dataDBTmp.get_trial_type_names()

    for trialType in trialTypes:
        for intervName in intervNames:
            intervLabel = intervName if intervName not in intervNameMap else intervNameMap[intervName]
            plotSuffix = trialType + '_' + intervLabel
            print(plotSuffix)

            fig, ax = plt.subplots(nrows=2, ncols=nMice, figsize=(4*nMice, 4*2), tight_layout=True)

            for iDB, (dbName, dataDB) in enumerate(dfDict.items()):
                ax[iDB][0].set_ylabel(dbName)

                intervEffName = intervName if (dbName, intervName) not in intervOrdMap else intervOrdMap[
                    (dbName, intervName)]

                for iMouse, mousename in enumerate(mice):
                    ax[0][iMouse].set_title(mousename)

                    kwargs = {'mousename': mousename, 'intervName': intervEffName,
                              'trialType': trialType, 'datatype': 'bn_session'}

                    if np.all([not subset_dict(excl, kwargs) for excl in exclQueryLst]):
                        del kwargs['mousename']
                        kwargs = {k: v if v!='None' else None for k, v in kwargs.items()}

                        channelLabels, rez2D = calc_corr_mouse(dataDB, mc, mousename, strategy=corrStrategy,
                                                               nDropPCA=nDropPCA, dropChannels=dropChannels,
                                                               estimator=estimator, **kwargs)

                        imshow(fig, ax[iDB][iMouse], rez2D,
                               limits=[-1, 1], cmap='jet', haveColorBar=iMouse == nMice-1)

            # Save image
            plt.savefig('corr_bystim_dropPCA_' + str(nDropPCA) + '_bn_session_' + plotSuffix + '.png')
            plt.close()


###############################
# Diff
###############################

def plot_corr_consistency_l1_mouse(dataDB, nDropPCA=None, dropChannels=None, exclQueryLst=None, **kwargs): # performances=None, trialTypes=None,

    assert 'intervName' in kwargs.keys(), 'Requires phases'
    dps = DataParameterSweep(dataDB, exclQueryLst, mousename='auto', intervName='auto', datatype='auto', **kwargs)
    mice = sorted(dataDB.mice)
    nMice = len(mice)

    for paramExtraVals, dfTmp in dps.sweepDF.groupby(dps.invert_param(['mousename', 'datatype', 'intervName'])):
        plotExtraSuffix = param_vals_to_suffix(paramExtraVals)

        dfColumns = ['datatype', 'phase', 'consistency']
        dfConsistency = pd.DataFrame(columns=dfColumns)

        for paramVals, dfMouse in dfTmp.groupby(dps.invert_param(['mousename'])):
            plotSuffix = param_vals_to_suffix(paramVals)
            print(plotSuffix)

            corrLst = []
            for idx, row in dfMouse.iterrows():
                plotSuffix = '_'.join([str(s) for s in row.values])
                print(plotSuffix)

                kwargsThis = pd_row_to_kwargs(row, parseNone=True, dropKeys=['mousename'])
                dataRSPLst = dataDB.get_neuro_data({'mousename' : row['mousename']},    # NOTE: zscore channels for each session to avoid session-wise effects
                                                    zscoreDim='rs',
                                                    **kwargsThis)

                dataRSP = np.concatenate(dataRSPLst, axis=0)
                dataRP = np.mean(dataRSP, axis=1)
                # dataRP = zscore(dataRP, axis=0)

                if dropChannels is not None:
                    channelMask = np.ones(dataRP.shape[1]).astype(bool)
                    channelMask[dropChannels] = 0
                    dataRP = dataRP[:, channelMask]

                if nDropPCA is not None:
                    dataRP = drop_PCA(dataRP, nDropPCA)

                corrLst += [tril_1D(np.corrcoef(dataRP.T))]

            # fig, ax = plt.subplots(nrows=nMice, ncols=nMice, figsize=(4 * nMice, 4 * nMice))

            pairDict = {}
            rezMat = np.zeros((nMice, nMice))
            for iMouse, mousename in enumerate(mice):
                # ax[iMouse][0].set_ylabel(mice[iMouse])
                # ax[-1][iMouse].set_xlabel(mice[iMouse])
                pairDict[mousename] = corrLst[iMouse]

                for jMouse in range(nMice):
                    # rezMat[iMouse][jMouse] = 1 - rmae(corrLst[iMouse], corrLst[jMouse])
                    rezMat[iMouse][jMouse] = np.corrcoef(corrLst[iMouse], corrLst[jMouse])[0, 1]

                    # cci = offdiag_1D(corrLst[iMouse])
                    # ccj = offdiag_1D(corrLst[jMouse])
                    # ax[iMouse][jMouse].plot(cci, ccj, '.')

            pPlot = sns.pairplot(data=pd.DataFrame(pairDict), vars=mice, kind='kde')

            plt.savefig('corr_consistency_bymouse_scatter_' + plotSuffix + '.png')
            plt.close()

            fig, ax = plt.subplots()
            imshow(fig, ax, rezMat, haveColorBar=True, limits=[0,1], xTicks=mice, yTicks=mice, cmap='jet')
            plt.savefig('corr_consistency_bymouse_metric_' + plotSuffix + '.png')
            plt.close()

            avgConsistency = np.round(np.mean(offdiag_1D(rezMat)), 2)
            dfConsistency = pd_append_row(dfConsistency, [row['datatype'], row['intervName'], avgConsistency])

        fig, ax = plt.subplots()
        dfPivot = pd_pivot(dfConsistency, *dfColumns)
        sns.heatmap(data=dfPivot, ax=ax, annot=True, vmin=0, vmax=1, cmap='jet')
        fig.savefig('consistency_coactivity_metric_dropPCA_' + str(nDropPCA)+ '_' + plotExtraSuffix+'.png')
        plt.close()


def plot_corr_consistency_l1_trialtype(dataDB, nDropPCA=None, dropChannels=None, performance=None, trialTypes=None,
                                       datatype=None):
    mice = sorted(dataDB.mice)
    if trialTypes is None:
        trialTypes = dataDB.get_data_types()
    nTT = len(trialTypes)

    dfColumns = ['mousename', 'phase', 'consistency']

    dfConsistency = pd.DataFrame(columns=dfColumns)

    for iMouse, mousename in enumerate(mice):
        for intervName in dataDB.get_interval_names():
            fnameSuffix = '_'.join([datatype, mousename, intervName, str(performance)])
            print(fnameSuffix)

            corrLst = []
            for trialType in trialTypes:
                kwargs = {'datatype' : datatype, 'intervName' : intervName, 'trialType' : trialType}
                if performance is not None:
                    kwargs['performance'] = performance

                dataRSPLst = dataDB.get_neuro_data({'mousename': mousename}, zscoreDim='rs', **kwargs)

                dataRSP = np.concatenate(dataRSPLst, axis=0)
                dataRP = np.mean(dataRSP, axis=1)
                # dataRP = zscore(dataRP, axis=0)

                if dropChannels is not None:
                    channelMask = np.ones(dataRP.shape[1]).astype(bool)
                    channelMask[dropChannels] = 0
                    dataRP = dataRP[:, channelMask]

                if nDropPCA is not None:
                    dataRP = drop_PCA(dataRP, nDropPCA)

                corrLst += [tril_1D(np.corrcoef(dataRP.T))]

            # fig, ax = plt.subplots(nrows=nMice, ncols=nMice, figsize=(4 * nMice, 4 * nMice))

            pairDict = {}
            rezMat = np.zeros((nTT, nTT))
            for idxTTi, iTT in enumerate(trialTypes):
                # ax[iMouse][0].set_ylabel(mice[iMouse])
                # ax[-1][iMouse].set_xlabel(mice[iMouse])
                pairDict[iTT] = corrLst[idxTTi]

                for idxTTj, jTT in enumerate(trialTypes):
                    # rezMat[iMouse][jMouse] = 1 - rmae(corrLst[iMouse], corrLst[jMouse])
                    rezMat[idxTTi][idxTTj] = np.corrcoef(corrLst[idxTTi], corrLst[idxTTj])[0, 1]

                    # cci = offdiag_1D(corrLst[iMouse])
                    # ccj = offdiag_1D(corrLst[jMouse])
                    # ax[iMouse][jMouse].plot(cci, ccj, '.')

            pPlot = sns.pairplot(data=pd.DataFrame(pairDict), vars=trialTypes, kind='kde')

            plt.savefig('corr_consistency_bymouse_scatter_' + fnameSuffix + '.png')
            plt.close()

            fig, ax = plt.subplots()
            imshow(fig, ax, rezMat, haveColorBar=True, limits=[0,1], xTicks=trialTypes, yTicks=trialTypes, cmap='jet')
            plt.savefig('corr_consistency_bymouse_metric_' + fnameSuffix + '.png')
            plt.close()

            avgConsistency = np.round(np.mean(offdiag_1D(rezMat)), 2)
            dfConsistency = pd_append_row(dfConsistency, [mousename, intervName, avgConsistency])

    fig, ax = plt.subplots()
    dfPivot = pd_pivot(dfConsistency, *dfColumns)
    sns.heatmap(data=dfPivot, ax=ax, annot=True, vmin=0, vmax=1, cmap='jet')
    fig.savefig('consistency_coactivity_metric_dropPCA_' + str(nDropPCA) + '_' + datatype + '_' + str(performance) + '.png')
    plt.close()


def plot_corr_consistency_l1_phase(dataDB, nDropPCA=None, dropChannels=None, performance=None, datatype=None):
    mice = sorted(dataDB.mice)
    phases =  dataDB.get_interval_names()
    nPhases = len(phases)

    dfColumns = ['mousename', 'trialtype', 'consistency']
    dfConsistency = pd.DataFrame(columns=dfColumns)

    for iMouse, mousename in enumerate(mice):
        for trialType in dataDB.get_trial_type_names():
            fnameSuffix = '_'.join([datatype, mousename, trialType, str(performance)])
            print(fnameSuffix)

            corrLst = []
            for intervName in phases:
                kwargs = {'datatype' : datatype, 'intervName' : intervName, 'trialType' : trialType}
                if performance is not None:
                    kwargs['performance'] = performance

                dataRSPLst = dataDB.get_neuro_data({'mousename': mousename}, zscoreDim='rs', **kwargs)

                dataRSP = np.concatenate(dataRSPLst, axis=0)
                dataRP = np.mean(dataRSP, axis=1)
                # dataRP = zscore(dataRP, axis=0)

                if dropChannels is not None:
                    channelMask = np.ones(dataRP.shape[1]).astype(bool)
                    channelMask[dropChannels] = 0
                    dataRP = dataRP[:, channelMask]

                if nDropPCA is not None:
                    dataRP = drop_PCA(dataRP, nDropPCA)

                corrLst += [tril_1D(np.corrcoef(dataRP.T))]

            # fig, ax = plt.subplots(nrows=nMice, ncols=nMice, figsize=(4 * nMice, 4 * nMice))

            pairDict = {}
            rezMat = np.zeros((nPhases, nPhases))
            for idxNamei, iName in enumerate(phases):
                pairDict[iName] = corrLst[idxNamei]

                for idxNamej, jName in enumerate(phases):
                    rezMat[idxNamei][idxNamej] = np.corrcoef(corrLst[idxNamei], corrLst[idxNamej])[0, 1]

            pPlot = sns.pairplot(data=pd.DataFrame(pairDict), vars=phases, kind='kde')

            plt.savefig('corr_consistency_bymouse_scatter_' + fnameSuffix + '.png')
            plt.close()

            fig, ax = plt.subplots()
            imshow(fig, ax, rezMat, haveColorBar=True, limits=[0,1], xTicks=phases, yTicks=phases, cmap='jet')
            plt.savefig('corr_consistency_bymouse_metric_' + fnameSuffix + '.png')
            plt.close()

            avgConsistency = np.round(np.mean(offdiag_1D(rezMat)), 2)
            dfConsistency = pd_append_row(dfConsistency, [mousename, trialType, avgConsistency])

    fig, ax = plt.subplots()
    dfPivot = pd_pivot(dfConsistency, *dfColumns)
    sns.heatmap(data=dfPivot, ax=ax, annot=True, vmin=0, vmax=1, cmap='jet')
    fig.savefig('consistency_coactivity_metric_dropPCA_' + str(nDropPCA) + '_' + datatype + '_' + str(performance) + '.png')
    plt.close()


###############################
# PCA
###############################


def plot_pca_alignment_bymouse(dataDB, datatype='bn_session', trialType=None, intervName=None):
    nMouse = len(dataDB.mice)
    mice = sorted(dataDB.mice)

    fig1, ax1 = plt.subplots(ncols=2, figsize=(8, 4))
    fig2, ax2 = plt.subplots(nrows=nMouse, ncols=nMouse, figsize=(4 * nMouse, 4 * nMouse))

    compLst = []
    varLst = []
    for mousename in sorted(dataDB.mice):
        dataRSP = dataDB.get_neuro_data({'mousename': mousename},
                                        intervName=intervName, datatype=datatype, trialType=trialType)
        dataRSP = np.concatenate(dataRSP, axis=0)
        dataRP = np.concatenate(dataRSP, axis=0)  # Use timesteps as samples

        pca = PCA(n_components=dataRP.shape[1])
        pca.fit(dataRP)
        ax1[0].semilogy(pca.explained_variance_ratio_, label=mousename)

        compLst += [np.copy(pca.components_)]
        varLst += [np.sqrt(pca.explained_variance_ratio_)]

    # Compute pca_alignment coefficient
    matAlign = np.zeros((nMouse, nMouse))
    for iMouse in range(nMouse):
        ax2[iMouse][0].set_ylabel(mice[iMouse])
        ax2[-1][iMouse].set_xlabel(mice[iMouse])

        for jMouse in range(nMouse):
            matVar = np.outer(varLst[iMouse], varLst[jMouse])
            matComp = np.abs(compLst[iMouse].dot(compLst[jMouse].T))
            matTot = matVar * matComp ** 2
            matAlign[iMouse, jMouse] = np.sum(matTot)

            print(np.sum(matComp ** 2))

            ax2[iMouse, jMouse].imshow(matComp, vmin=0, vmax=1)

    ax1[0].set_xlabel('PCA')
    ax1[0].set_ylabel('Explained Variance')

    imshow(fig1, ax1[1], matAlign, limits=[0, 1], title='PCA alignment',
           xTicks=mice, yTicks=mice, haveColorBar=True, cmap='jet')

    ax1[0].legend()

    fig1.savefig('PCA_alignment_values_' + datatype + '_' + str(trialType) + '.png')
    fig2.savefig('PCA_alignment_matrix_' + datatype + '_' + str(trialType) + '.png')
    plt.close()


# FIXME: enable exclude queries
def plot_pca_alignment_byphase(dataDB, intervNames=None, datatype='bn_session', trialType=None):
    for mousename in sorted(dataDB.mice):
        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))

        compLst = []
        varLst = []

        if intervNames is None:
            intervNames = dataDB.get_interval_names()

        for intervName in intervNames:
            dataRSP = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype,
                                            intervName=intervName, trialType=trialType)
            dataRSP = np.concatenate(dataRSP, axis=0)
            dataRP = np.mean(dataRSP, axis=1)

            pca = PCA(n_components=dataRP.shape[1])
            pca.fit(dataRP)
            ax[0].semilogy(pca.explained_variance_ratio_, label=intervName)

            compLst += [np.copy(pca.components_)]
            varLst += [np.copy(pca.explained_variance_ratio_)]

            '''
                TODO: How to use PCA of X to compute explained variance for Y ???
            '''

        #
        matVar = np.outer(varLst[0], varLst[1])
        matComp = np.abs(compLst[0].dot(compLst[1].T))
        matTot = matVar * matComp
        print(np.sum(matTot))

        img = ax[1].imshow(matComp, vmin=0, vmax=1)
        imshow_add_color_bar(fig, ax[1], img)
        ax[0].legend()

        fig.savefig('PCA_alignment_byphase_' + '_'.join([datatype, mousename, str(trialType)]) + '.png')
        plt.close()


# FIXME: enable exclude queries
def plot_pca_consistency(dataDB, intervNames=None, dropFirst=None, dropChannels=None):
    mice = sorted(dataDB.mice)
    nMice = len(mice)

    dfColumns = ['datatype', 'phase', 'consistency']
    dfConsistency = pd.DataFrame(columns=dfColumns)

    if intervNames is None:
        intervNames = dataDB.get_interval_names()

    for datatype in dataDB.get_data_types():
        for intervName in intervNames:
            fnameSuffix = datatype + '_' + intervName
            print(fnameSuffix)

            dataLst = []

            for iMouse, mousename in enumerate(mice):
                dataRSPLst = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype,
                                                   intervName=intervName)

                print(set([d.shape[1:] for d in dataRSPLst]))

                dataRSP = np.concatenate(dataRSPLst, axis=0)
                dataRP = np.mean(dataRSP, axis=1)
                dataRP = zscore(dataRP, axis=0)

                if dropChannels is not None:
                    channelMask = np.ones(dataRP.shape[1]).astype(bool)
                    channelMask[dropChannels] = 0
                    dataRP = dataRP[:, channelMask]

                dataLst += [dataRP]

            fig, ax = plt.subplots(nrows=nMice, ncols=nMice, figsize=(4 * nMice, 4 * nMice))

            rezMat = np.zeros((nMice, nMice))
            for iMouse in range(nMice):
                ax[iMouse][0].set_ylabel(mice[iMouse])
                ax[-1][iMouse].set_xlabel(mice[iMouse])

                for jMouse in range(nMice):
                    rezXY, rezYX, arrXX, arrXY, arrYY, arrYX = pca.paired_comparison(
                        dataLst[iMouse], dataLst[jMouse], dropFirst=dropFirst)
                    rezMat[iMouse][jMouse] = rezXY # np.mean([rezXY, rezYX])

                    ax[iMouse][jMouse].plot(arrXX, label='XX')
                    ax[iMouse][jMouse].plot(arrXY, label='XY')
                    ax[iMouse][jMouse].legend()

            plt.savefig('pca_consistency_bymouse_evals_' + fnameSuffix + '.png')
            plt.close()

            fig, ax = plt.subplots()
            imshow(fig, ax, rezMat, haveColorBar=True, limits=[0,1], xTicks=mice, yTicks=mice)
            plt.savefig('pca_consistency_bymouse_metric_' + fnameSuffix + '.png')
            plt.close()

            avgConsistency = np.round(np.mean(offdiag_1D(rezMat)), 2)
            dfConsistency = pd_append_row(dfConsistency, [datatype, intervName, avgConsistency])

    fig, ax = plt.subplots()
    dfPivot = pd_pivot(dfConsistency, *dfColumns)
    sns.heatmap(data=dfPivot, ax=ax, annot=True, vmin=0, vmax=1, cmap='jet')
    fig.savefig('consistency_coactivity_metric.png')
    plt.close()


#######################
# Movies
#######################

def plot_corr_movie_mousetrialtype(dataDB, mc, estimator, exclQueryLst=None, nDropPCA=None, haveDelay=False, fontsize=20, **kwargs):
    assert 'trialType' in kwargs.keys(), 'Requires trial types'
    dps = DataParameterSweep(dataDB, exclQueryLst, mousename='auto', **kwargs)
    nMice = dps.param_size('mousename')
    nTrialType = dps.param_size('trialType')

    for paramVals, dfTmp in dps.sweepDF.groupby(dps.invert_param(['mousename', 'trialType'])):
        plotSuffix = param_vals_to_suffix(paramVals)

        # Store all preprocessed data first
        dataDict = {}
        for mousename, dfMouse in dfTmp.groupby(['mousename']):
            for idx, row in dfMouse.iterrows():
                trialType = row['trialType']

                print('Reading data, ', plotSuffix, mousename, trialType)

                kwargsThis = pd_row_to_kwargs(row, parseNone=True, dropKeys=['mousename'])
                dataLst = get_data_list(dataDB, haveDelay, mousename, **kwargsThis)
                dataRSP = np.concatenate(dataLst, axis=0)

                if nDropPCA is not None:
                    dataRSP = drop_PCA_3D(dataRSP, nDropPCA)

                mc.set_data(dataRSP, 'rsp')
                metricSettings = {'havePVal': False, 'estimator': estimator}
                rezS = mc.metric3D('corr', 's', metricSettings=metricSettings)
                dataDict[(mousename, trialType)] = rezS

        # Test that all datasets have the same duration
        shapeSet = set([v.shape for v in dataDict.values()])
        assert len(shapeSet) == 1
        nTimes = shapeSet.pop()[0]

        progBar = IntProgress(min=0, max=nTimes, description=plotSuffix)
        display(progBar)  # display the bar
        for iTime in range(nTimes):
            fig, ax = plt.subplots(nrows=nMice, ncols=nTrialType, figsize=(4*nTrialType, 4*nMice), tight_layout=True)

            for iMouse, mousename in enumerate(dps.param('mousename')):
                ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
                for iTT, trialType in enumerate(dps.param('trialType')):
                    ax[0][iTT].set_title(trialType, fontsize=fontsize)
                    # print(datatype, mousename)

                    rezS = dataDict[(mousename, trialType)][iTime]

                    haveColorBar = iTT == nTrialType - 1
                    imshow(fig, ax[iMouse][iTT], rezS, limits=[-1, 1], cmap='jet', haveColorBar=haveColorBar)

            plt.savefig('corr_mouseTrialType_dropPCA_' + str(nDropPCA) + '_' + plotSuffix + '_' + str(iTime) + '.png')
            plt.close()
            progBar.value += 1