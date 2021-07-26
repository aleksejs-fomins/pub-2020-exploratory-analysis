import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from mesostat.utils.signals.filter import zscore, drop_PCA
import mesostat.stat.consistency.pca as pca
from mesostat.stat.connectomics import tril_1D, offdiag_1D
from mesostat.utils.pandas_helper import outer_product_df, pd_append_row, pd_pivot, drop_rows_byquery, pd_is_one_row, pd_query
from mesostat.visualization.mpl_matrix import imshow
from mesostat.visualization.mpl_colorbar import imshow_add_color_bar
from mesostat.stat.clustering import cluster_dist_matrix_max, cluster_plot


###############################
# Correlation Plots
###############################


def subset_dict(d1, d2):
    return d1.items() <= d2.items()


def _cluster_brain_plot(fig, ax, dataDB, clusters, dropChannels=None, haveColorBar=True):
    clusterDict = {c: np.where(clusters == c)[0] for c in sorted(set(clusters))}
    if dropChannels is not None:
        # Correct channel indices given that some channels were dropped
        dropChannels = np.array(dropChannels)
        clusterDict = {c: [el + np.sum(dropChannels < el) for el in v] for c, v in clusterDict.items()}

    dataDB.plot_area_clusters(fig, ax, clusterDict, haveLegend=True, haveColorBar=haveColorBar)


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


def calc_corr_mouse(dataDB, mc, mousename, nDropPCA=1, dropChannels=None, estimator='corr', **kwargs):
    # NOTE: zscore channels for each session to avoid session-wise effects
    dataRSPLst = dataDB.get_neuro_data({'mousename': mousename},
                                       zscoreDim='rs',
                                       **kwargs)
    dataRSP = np.concatenate(dataRSPLst, axis=0)
    dataRP = np.mean(dataRSP, axis=1)
    channelLabels = np.array(dataDB.get_channel_labels())

    if nDropPCA is not None:
        dataRP = drop_PCA(dataRP, nDropPCA)

    if dropChannels is not None:
        nChannels = dataRP.shape[1]
        channelMask = np.ones(nChannels).astype(bool)
        channelMask[dropChannels] = 0
        dataRP = dataRP[:, channelMask]
        channelLabels = channelLabels[channelMask]

    mc.set_data(dataRP, 'rp')
    # mc.set_data(dataRSP, 'rsp')
    # metricSettings={'timeAvg' : True, 'havePVal' : False, 'estimator' : estimator}
    metricSettings = {'havePVal': False, 'estimator': estimator}
    rez2D = mc.metric3D('corr', '', metricSettings=metricSettings)
    return channelLabels, rez2D


def plot_corr_mousephase(dataDB, mc, estimator, intervNames=None, dataTypes=None, nDropPCA=None,
                                dropChannels=None, haveBrain=False, trialTypes=None, performances=None, haveMono=True,
                                exclQueryLst=None, thrMono=0.4, clusterParam=-10, fontsize=20):

    mice = sorted(dataDB.mice)
    intervNames = intervNames if intervNames is not None else dataDB.get_interval_names()
    nMice = len(mice)
    nInterv = len(intervNames)

    argSweepDict = {
        'datatype': dataTypes if dataTypes is not None else dataDB.get_data_types(),
        'trialType': trialTypes if trialTypes is not None else dataDB.get_trial_type_names()
    }
    if performances is not None:
        argSweepDict['performance'] = performances

    sweepDF = outer_product_df(argSweepDict)
    for idx, row in sweepDF.iterrows():
        plotSuffix = '_'.join([str(s) for s in row.values])
        print(plotSuffix)

        figCorr, axCorr = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4*nInterv, 4*nMice), tight_layout=True)
        figClust, axClust = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4 * nInterv, 4 * nMice), tight_layout=True)
        if haveBrain:
            figBrain, axBrain = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4 * nInterv, 4 * nMice), tight_layout=True)
        if haveMono:
            figMono, axMono = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4 * nInterv, 4 * nMice), tight_layout=True)

        for iMouse, mousename in enumerate(mice):
            axCorr[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
            axClust[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

            if haveBrain:
                axBrain[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
            if haveMono:
                axMono[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

            for iInterv, intervName in enumerate(intervNames):
                axCorr[0][iInterv].set_title(intervName, fontsize=fontsize)
                axClust[0][iInterv].set_title(intervName, fontsize=fontsize)

                if haveBrain:
                    axBrain[0][iInterv].set_title(intervName, fontsize=fontsize)
                if haveMono:
                    axMono[0][iInterv].set_title(intervName, fontsize=fontsize)

                kwargsThis = {**dict(row), **{'mousename': mousename, 'intervName': intervName}}
                kwargsThis = {k: v if v != 'None' else None for k, v in kwargsThis.items()}

                if any([subset_dict(d, kwargsThis) for d in exclQueryLst]):
                    print('--skipping', kwargsThis)
                    continue

                del kwargsThis['mousename']
                channelLabels, rez2D = calc_corr_mouse(dataDB, mc, mousename,
                                                       nDropPCA=nDropPCA, dropChannels=dropChannels,
                                                       estimator=estimator, **kwargsThis)

                haveColorBar = iInterv == nInterv - 1

                # Plot correlations
                imshow(figCorr, axCorr[iMouse][iInterv], rez2D, title='corr', limits=[-1,1], cmap='jet',
                       haveColorBar=haveColorBar)

                # Plot clustering
                clusters = cluster_dist_matrix_max(rez2D, clusterParam, method='Affinity')
                cluster_plot(figClust, axClust[iMouse][iInterv], rez2D, clusters, channelLabels, limits=[-1,1],
                             cmap='jet', haveColorBar=haveColorBar)

                if haveBrain:
                    _cluster_brain_plot(figBrain, axBrain[iMouse][iInterv], dataDB, clusters,
                                        dropChannels=dropChannels, haveColorBar=haveColorBar)

                if haveMono:
                    _plot_corr_1D(figMono, axMono[iMouse][iInterv], channelLabels, rez2D, thrMono)

        # Save image
        figCorr.savefig('corr_mousephase_' + plotSuffix + '.png')
        plt.close(figCorr)
        figClust.savefig('corr_clust_mousephase_' + plotSuffix + '.png')
        plt.close(figClust)
        if haveBrain:
            figBrain.savefig('corr_clust_brainplot_mousephase_' + plotSuffix + '.png')
            plt.close(figBrain)
        if haveMono:
            figMono.savefig('corr_1D_mousephase_' + plotSuffix + '.png')
            plt.close(figMono)


def plot_corr_mousetrialtype(dataDB, mc, estimator, intervNames=None, dataTypes=None, nDropPCA=None,
                             dropChannels=None, haveBrain=False, trialTypes=None, performances=None, haveMono=True,
                             exclQueryLst=None, thrMono=0.4, clusterParam=-10, fontsize=20):
    mice = sorted(dataDB.mice)
    nMice = len(mice)
    trialTypes = trialTypes if trialTypes is not None else dataDB.get_trial_type_names()
    nTrialType = len(trialTypes)

    argSweepDict = {
        'datatype': dataTypes if dataTypes is not None else dataDB.get_data_types(),
        'intervName': intervNames if intervNames is not None else dataDB.get_interval_names()
    }
    if performances is not None:
        argSweepDict['performance'] = performances

    sweepDF = outer_product_df(argSweepDict)
    for idx, row in sweepDF.iterrows():
        plotSuffix = '_'.join([str(s) for s in row.values])
        print(plotSuffix)

        figCorr, axCorr = plt.subplots(nrows=nMice, ncols=nTrialType, figsize=(4*nTrialType, 4*nMice), tight_layout=True)
        figClust, axClust = plt.subplots(nrows=nMice, ncols=nTrialType, figsize=(4 * nTrialType, 4 * nMice), tight_layout=True)
        if haveBrain:
            figBrain, axBrain = plt.subplots(nrows=nMice, ncols=nTrialType, figsize=(4 * nTrialType, 4 * nMice), tight_layout=True)
        if haveMono:
            figMono, axMono = plt.subplots(nrows=nMice, ncols=nTrialType, figsize=(4 * nTrialType, 4 * nMice), tight_layout=True)

        for iMouse, mousename in enumerate(mice):
            axCorr[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
            axClust[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

            if haveBrain:
                axBrain[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
            if haveMono:
                axMono[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

            for iTrialType, trialType in enumerate(trialTypes):
                axCorr[0][iTrialType].set_title(trialType, fontsize=fontsize)
                axClust[0][iTrialType].set_title(trialType, fontsize=fontsize)

                if haveBrain:
                    axBrain[0][iTrialType].set_title(trialType, fontsize=fontsize)
                if haveMono:
                    axMono[0][iTrialType].set_title(trialType, fontsize=fontsize)

                kwargsThis = {**dict(row), **{'mousename': mousename, 'trialType': trialType}}
                kwargsThis = {k: v if v != 'None' else None for k, v in kwargsThis.items()}

                if any([subset_dict(d, kwargsThis) for d in exclQueryLst]):
                    print('--skipping', kwargsThis)
                    continue

                del kwargsThis['mousename']
                channelLabels, rez2D = calc_corr_mouse(dataDB, mc, mousename,
                                                       nDropPCA=nDropPCA, dropChannels=dropChannels,
                                                       estimator=estimator, **kwargsThis)

                haveColorBar = iTrialType == nTrialType - 1

                # Plot correlations
                imshow(figCorr, axCorr[iMouse][iTrialType], rez2D, title='corr', limits=[-1,1], cmap='jet',
                       haveColorBar=haveColorBar)

                # Plot clustering
                clusters = cluster_dist_matrix_max(rez2D, clusterParam, method='Affinity')
                cluster_plot(figClust, axClust[iMouse][iTrialType], rez2D, clusters, channelLabels, limits=[-1,1],
                             cmap='jet', haveColorBar=haveColorBar)

                if haveBrain:
                    _cluster_brain_plot(figBrain, axBrain[iMouse][iTrialType], dataDB, clusters,
                                        dropChannels=dropChannels, haveColorBar=haveColorBar)

                if haveMono:
                    _plot_corr_1D(figMono, axMono[iMouse][iTrialType], channelLabels, rez2D, thrMono)

        # Save image
        figCorr.savefig('corr_mouseTrialType_' + plotSuffix + '.png')
        plt.close(figCorr)
        figClust.savefig('corr_clust_mouseTrialType_' + plotSuffix + '.png')
        plt.close(figClust)
        if haveBrain:
            figBrain.savefig('corr_clust_brainplot_mouseTrialType_' + plotSuffix + '.png')
            plt.close(figBrain)
        if haveMono:
            figMono.savefig('corr_1D_mouseTrialType_' + plotSuffix + '.png')
            plt.close(figMono)


def plot_corr_mousephase_subpre(dataDB, mc, estimator, intervNames=None, nDropPCA=None,
                                dropChannels=None, trialTypes=None, performances=None, exclQueryLst=None, fontsize=20):
    mice = sorted(dataDB.mice)
    intervNames = intervNames if intervNames is not None else dataDB.get_interval_names()
    nMice = len(mice)
    nInterv = len(intervNames)

    argSweepDict = {
        'datatype': ['bn_session'],
        'trialType': trialTypes if trialTypes is not None else dataDB.get_trial_type_names()
    }
    if performances is not None:
        argSweepDict['performance'] = performances

    sweepDF = outer_product_df(argSweepDict)
    for idx, row in sweepDF.iterrows():
        plotSuffix = '_'.join([str(s) for s in row.values])
        print(plotSuffix)

        figCorr, axCorr = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4*nInterv, 4*nMice), tight_layout=True)

        for iMouse, mousename in enumerate(mice):
            axCorr[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
            rezDict = {}
            for iInterv, intervName in enumerate(intervNames):
                axCorr[0][iInterv].set_title(intervName, fontsize=fontsize)

                assert (iInterv != 0) or (intervName == 'PRE')
                kwargsThis = {**dict(row), **{'mousename': mousename, 'intervName': intervName}}
                kwargsThis = {k: v if v != 'None' else None for k, v in kwargsThis.items()}

                if any([subset_dict(d, kwargsThis) for d in exclQueryLst]):
                    print('--skipping', kwargsThis)
                    continue

                del kwargsThis['mousename']
                channelLabels, rez2D = calc_corr_mouse(dataDB, mc, mousename,
                                                       nDropPCA=nDropPCA, dropChannels=dropChannels,
                                                       estimator=estimator, **kwargsThis)

                rezDict[intervName] = rez2D

            # Plot correlations
            for iInterv, intervName in enumerate(intervNames):
                if (intervName in rezDict.keys()) and (intervName != 'PRE'):
                    haveColorBar = iInterv == nInterv - 1
                    imshow(figCorr, axCorr[iMouse][iInterv], rezDict[intervName] - rezDict['PRE'],
                           title='corr', haveColorBar=haveColorBar, limits=[-1, 1], cmap='RdBu_r')

        # Save image
        figCorr.savefig('corr_subpre_' + plotSuffix + '.png')
        plt.close()


def plot_corr_mousephase_submouse(dataDB, mc, estimator, intervNames=None, dataTypes=None, nDropPCA=None,
                               dropChannels=None, trialTypes=None, performances=None, exclQueryLst=None, fontsize=20):
    mice = sorted(dataDB.mice)
    intervNames = intervNames if intervNames is not None else dataDB.get_interval_names()
    nMice = len(mice)
    nInterv = len(intervNames)

    argSweepDict = {
        'datatype': dataTypes if dataTypes is not None else dataDB.get_data_types(),
        'trialType': trialTypes if trialTypes is not None else dataDB.get_trial_type_names()
    }
    if performances is not None:
        argSweepDict['performance'] = performances

    sweepDF = outer_product_df(argSweepDict)
    for idx, row in sweepDF.iterrows():
        plotSuffix = '_'.join([str(s) for s in row.values])
        print(plotSuffix)

        figCorr, axCorr = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4*nInterv, 4*nMice))

        for iInterv, intervName in enumerate(intervNames):
            assert (iInterv != 0) or (intervName == 'PRE')

            axCorr[0][iInterv].set_title(intervName, fontsize=fontsize)

            rezDict = {}
            for iMouse, mousename in enumerate(mice):

                axCorr[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

                kwargsThis = {**dict(row), **{'mousename': mousename, 'intervName': intervName}}
                kwargsThis = {k: v if v != 'None' else None for k, v in kwargsThis.items()}

                if any([subset_dict(d, kwargsThis) for d in exclQueryLst]):
                    print('--skipping', kwargsThis)
                    continue

                del kwargsThis['mousename']
                channelLabels, rez2D = calc_corr_mouse(dataDB, mc, mousename,
                                                       nDropPCA=nDropPCA, dropChannels=dropChannels,
                                                       estimator=estimator, **kwargsThis)

                rezDict[mousename] = rez2D

            # Plot correlations
            rezMean = np.mean(list(rezDict.values()), axis=0)
            for iMouse, mousename in enumerate(mice):
                if mousename in rezDict.keys():
                    haveColorBar = iInterv == nInterv - 1
                    imshow(figCorr, axCorr[iMouse][iInterv], rezDict[mousename] - rezMean,
                           title='corr', haveColorBar=haveColorBar, limits=[-1,1], cmap='RdBu_r')

        # Save image
        figCorr.savefig('corr_submouse_' + plotSuffix + '.png')
        plt.close()


def plot_corr_mouse_2DF(dfDict, mc, estimator, intervNameMap, intervOrdMap,
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

                        channelLabels, rez2D = calc_corr_mouse(dataDB, mc, mousename,
                                                               nDropPCA=nDropPCA, dropChannels=dropChannels,
                                                               estimator=estimator, **kwargs)

                        imshow(fig, ax[iDB][iMouse], rez2D,
                               limits=[-1, 1], cmap='jet', haveColorBar=iMouse == nMice-1)

            # Save image
            plt.savefig('corr_bystim_bn_session_' + plotSuffix + '.png')
            plt.close()


###############################
# Diff
###############################

def plot_corr_consistency_l1_mouse(dataDB, nDropPCA=None, dropChannels=None, performances=None,
                                   trialTypes=None, exclQueryLst=None):
    mice = sorted(dataDB.mice)
    nMice = len(mice)

    argSweepDict = {
        'mousename': dataDB.mice,
        'intervName': dataDB.get_interval_names(),
        'datatype': dataDB.get_data_types(),
        'trialType': trialTypes if trialTypes is not None else [None],
        'performance': performances if performances is not None else [None],
    }

    sweepDF = outer_product_df(argSweepDict)
    if exclQueryLst is not None:
        sweepDF = drop_rows_byquery(sweepDF, exclQueryLst)

    sweepParam = list(argSweepDict.keys())
    sweepParamNoMouse = list(set(sweepParam) - {'mousename'})

    for paramExtra, dfExtra in sweepDF.groupby(['trialType', 'performance']):
        plotExtraSuffix = '_'.join([str(s) for s in paramExtra if s is not None])

        dfColumns = ['datatype', 'phase', 'consistency']
        dfConsistency = pd.DataFrame(columns=dfColumns)

        for paramVals, dfMouse in dfExtra.groupby(sweepParamNoMouse):
            plotSuffix = '_'.join([str(s) for s in paramExtra + paramVals if s is not None])
            print(plotSuffix)

            corrLst = []
            for idx, row in dfMouse.iterrows():
                plotSuffix = '_'.join([str(s) for s in row.values])
                print(plotSuffix)

                kwargs = dict(row)
                del kwargs['mousename']

                # NOTE: zscore channels for each session to avoid session-wise effects
                dataRSPLst = dataDB.get_neuro_data({'mousename' : row['mousename']},
                                                    zscoreDim='rs',
                                                    **kwargs)

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
        fig.savefig('consistency_coactivity_metric_'+plotExtraSuffix+'.png')
        plt.close()


def plot_corr_consistency_l1_trialtype(dataDB, nDropPCA=None, dropChannels=None, performance=None,
                                       trialTypes=None, datatype=None):
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
    fig.savefig('consistency_coactivity_metric_' + datatype + '_' + str(performance) + '.png')
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
    fig.savefig('consistency_coactivity_metric_' + datatype + '_' + str(performance) + '.png')
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


