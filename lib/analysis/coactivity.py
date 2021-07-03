import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from mesostat.utils.signals.filter import zscore, drop_PCA
import mesostat.stat.consistency.pca as pca
from mesostat.stat.connectomics import tril_1D, offdiag_1D
from mesostat.utils.pandas_helper import outer_product_df, pd_append_row, pd_pivot, drop_rows_byquery
from mesostat.visualization.mpl_matrix import imshow
from mesostat.visualization.mpl_colorbar import imshow_add_color_bar
from mesostat.stat.clustering import cluster_dist_matrix, cluster_plot

###############################
# Correlation Plots
###############################


def corr_plot_session_composite(dataDB, mc, estimator, intervNames=None, dataTypes=None, nDropPCA=None,
                                dropChannels=None, haveBrain=False, trialTypes=None, performances=None, haveMono=True,
                                exclQueryLst=None, thrMono=0.4, clusterParam=-10):
    argSweepDict = {
        'mousename' : dataDB.mice,
        'intervName' : intervNames if intervNames is not None else dataDB.get_interval_names()
    }

    if dataTypes is not None:
        argSweepDict['datatype'] = dataTypes
    if trialTypes is not None:
        argSweepDict['trialType'] = trialTypes
    if performances is not None:
        argSweepDict['performance'] = performances

    sweepDF = outer_product_df(argSweepDict)
    if exclQueryLst is not None:
        sweepDF = drop_rows_byquery(sweepDF, exclQueryLst)

    for idx, row in sweepDF.iterrows():
        plotSuffix = '_'.join([str(s) for s in row.values])
        print(plotSuffix)

        kwargs = dict(row)
        del kwargs['mousename']
        del kwargs['intervName']

        results = []
        # NOTE: zscore channels for each session to avoid session-wise effects
        dataRSPLst = dataDB.get_neuro_data({'mousename' : row['mousename']},
                                            intervName=row['intervName'],
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
        metricSettings={'havePVal' : False, 'estimator' : estimator}
        rez2D = mc.metric3D('corr', '', metricSettings=metricSettings)

        # Plot correlations
        fig, ax = plt.subplots(nrows=2, figsize=(4, 8))
        imshow(fig, ax[0], rez2D, title='corr', haveColorBar=True, limits=[-1,1], cmap='jet')

        # Plot clustering
        clusters = cluster_dist_matrix(rez2D, clusterParam, method='affinity')
        cluster_plot(fig, ax[1], rez2D, clusters, channelLabels)

        # Save image
        plt.savefig('corr_all_' + plotSuffix + '.png')
        plt.close()

        if haveBrain:
            clusterDict = {c : np.where(clusters == c)[0] for c in sorted(set(clusters))}
            if dropChannels is not None:
                # Correct channel indices given that some channels were dropped
                dropChannels = np.array(dropChannels)
                clusterDict = {c : [el + np.sum(dropChannels < el) for el in v] for c,v in clusterDict.items()}

            fig, ax = dataDB.plot_area_clusters(clusterDict, haveLegend=True)
            fig.savefig('corr_all_' + plotSuffix + '_brainplot.png')
            plt.close()

        # Plot channels by their average correlation
        if haveMono:
            np.fill_diagonal(rez2D, np.nan)
            rez1D = np.nanmean(rez2D, axis=0)
            idxsPlot = rez1D < thrMono

            if np.sum(idxsPlot) == 0:
                print('No channels with avgcorr < ', thrMono, 'for', plotSuffix)
            else:
                vals = rez1D[idxsPlot]
                chLabels = np.array(channelLabels)[idxsPlot]
                idxsValsSort = np.argsort(vals)
                vals = vals[idxsValsSort]
                chLabels = chLabels[idxsValsSort]

                fig, ax = plt.subplots()
                g = sns.barplot(ax=ax, x=chLabels, y=vals)
                g.set_xticklabels(chLabels, rotation=90)
                ax.set_ylim(-1, 1)
                ax.axhline(y=0, color='pink', linestyle='--')
                fig.savefig('corr_1D_all_' + plotSuffix + '.png')
                plt.close()


###############################
# Diff
###############################

def plot_corr_consistency_l1_mouse(dataDB, intervDict, nDropPCA=None, dropChannels=None, performance=None):
    mice = sorted(dataDB.mice)
    nMice = len(mice)

    dfColumns = ['datatype', 'phase', 'consistency']
    dfConsistency = pd.DataFrame(columns=dfColumns)

    for datatype in dataDB.get_data_types():
        for intervName, interv in intervDict.items():
            fnameSuffix = '_'.join([datatype, intervName, str(performance)])
            print(fnameSuffix)

            corrLst = []
            for iMouse, mousename in enumerate(mice):
                kwargs = {'datatype' : datatype, 'cropTime' : interv}
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

            plt.savefig('corr_consistency_bymouse_scatter_' + fnameSuffix + '.png')
            plt.close()

            fig, ax = plt.subplots()
            imshow(fig, ax, rezMat, haveColorBar=True, limits=[0,1], xTicks=mice, yTicks=mice, cmap='jet')
            plt.savefig('corr_consistency_bymouse_metric_' + fnameSuffix + '.png')
            plt.close()

            avgConsistency = np.round(np.mean(offdiag_1D(rezMat)), 2)
            dfConsistency = pd_append_row(dfConsistency, [datatype, intervName, avgConsistency])

    fig, ax = plt.subplots()
    dfPivot = pd_pivot(dfConsistency, *dfColumns)
    sns.heatmap(data=dfPivot, ax=ax, annot=True, vmin=0, vmax=1, cmap='jet')
    fig.savefig('consistency_coactivity_metric_'+str(performance)+'.png')
    plt.close()


def plot_corr_consistency_l1_trialtype(dataDB, intervDict, nDropPCA=None, dropChannels=None, performance=None,
                                       trialTypes=None, datatype=None):
    mice = sorted(dataDB.mice)
    if trialTypes is None:
        trialTypes = dataDB.get_data_types()
    nTT = len(trialTypes)

    dfColumns = ['mousename', 'phase', 'consistency']

    dfConsistency = pd.DataFrame(columns=dfColumns)

    for iMouse, mousename in enumerate(mice):
        for intervName, interv in intervDict.items():
            fnameSuffix = '_'.join([datatype, mousename, intervName, str(performance)])
            print(fnameSuffix)

            corrLst = []
            for trialType in trialTypes:
                kwargs = {'datatype' : datatype, 'cropTime' : interv, 'trialType' : trialType}
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


def plot_corr_consistency_l1_phase(dataDB, intervDict, nDropPCA=None, dropChannels=None, performance=None, datatype=None):
    mice = sorted(dataDB.mice)
    phases = list(intervDict.keys())
    nPhases = len(intervDict)

    dfColumns = ['mousename', 'trialtype', 'consistency']
    dfConsistency = pd.DataFrame(columns=dfColumns)

    for iMouse, mousename in enumerate(mice):
        for trialType in dataDB.get_trial_type_names():
            fnameSuffix = '_'.join([datatype, mousename, trialType, str(performance)])
            print(fnameSuffix)

            corrLst = []
            for intervName, interv in intervDict.items():
                kwargs = {'datatype' : datatype, 'cropTime' : interv, 'trialType' : trialType}
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


