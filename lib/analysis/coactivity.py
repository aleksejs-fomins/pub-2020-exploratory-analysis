import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from mesostat.utils.signals.filter import zscore, drop_PCA
import mesostat.stat.consistency.pca as pca
from mesostat.stat.connectomics import offdiag_1D
from mesostat.utils.pandas_helper import outer_product_df, pd_append_row
from mesostat.visualization.mpl_matrix import imshow
from mesostat.visualization.mpl_colorbar import imshow_add_color_bar


def corr_plot_session_composite(dataDB, mc, intervDict, estimator, datatype, nDropPCA=None,
                                trialTypes=None, performances=None, haveMono=True, thrMono=0.4):
    argSweepDict = {
        'mousename' : dataDB.mice,
        'intervName' : list(intervDict.keys())
    }
    if trialTypes is not None:
        argSweepDict['trialType'] = trialTypes
    if performances is not None:
        argSweepDict['performance'] = performances

    sweepDF = outer_product_df(argSweepDict)

    for idx, row in sweepDF.iterrows():
        kwargs = dict(row)
        del kwargs['mousename']
        del kwargs['intervName']

        results = []
        # NOTE: zscore channels for each session to avoid session-wise effects
        dataRSPLst = dataDB.get_neuro_data({'mousename' : row['mousename']}, datatype=datatype,
                                            cropTime=intervDict[row['intervName']],
                                            zscoreDim='rs',
                                            **kwargs)
        dataRSP = np.concatenate(dataRSPLst, axis=0)
        dataRP = np.mean(dataRSP, axis=1)

        if nDropPCA is not None:
            dataRP = drop_PCA(dataRP, nDropPCA)

        mc.set_data(dataRP, 'rp')
        # mc.set_data(dataRSP, 'rsp')
        # metricSettings={'timeAvg' : True, 'havePVal' : False, 'estimator' : estimator}
        metricSettings={'havePVal' : False, 'estimator' : estimator}
        rez2D = mc.metric3D('corr', '', metricSettings=metricSettings)


        plotSuffix = datatype + '_' + '_'.join(list(row.values))
        plt.figure()
        plt.imshow(rez2D, vmin=-1, vmax=1, cmap='jet')
        plt.colorbar()
        plt.savefig('corr_all_' + plotSuffix + '.png')
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
                chLabels = np.array(dataDB.get_channel_labels(row['mousename']))[idxsPlot]
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


def plot_pca_alignment_bymouse(dataDB, datatype='bn_session', trialType=None):
    nMouse = len(dataDB.mice)
    mice = sorted(dataDB.mice)

    fig1, ax1 = plt.subplots(ncols=2, figsize=(8, 4))
    fig2, ax2 = plt.subplots(nrows=nMouse, ncols=nMouse, figsize=(4 * nMouse, 4 * nMouse))

    compLst = []
    varLst = []
    for mousename in sorted(dataDB.mice):
        dataRSP = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype, trialType=trialType)
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


def plot_pca_alignment_byphase(dataDB, intervDict, datatype='bn_session', trialType=None):
    for mousename in sorted(dataDB.mice):
        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))

        compLst = []
        varLst = []

        for intervName, interv in intervDict.items():
            dataRSP = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype,
                                            cropTime=interv, trialType=trialType)
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


def plot_pca_consistency(dataDB, intervDict, dropFirst=None):
    mice = sorted(dataDB.mice)
    nMice = len(mice)

    dfConsistency = pd.DataFrame(columns=['datatype', 'phase', 'consistency'])

    for datatype in dataDB.get_data_types():
        for intervName, interv in intervDict.items():
            fnameSuffix = datatype + '_' + intervName
            print(fnameSuffix)

            dataLst = []

            for iMouse, mousename in enumerate(mice):
                dataRSPLst = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype,
                                                   cropTime=interv)

                print(set([d.shape[1:] for d in dataRSPLst]))

                dataRSP = np.concatenate(dataRSPLst, axis=0)
                dataRP = np.mean(dataRSP, axis=1)
                dataRP = zscore(dataRP, axis=0)

                dataLst += [dataRP]

            fig, ax = plt.subplots(nrows=nMice, ncols=nMice, figsize=(4 * nMice, 4 * nMice))

            rezMat = np.zeros((nMice, nMice))
            for iMouse in range(nMice):
                for jMouse in range(nMice):
                    rezXY, rezYX, arrXX, arrXY, arrYY, arrYX = pca.paired_comparison(
                        dataLst[iMouse], dataLst[jMouse], dropFirst=dropFirst)
                    rezMat[iMouse][jMouse] = rezXY # np.mean([rezXY, rezYX])

                    ax[iMouse][jMouse].plot(arrXX, label='XX')
                    ax[iMouse][jMouse].plot(arrXY, label='XY')
                    ax[iMouse][jMouse].legend()

            plt.savefig('pca_consistency_bymouse_evals_' + fnameSuffix + '.png')
            plt.close()

            plt.figure()
            plt.imshow(rezMat, vmin=0, vmax=1)
            plt.colorbar()
            plt.savefig('pca_consistency_bymouse_metric_' + fnameSuffix + '.png')
            plt.close()

            avgConsistency = np.round(np.mean(offdiag_1D(rezMat)), 2)
            dfConsistency = pd_append_row(dfConsistency, [datatype, intervName, avgConsistency])

    dfPivot = dfConsistency.pivot(index='datatype', columns='phase', values='consistency')
    fig, ax = plt.subplots()
    sns.heatmap(data=dfPivot, ax=ax, annot=True, vmin=0, vmax=1, cmap='jet')
    fig.savefig('consistency_coactivity_metric.png')
    plt.close()


