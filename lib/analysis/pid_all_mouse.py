import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, fisher_exact
from sklearn.metrics import cohen_kappa_score

from IPython.display import display

from mesostat.stat.stat import continuous_empirical_CDF
from mesostat.stat.connectomics import tril_1D, offdiag_1D
from mesostat.stat.classification import confusion_matrix
from mesostat.stat.clustering import cluster_dist_matrix_min, cluster_plot

from mesostat.utils.pandas_helper import pd_query, pd_merge_multiple, pd_is_one_row, pd_append_row, pd_pivot
from mesostat.visualization.mpl_barplot import barplot_stacked_indexed, barplot_labeled
from mesostat.visualization.mpl_matrix import imshow


def pid_all_parse_key(key):
    lst = key.split('_')
    rez = {
        'mousename': '_'.join(lst[1:3]),
        'phase': lst[3],
        'datatype': '_'.join(lst[4:6]),
        'trialType': lst[6]
    }

    if len(lst) == 7:
        return rez
    elif len(lst) == 8:
        rez['performance'] = lst[7]
        return rez
    else:
        raise ValueError('Unexpected key', key)


def pid_all_summary_df(h5fname):
    with h5py.File(h5fname, 'r') as f:
        keys = set(f.keys()) - {'lock'}

    # Only keep value keys, ignore labels for now
    keys = [key for key in keys if key[:5] != 'Label']

    summaryDF = pd.DataFrame()
    for key in keys:
        summaryDF = summaryDF.append(pd.DataFrame({**{'key': key}, **pid_all_parse_key(key)}, index=[0]))

    return summaryDF.reset_index(drop=True)


def read_computed_3D(h5fname, keyVals, pidType):
    keyLabels = 'Label_' + keyVals[4:]

    with h5py.File(h5fname, 'r') as f:
        labels = np.copy(f[keyLabels])
        vals = np.copy(f[keyVals])

    # Currently expect shape (nTriplets, 4 pid types)
    assert vals.ndim == 2
    assert labels.ndim == 2
    assert vals.shape[0] == labels.shape[0]
    assert vals.shape[1] == 4
    assert labels.shape[1] == 3

    # Drop negatives
    vals = np.clip(vals, 0, None)

    if pidType == 'unique':
        valsU1 = vals[:, 0]
        valsU2 = vals[:, 1]
        labelsU1 = labels
        labelsU2 = labels[:, [1,0,2]]  # Swap sources, keep target
        return np.concatenate(labelsU1, labelsU2, axis=0), np.concatenate(valsU1, valsU2, axis=0)
    elif pidType == 'red':
        return labels, vals[:, 2]
    elif pidType == 'syn':
        return labels, vals[:, 3]
    else:
        raise ValueError(pidType)


def list_to_3Dmat(idxs, vals, nChannel):
    rezMat = np.full((nChannel, nChannel, nChannel), np.nan)
    rezMat[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = vals
    return rezMat


def plot_cdf(h5fname, dfSummary):
    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):
        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))

        for iPID, pidType in enumerate(['unique', 'syn', 'red']):
            for idx, row in dataMouse:
                idxs, vals = read_computed_3D(h5fname, row['key'], pidType=pidType)
                x, y = continuous_empirical_CDF(vals)

                ax[iPID].plot(x, y, label=row['mousename'])
                ax[iPID].set_title(pidType)

            ax[iPID].legend()
        plt.savefig('pics/PID_cdf_'+'_'.join(key))
        plt.close()


def plot_violin_test(h5fname, h5fnameRand, dfSummary, dfSummaryRand):
    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):
        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))

        for iPID, pidType in enumerate(['unique', 'syn', 'red']):

            dfPID = pd.DataFrame()

            for idx, row in dataMouse:
                rowRand = pd_is_one_row(pd_query(dfSummaryRand, dict(row)))[1]

                dfTrue = pd.DataFrame()
                idxs, vals = read_computed_3D(h5fname, row['key'], pidType=pidType)
                dfTrue[pidType] = vals
                dfTrue['mousename'] = row['mousename']
                dfTrue['type'] = 'Measured'

                dfRand = pd.DataFrame()
                idxsRand, valsRand = read_computed_3D(h5fnameRand, rowRand['key'], pidType=pidType)
                dfRand[pidType] = valsRand
                dfRand['mousename'] = row['mousename']
                dfRand['type'] = 'Shuffle'

                print('Test:', pidType, row['mousename'], 'pval =', mannwhitneyu(vals, valsRand, alternative='greater')[1])

            sns.violinplot(ax=ax[iPID], x="mousename", y=pidType, hue="type", data=dfPID, scale='width', cut=0)
            ax[iPID].set_title(pidType)

        plt.savefig('pics/PID_violin_'+'_'.join(key))
        plt.close()


def plot_top_triplets(dataDB, h5fname, dfSummary, nTop=20):
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    pidTypes = ['unique', 'syn', 'red']

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):

        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        for iPid, pidType in enumerate(pidTypes):
            ax[iPid].set_title(pidType)

            dfRez = pd.DataFrame()
            for idx, row in dataMouse.iterrows():
                idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                dfRez[row['mousename']] = vals

            mice = sorted(list(set(dfRez.columns)))
            dfRez['bits_mean'] = dfRez.mean(axis=1)
            dfRez = dfRez.sort_values('bits_mean', axis=0, ascending=False)
            dfRez = dfRez.head(nTop)

            labels = [str((labelsCanon[s1],labelsCanon[s2],labelsCanon[t]))
                      for s1,s2,t in zip(idxs[:, 0], idxs[:, 1], idxs[:, 2])]

            rezDict = {mousename : np.array(dfRez[mousename]) for mousename in mice}

            # Plot stacked barplot with absolute numbers. Set ylim_max to total number of sessions
            barplot_stacked_indexed(ax[iPid], rezDict, xTickLabels=labels, xLabel='triplet',
                                    yLabel='bits', title=pidType, iMax=None, rotation=90)

        fig.suptitle('_'.join(key))
        plt.show()


def plot_top_singlets(dataDB, h5fname, dfSummary):
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    nChannel = len(labelsCanon)
    pidTypes = ['unique', 'syn', 'red']

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):

        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        for iPid, pidType in enumerate(pidTypes):
            ax[iPid].set_title(pidType)

            rezDict = {}
            for idx, row in dataMouse.iterrows():
                idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                rezDict[row['mousename']] = [np.mean(vals[idxs[:, 2] == iCh]) for iCh in range(nChannel)]

            # Plot stacked barplot with absolute numbers. Set ylim_max to total number of sessions
            barplot_stacked_indexed(ax[iPid], rezDict, xTickLabels=labelsCanon, xLabel='singlet',
                                    yLabel='bits', title=pidType, iMax=None, rotation=90)

        fig.suptitle('_'.join(key))
        plt.show()


# FIXME: Averaging is not aware of dropped channels
def plot_unique_top_pairs(dataDB, h5fname, dfSummary, nTop=20):
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    nChannel = len(labelsCanon)
    mice = dataDB.mice
    pidTypes = ['unique', 'syn', 'red']

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):

        fig, ax = plt.subplots(nrows=2, ncols=len(mice), figsize=(6*len(mice), 8))
        for iPid, pidType in enumerate(pidTypes):
            ax[iPid].set_title(pidType)

            for iMouse, mousename in enumerate(mice):
                row = pd_is_one_row(pd_query(dataMouse, {'mousename' : mousename}))[1]
                if row is not None:
                    idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                    Mrez3D = list_to_3Dmat(idxs, vals, nChannel)
                    Mrez2D = np.nansum(Mrez3D, axis=1) / (nChannel - 2)  # Target can't be either of the sources

                    imshow(fig, ax[0][iMouse], Mrez2D, title=mousename, ylabel='Unique2D', haveColorBar=True, cmap='jet')

                    # Find 2D indices of nTop strongest links
                    vals1D = Mrez2D.flatten()
                    idxsMax1D = np.argsort(vals1D)[::-1][:2*nTop]
                    idxsMax2D = np.vstack([idxsMax1D // nChannel, idxsMax1D % nChannel]).T
                    valsMax1D = vals1D[idxsMax1D]
                    labelsMax2D = [(labelsCanon[i], labelsCanon[j]) for i,j in idxsMax2D]

                    valsMax1DUnpaired = []
                    labelsMax2DUnpaired = []
                    for l,v in zip(labelsMax2D, valsMax1D):
                        if str((l[1], l[0])) not in labelsMax2DUnpaired:
                            labelsMax2DUnpaired += [str(l)]
                            valsMax1DUnpaired += [v]
                    valsMax1DUnpaired = np.array(valsMax1DUnpaired)
                    labelsMax2DUnpaired = np.array(labelsMax2DUnpaired)

                    barplot_labeled(ax[1][iMouse], valsMax1DUnpaired, labelsMax2DUnpaired, rotation=90)

                    idxsHigh = valsMax1DUnpaired > 0.5
                    print(mousename, labelsMax2DUnpaired[idxsHigh])

        fig.suptitle('_'.join(key))
        plt.show()


# FIXME: Averaging is not aware of dropped channels
# TODO: Impl mouse-specific
def plot_2D_avg(dataDB, h5fname, dfSummary, avgAxis=2):
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    nChannel = len(labelsCanon)
    pidTypes = ['unique', 'syn', 'red']

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):

        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        for iPid, pidType in enumerate(pidTypes):
            Mrez2D = np.zeros((nChannel, nChannel))
            for idx, row in dataMouse.iterrows():
                idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                Mrez3D = list_to_3Dmat(idxs, vals, nChannel)
                Mrez2D += np.nansum(Mrez3D, axis=avgAxis) / (nChannel - 2)  # Target can't be either of the sources

            # Average over mice
            nMice = len(dataMouse)
            Mrez2D /= nMice

            imshow(fig, ax[iPid], Mrez2D, title=pidType, haveColorBar=True, cmap='jet')

        fig.suptitle('_'.join(key))
        plt.show()


# FIXME: Averaging is not aware of dropped channels
# TODO: Impl mouse-specific
def plot_2D_bytarget(dataDB, h5fname, dfSummary, trgChName):
    labels = dataDB.get_channel_labels()
    nChannel = len(labels)
    trgIdx = labels.index(trgChName)
    pidTypes = ['unique', 'syn', 'red']

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):

        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        for iPid, pidType in enumerate(pidTypes):
            Mrez2D = np.zeros((nChannel, nChannel))
            for idx, row in dataMouse.iterrows():
                idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                Mrez3D = list_to_3Dmat(idxs, vals, nChannel)
                Mrez2D += Mrez3D[:, :, trgIdx]

            # Average over mice
            nMice = len(dataMouse)
            Mrez2D /= nMice

            imshow(fig, ax[iPid], Mrez2D, title=pidType, haveColorBar=True, cmap='jet')

        fig.suptitle('_'.join(key))
        plt.show()


def plot_2D_bytarget_synergy_cluster(dataDB, h5fname, dfSummary, trgChName, clusterParam=1.0, dropWeakChannelThr=None):
    labels = dataDB.get_channel_labels()
    nChannel = len(labels)
    trgIdx = labels.index(trgChName)

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):

        mice = list(sorted(set(dataMouse['mousename'])))
        nMice = len(mice)
        fig, ax = plt.subplots(nrows=2, ncols=nMice, figsize=(4*nMice, 8))

        for iMouse, (idx, row) in enumerate(dataMouse.iterrows()):
            idxs, vals = read_computed_3D(h5fname, row['key'], 'syn')
            Mrez3D = list_to_3Dmat(idxs, vals, nChannel)
            Mrez2D = Mrez3D[:, :, trgIdx]

            # Drop rows that are
            if dropWeakChannelThr is not None:
                keepIdxs = np.max(Mrez2D, axis=0) > dropWeakChannelThr
                Mrez2DEff = Mrez2D[keepIdxs][:, keepIdxs]
            else:
                Mrez2DEff = Mrez2D

            imshow(fig, ax[0, iMouse], Mrez2D, title=row['mousename'], haveColorBar=True, cmap='jet')
            clusters = cluster_dist_matrix_min(Mrez2DEff, clusterParam, method='OPTICS')
            print(clusters)

            cluster_plot(fig, ax[1, iMouse], Mrez2DEff, clusters, limits=None, cmap='jet')

        ax[0, 0].set_ylabel('Original')
        ax[1, 0].set_ylabel('Clustered')

        fig.suptitle('_'.join(key))
        plt.show()


########################################
# Consistency
########################################

def plot_consistency_bymouse(h5fname, dfSummary, performance=None, kind='point', fisherThr=0.1, limits=None):
    pidTypes = ['unique', 'syn', 'red']
    limitKWargs = {'vmin': limits[0], 'vmax': limits[1]} if limits is not None else {}

    dfSummaryEff = dfSummary if performance is None else pd_query(dfSummary, {'performance' : performance})
    groupLst = sorted(list(set(dfSummaryEff.columns) - {'key', 'mousename', 'performance', 'trialType'}))
    dfColumns = groupLst+['consistency']

    for trialType, dfTrialType in dfSummaryEff.groupby(['trialType']):
        dfConsistencyDict = {pidType : pd.DataFrame(columns=dfColumns) for pidType in pidTypes}
        for key, dfMouse in dfTrialType.groupby(groupLst):
            fnameSuffix = '_'.join(list(key) + [trialType, str(performance)])
            mice = list(sorted(set(dfMouse['mousename'])))
            nMice = len(mice)

            for iPid, pidType in enumerate(pidTypes):
                maxRange = 0.35 if pidType == 'syn' else 1.0

                dfFilter = pd.DataFrame()
                for idx, row in dfMouse.iterrows():
                    idxs, vals = read_computed_3D(h5fname, row['key'], 'syn')
                    dfFilter[row['mousename']] = vals

                # As consistency metric perform Fischer's exact test for significant vs unsignificant links
                # As pairplot show contingency tables
                if kind == 'fisher':
                    fischerLabels = ['low', 'high']
                    rezMat = np.full((nMice, nMice), np.nan)
                    fig, ax = plt.subplots(nrows=nMice, ncols=nMice, figsize=(4 * nMice, 4 * nMice))
                    for idxMousei, iMousename in enumerate(mice):
                        ax[idxMousei, 0].set_ylabel(iMousename)
                        ax[-1, idxMousei].set_xlabel(iMousename)
                        for idxMousej, jMousename in enumerate(mice):
                            if idxMousei == idxMousej:
                                ax[idxMousei][idxMousej].hist(dfFilter[iMousename], range=[0, maxRange], bins=50)
                                ax[idxMousei][idxMousej].axvline(x=fisherThr, linestyle='--', color='pink')
                            else:
                                iBin = dfFilter[iMousename] >= fisherThr
                                jBin = dfFilter[jMousename] >= fisherThr
                                M = confusion_matrix(iBin, jBin)
                                M = M.astype(float) / np.sum(M)

                                # consistency = fisher_exact(M, alternative='two_sided')[0]
                                # consistency = -np.log10(fisher_exact(M, alternative='two_sided')[1])
                                consistency = cohen_kappa_score(iBin, jBin)
                                rezMat[idxMousei][idxMousej] = consistency

                                sns.heatmap(ax=ax[idxMousei][idxMousej], data=M, annot=True, cmap='jet',
                                            xticklabels=fischerLabels, yticklabels=fischerLabels)
                else:
                    # As consistency metric use correlation coefficient between values
                    rezMat = np.zeros((nMice, nMice))
                    for idxMousei, iMousename in enumerate(mice):
                        for idxMousej, jMousename in enumerate(mice):
                            rezMat[idxMousei][idxMousej] = np.corrcoef(dfFilter[iMousename], dfFilter[jMousename])[0, 1]

                    if kind == 'point':
                        # As pairplot use scatter
                        pPlot = sns.pairplot(data=dfFilter, vars=mice)#, kind='kde')
                    elif kind == 'heatmap':
                        # As pairplot use heatmap of binned scatter points
                        fig, ax = plt.subplots(nrows=nMice, ncols=nMice, figsize=(4*nMice, 4*nMice))

                        for idxMousei, iMousename in enumerate(mice):
                            ax[idxMousei, 0].set_ylabel(iMousename)
                            ax[-1, idxMousei].set_xlabel(iMousename)
                            for idxMousej, jMousename in enumerate(mice):
                                if idxMousei == idxMousej:
                                    ax[idxMousei][idxMousej].hist(dfFilter[iMousename], range=[0, maxRange], bins=50)
                                else:
                                    ax[idxMousei][idxMousej].hist2d(dfFilter[iMousename], dfFilter[jMousename],
                                                                    range=[[0, maxRange], [0, maxRange]], bins=[50, 50], cmap='jet')

                plt.savefig('pics/' + pidType + '_consistency_bymouse_scatter_' + fnameSuffix + '.png')
                plt.close()

                fig, ax = plt.subplots()
                sns.heatmap(ax=ax, data=rezMat, annot=True, cmap='jet', xticklabels=mice, yticklabels=mice, **limitKWargs)
                # imshow(fig, ax, rezMat, haveColorBar=True, limits=[0,1], xTicks=mice, yTicks=mice, cmap='jet')
                plt.savefig('pics/' + pidType + '_consistency_bymouse_metric_' + fnameSuffix + '.png')
                plt.close()

                avgConsistency = np.round(np.mean(offdiag_1D(rezMat)), 2)
                dfConsistencyDict[pidType] = pd_append_row(dfConsistencyDict[pidType], list(key) + [avgConsistency])

        for iPid, pidType in enumerate(pidTypes):
            fig, ax = plt.subplots()
            dfPivot = pd_pivot(dfConsistencyDict[pidType], *dfColumns)
            sns.heatmap(data=dfPivot, ax=ax, annot=True, cmap='jet', **limitKWargs)
            fig.savefig('pics/' + 'summary_' + pidType + '_consistency_metric_' + trialType + '_' + str(performance) + '.png')
            plt.close()


def plot_consistency_byphase(h5fname, dfSummary, performance=None, datatype=None,
                             kind='point', fisherThr=0.1, limits=None):

    pidTypes = ['unique', 'syn', 'red']
    limitKWargs = {'vmin': limits[0], 'vmax': limits[1]} if limits is not None else {}

    if performance is None:
        dfSummaryEff = pd_query(dfSummary, {'datatype' : datatype})
    else:
        dfSummaryEff = pd_query(dfSummary, {'datatype' : datatype, 'performance' : performance})

    dfColumns = ['mousename', 'trialtype', 'consistency']
    dfConsistencyDict = {pidType: pd.DataFrame(columns=dfColumns) for pidType in pidTypes}
    for (mousename, trialType), df1 in dfSummaryEff.groupby(['mousename', 'trialType']):
        fnameSuffix = '_'.join([mousename, datatype, trialType, str(performance)])
        phases = sorted(list(set(df1['phase'])))
        nPhase = len(phases)

        dfPhaseDict = {}
        for pidType in pidTypes:
            dfPhaseDict[pidType] = pd.DataFrame()
            for idx, row in df1.iterrows():
                idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                dfPhaseDict[pidType][row['phase']] = vals

        for iPid, pidType in enumerate(pidTypes):
            maxRange = 0.35 if pidType == 'syn' else 1.0

            # As consistency metric perform Fischer's exact test for significant vs unsignificant links
            # As pairplot show contingency tables
            if kind == 'fisher':
                fischerLabels = ['low', 'high']
                rezMat = np.full((nPhase, nPhase), np.nan)
                fig, ax = plt.subplots(nrows=nPhase, ncols=nPhase, figsize=(4 * nPhase, 4 * nPhase))
                for idxPhasei, iPhase in enumerate(phases):
                    ax[idxPhasei, 0].set_ylabel(iPhase)
                    ax[-1, idxPhasei].set_xlabel(iPhase)
                    for idxPhasej, jPhase in enumerate(phases):
                        if idxPhasei == idxPhasej:
                            ax[idxPhasei][idxPhasej].hist(dfPhaseDict[pidType][iPhase], range=[0, maxRange], bins=50)
                            ax[idxPhasei][idxPhasej].axvline(x=fisherThr, linestyle='--', color='pink')
                        else:
                            iBin = dfPhaseDict[pidType][iPhase] >= fisherThr
                            jBin = dfPhaseDict[pidType][jPhase] >= fisherThr
                            M = confusion_matrix(iBin, jBin)
                            M = M.astype(float) / np.sum(M)

                            # consistency = fisher_exact(M, alternative='two_sided')[0]
                            # consistency = -np.log10(fisher_exact(M, alternative='two_sided')[1])
                            consistency = cohen_kappa_score(iBin, jBin)
                            rezMat[idxPhasei][idxPhasej] = consistency

                            sns.heatmap(ax=ax[idxPhasei][idxPhasej], data=M, annot=True, cmap='jet',
                                        xticklabels=fischerLabels, yticklabels=fischerLabels)
            else:
                # As consistency metric use correlation coefficient between values
                rezMat = np.zeros((nPhase, nPhase))
                for idxPhasei, iPhase in enumerate(phases):
                    for idxPhasej, jPhase in enumerate(phases):
                        rezMat[idxPhasei][idxPhasej] = np.corrcoef(dfPhaseDict[pidType][iPhase],
                                                                   dfPhaseDict[pidType][jPhase])[0, 1]

                if kind == 'point':
                    # As pairplot use scatter
                    pPlot = sns.pairplot(data=dfPhaseDict[pidType], vars=phases)#, kind='kde')
                elif kind == 'heatmap':
                    # As pairplot use heatmap of binned scatter points
                    fig, ax = plt.subplots(nrows=nPhase, ncols=nPhase, figsize=(4*nPhase, 4*nPhase))

                    for idxPhasei, iPhase in enumerate(phases):
                        ax[idxPhasei, 0].set_ylabel(iPhase)
                        ax[-1, idxPhasei].set_xlabel(iPhase)
                        for idxPhasej, jPhase in enumerate(phases):
                            if idxPhasei == idxPhasej:
                                ax[idxPhasei][idxPhasej].hist(dfPhaseDict[pidType][iPhase], range=[0, maxRange], bins=50)
                            else:
                                ax[idxPhasei][idxPhasej].hist2d(dfPhaseDict[pidType][iPhase],
                                                                dfPhaseDict[pidType][jPhase],
                                                                range=[[0, maxRange], [0, maxRange]], bins=[50, 50], cmap='jet')

            plt.savefig('pics/' + pidType + '_consistency_bymouse_scatter_' + fnameSuffix + '.png')
            plt.close()

            fig, ax = plt.subplots()
            sns.heatmap(ax=ax, data=rezMat, annot=True, cmap='jet', xticklabels=phases, yticklabels=phases, **limitKWargs)
            # imshow(fig, ax, rezMat, haveColorBar=True, limits=[0,1], xTicks=mice, yTicks=mice, cmap='jet')
            plt.savefig('pics/' + pidType + '_consistency_bymouse_metric_' + fnameSuffix + '.png')
            plt.close()

            avgConsistency = np.round(np.mean(offdiag_1D(rezMat)), 2)
            dfConsistencyDict[pidType] = pd_append_row(dfConsistencyDict[pidType], [mousename, trialType, avgConsistency])

    for iPid, pidType in enumerate(pidTypes):
        fig, ax = plt.subplots()
        dfPivot = pd_pivot(dfConsistencyDict[pidType], *dfColumns)
        sns.heatmap(data=dfPivot, ax=ax, annot=True, cmap='jet', **limitKWargs)
        fig.savefig('pics/' + 'summary_' + pidType + '_consistency_metric_' + datatype + '_' + str(performance) + '.png')
        plt.close()


def plot_consistency_bytrialtype(h5fname, dfSummary, performance=None, datatype=None,
                                 trialTypes=None, kind='point', fisherThr=0.1, limits=None):

    pidTypes = ['unique', 'syn', 'red']
    limitKWargs = {'vmin': limits[0], 'vmax': limits[1]} if limits is not None else {}

    if performance is None:
        dfSummaryEff = pd_query(dfSummary, {'datatype' : datatype})
    else:
        dfSummaryEff = pd_query(dfSummary, {'datatype' : datatype, 'performance' : performance})

    dfColumns = ['mousename', 'phase', 'consistency']
    dfConsistencyDict = {pidType: pd.DataFrame(columns=dfColumns) for pidType in pidTypes}
    for (mousename, phase), df1 in dfSummaryEff.groupby(['mousename', 'phase']):
        fnameSuffix = '_'.join([mousename, datatype, phase, str(performance)])
        trialTypes = trialTypes if trialTypes is not None else sorted(list(set(df1['trialType'])))
        nTrialTypes = len(trialTypes)

        dfTrialTypeDict = {}
        for iPid, pidType in enumerate(pidTypes):
            dfTrialTypeDict[pidType] = pd.DataFrame()
            for idx, row in df1.iterrows():
                if row['trialType'] in trialTypes:
                    idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                    dfTrialTypeDict[pidType][row['trialType']] = vals

        for iPid, pidType in enumerate(pidTypes):
            maxRange = 0.35 if pidType == 'syn' else 1.0

            # As consistency metric perform Fischer's exact test for significant vs unsignificant links
            # As pairplot show contingency tables
            if kind == 'fisher':
                fischerLabels = ['low', 'high']
                rezMat = np.full((nTrialTypes, nTrialTypes), np.nan)
                fig, ax = plt.subplots(nrows=nTrialTypes, ncols=nTrialTypes, figsize=(4 * nTrialTypes, 4 * nTrialTypes))
                for idxTTi, iTT in enumerate(trialTypes):
                    ax[idxTTi, 0].set_ylabel(iTT)
                    ax[-1, idxTTi].set_xlabel(iTT)
                    for idxTTj, jTT in enumerate(trialTypes):
                        if idxTTi == idxTTj:
                            ax[idxTTi][idxTTj].hist(dfTrialTypeDict[pidType][iTT], range=[0, maxRange], bins=50)
                            ax[idxTTi][idxTTj].axvline(x=fisherThr, linestyle='--', color='pink')
                        else:
                            iBin = dfTrialTypeDict[pidType][iTT] >= fisherThr
                            jBin = dfTrialTypeDict[pidType][jTT] >= fisherThr
                            M = confusion_matrix(iBin, jBin)
                            M = M.astype(float) / np.sum(M)

                            # consistency = fisher_exact(M, alternative='two_sided')[0]
                            # consistency = -np.log10(fisher_exact(M, alternative='two_sided')[1])
                            consistency = cohen_kappa_score(iBin, jBin)
                            rezMat[idxTTi][idxTTj] = consistency

                            sns.heatmap(ax=ax[idxTTi][idxTTj], data=M, annot=True, cmap='jet',
                                        xticklabels=fischerLabels, yticklabels=fischerLabels)
            else:
                # As consistency metric use correlation coefficient between values
                rezMat = np.zeros((nTrialTypes, nTrialTypes))
                for idxTTi, iTT in enumerate(trialTypes):
                    for idxTTj, jTT in enumerate(trialTypes):
                        rezMat[idxTTi][idxTTj] = np.corrcoef(dfTrialTypeDict[pidType][iTT],
                                                             dfTrialTypeDict[pidType][jTT])[0, 1]

                if kind == 'point':
                    # As pairplot use scatter
                    pPlot = sns.pairplot(data=dfTrialTypeDict[pidType], vars=trialTypes)#, kind='kde')
                elif kind == 'heatmap':
                    # As pairplot use heatmap of binned scatter points
                    fig, ax = plt.subplots(nrows=nTrialTypes, ncols=nTrialTypes, figsize=(4*nTrialTypes, 4*nTrialTypes))

                    for idxTTi, iTT in enumerate(trialTypes):
                        ax[idxTTi, 0].set_ylabel(iTT)
                        ax[-1, idxTTi].set_xlabel(iTT)
                        for idxTTj, jTT in enumerate(trialTypes):
                            if idxTTi == idxTTj:
                                ax[idxTTi][idxTTj].hist(dfTrialTypeDict[pidType][iTT], range=[0, maxRange], bins=50)
                            else:
                                ax[idxTTi][idxTTj].hist2d(dfTrialTypeDict[pidType][iTT],
                                                          dfTrialTypeDict[pidType][jTT],
                                                          range=[[0, maxRange], [0, maxRange]], bins=[50, 50], cmap='jet')

            plt.savefig('pics/' + pidType + '_consistency_bymouse_scatter_' + fnameSuffix + '.png')
            plt.close()

            fig, ax = plt.subplots()
            sns.heatmap(ax=ax, data=rezMat, annot=True, cmap='jet', xticklabels=trialTypes, yticklabels=trialTypes, **limitKWargs)
            # imshow(fig, ax, rezMat, haveColorBar=True, limits=[0,1], xTicks=mice, yTicks=mice, cmap='jet')
            plt.savefig('pics/' + pidType + '_consistency_bymouse_metric_' + fnameSuffix + '.png')
            plt.close()

            avgConsistency = np.round(np.mean(offdiag_1D(rezMat)), 2)
            dfConsistencyDict[pidType] = pd_append_row(dfConsistencyDict[pidType], [mousename, phase, avgConsistency])

    for iPid, pidType in enumerate(pidTypes):
        fig, ax = plt.subplots()
        dfPivot = pd_pivot(dfConsistencyDict[pidType], *dfColumns)
        sns.heatmap(data=dfPivot, ax=ax, annot=True, cmap='jet', **limitKWargs)
        fig.savefig('pics/' + 'summary_' + pidType + '_consistency_metric_' + datatype + '_' + str(performance) + '.png')
        plt.close()