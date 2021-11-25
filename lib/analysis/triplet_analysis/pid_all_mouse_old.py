import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import mannwhitneyu
from sklearn.metrics import cohen_kappa_score

from mesostat.stat.stat import continuous_empirical_CDF
from mesostat.utils.matrix import offdiag_1D
from mesostat.stat.classification import confusion_matrix
from mesostat.stat.clustering import cluster_dist_matrix_min, cluster_plot

from mesostat.utils.pandas_helper import pd_query, pd_merge_multiple, pd_is_one_row, pd_append_row, pd_pivot
from mesostat.visualization.mpl_barplot import barplot_stacked_indexed, barplot_labeled
from mesostat.visualization.mpl_matrix import imshow

from lib.analysis.triplet_analysis.pid_all_session_old import preprocess_unique, preprocess_drop_negative, preprocess_drop_channels


def pid_all_parse_key(key):
    lst = key.split('_')
    rez = {
        'mousename': '_'.join(lst[1:3]),
        'datatype': '_'.join(lst[3:5]),
        'phase': lst[5],
        'trialType': lst[6]
    }

    if len(lst) == 7:
        return rez
    elif len(lst) == 8:
        rez['performance'] = lst[7]
        return rez
    else:
        raise ValueError('Unexpected key', key)


def pid_all_parse_key_random(key):
    lst = key.split('_')
    rez = {
        'mousename': '_'.join(lst[1:3]),
        'trialType': lst[3]
    }

    if len(lst) == 4:
        return rez
    elif len(lst) == 5:
        rez['performance'] = lst[4]
        return rez
    else:
        raise ValueError('Unexpected key', key)


def pid_all_summary_df(h5fname, parserName='Orig'):
    parser = pid_all_parse_key if parserName == 'Orig' else pid_all_parse_key_random

    with h5py.File(h5fname, 'r') as f:
        keys = list(f.keys())

    summaryDF = pd.DataFrame()
    for key in keys:
        summaryDF = summaryDF.append(pd.DataFrame({**{'key': key}, **parser(key)}, index=[0]))

    return summaryDF.reset_index(drop=True)


def read_parse_joint_dataframe(dfSummaryByMouse, h5fname, mice, pidTypes, dropChannels=None):
    dfDict = defaultdict(list)
    for idx, row in dfSummaryByMouse.sort_values('mousename', axis=0).iterrows():
        # Read dataframe
        dfData = pd.read_hdf(h5fname, row['key'])

        # Preprocess dataframe
        dfData = preprocess_unique(dfData)
        dfData = preprocess_drop_negative(dfData)
        if dropChannels is not None:
            dfData = preprocess_drop_channels(dfData, dropChannels)

        # Filter data by PID type
        for pidType in pidTypes:
            dfDict[pidType] += [dfData[dfData['PID'] == pidType].drop(['PID', 'p', 'effSize', 'muRand'], axis=1)]

    # Merge data from different mice onto the same dataframe
    dfJointDict = {}
    for iPid, pidType in enumerate(pidTypes):
        dfJointDict[pidType] = pd_merge_multiple(mice, dfDict[pidType], ["S1", "S2", "T"])
        print(pidType, len(dfDict[pidType][0]), len(dfJointDict[pidType]))

    return dfJointDict


def _bitdict_to_3Dmat(dataDB, mouseBitDict, pidType, mice):
    labels = dataDB.get_channel_labels()
    labelDict = {l: i for i, l in enumerate(labels)}
    nChannel = len(labels)

    dfThis = mouseBitDict[pidType].copy()

    # Rename channels back to indices
    # df.replace({'S1': labelDict, 'S2': labelDict}, inplace=True)
    dfThis.replace({'S1': labelDict, 'S2': labelDict, 'T': labelDict}, inplace=True)

    Mrez = np.zeros((nChannel, nChannel, nChannel))
    for mousename in mice:
        # Construct as matrix
        M = np.zeros((nChannel, nChannel, nChannel))
        M[dfThis['S1'], dfThis['S2'], dfThis['T']] = dfThis['muTrue_'+mousename]

        if pidType != 'unique':
            # M += M.T
            M += M.transpose((1, 0, 2))
        Mrez += M

    return Mrez / len(mice)


def cdfplot(h5fname, dfSummary):
    pidTypes = ['unique', 'syn', 'red']
    dataLabels = ['p', 'effSize', 'muTrue']
    haveLog = [True, False, False]

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
        fig.suptitle('_'.join(key))

        for idx, row in dataMouse.iterrows():
            dfRezThis = pd.read_hdf(h5fname, row['key'])

            # Merge Unique1 and Unique2 for this plot
            dfRezThis.replace({'PID': {'U1': 'unique', 'U2': 'unique'}}, inplace=True)

            for iPid, pidType in enumerate(pidTypes):
                df1 = dfRezThis[dfRezThis['PID'] == pidType]

                for iLabel, label in enumerate(dataLabels):
                    x, y = continuous_empirical_CDF(df1[label])
                    ax[iPid][iLabel].plot(x, y, label=row['mousename'])

        for i in range(3):
            ax[i][0].set_ylabel(pidTypes[i])
            ax[-1][i].set_xlabel(dataLabels[i])
            for j in range(3):
                ax[i][j].legend()

                if haveLog[j]:
                    ax[i][j].set_xscale('log')

        plt.show()


def test_avg_bits(dataDB, mc, h5fname, h5fnameRand, dfSummary, dfSummaryRand):
    channelLabels = dataDB.get_channel_labels()
    nChannel = len(channelLabels)
    pidTypes = ['unique', 'syn', 'red']

    '''
    Plan:
    1. Loop over (col / key, mousename) in dfSummaryRand
    2. Query dfSummary using rand key
    3. Loop over (col / key, mousename) in dfSummaryQueried
    4. Loop over mousename in dfSummaryRand
    5. Query dfSummaryQueried using mousename
    6. Extract datasets
    7. Barplot+Statannot
    '''

    groupLstRand = sorted(list(set(dfSummaryRand.columns) - {'key', 'mousename'}))
    print(set(dfSummaryRand['performance']))

    print(groupLstRand)
    for keyRand, dfRandMouse in dfSummaryRand.groupby(groupLstRand):
        print(keyRand)
        if isinstance(keyRand, str):
            keyRand = [keyRand]

        selectorLstRand = dict(zip(groupLstRand, keyRand))
        dfSummQueried = pd_query(dfSummary, selectorLstRand)

        groupLstQueried = sorted(list(set(dfSummQueried.columns) - {'key', 'mousename'} - set(groupLstRand)))
        for key, dfMouse in dfSummQueried.groupby(groupLstQueried):
            print('--', key)

            dfTot = pd.DataFrame()
            for idx, row in dfMouse.iterrows():
                # Read and preprocess true data
                dfRezTrue = pd.read_hdf(h5fname, row['key'])
                dfRezTrue = preprocess_unique(dfRezTrue)
                dfRezTrue = preprocess_drop_negative(dfRezTrue)
                dfRezTrue['type'] = 'Measured'
                dfRezTrue['mousename'] = row['mousename']
                dfTot = dfTot.append(dfRezTrue)

                # Read and preprocess random data
                rowRand = pd_is_one_row(pd_query(dfRandMouse, {'mousename' : row['mousename']}))[1]
                dfRezRand = pd.read_hdf(h5fnameRand, rowRand['key'])
                dfRezRand = preprocess_unique(dfRezRand)
                dfRezRand = preprocess_drop_negative(dfRezRand)
                dfRezRand['type'] = 'Shuffle'
                dfRezRand['mousename'] = rowRand['mousename']
                dfTot = dfTot.append(dfRezRand)

            # Barplot differences
            fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
            fig.suptitle('_'.join(list(key) + list(keyRand)))
            for iPid, pidType in enumerate(pidTypes):
                dfPID = dfTot[dfTot['PID'] == pidType]
                sns.violinplot(ax=ax[iPid], x="mousename", y="muTrue", hue="type", data=dfPID, scale='width', cut=0)

                for mousename in sorted(set(dfPID['mousename'])):
                    dataTrue = pd_query(dfPID, {'mousename' : mousename, 'type' : 'Measured'})['muTrue']
                    dataRand = pd_query(dfPID, {'mousename': mousename, 'type' : 'Shuffle'})['muTrue']
                    print('Test:', pidType, mousename, 'pval =', mannwhitneyu(dataTrue, dataRand, alternative='greater')[1])

                ax[iPid].set_yscale('log')
                ax[iPid].set_ylabel('Bits')
                ax[iPid].set_title(pidType)
            plt.show()


def scatter_effsize_bits(h5fname, dfSummary):
    pidTypes = ['unique', 'syn', 'red']
    # dataLabels = ['p', 'effSize', 'muTrue']
    # haveLog = [True, False, False]

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):
        fig, ax = plt.subplots(ncols=3, figsize=(12, 4), tight_layout=True)
        fig.suptitle('_'.join(key))

        for idx, row in dataMouse.iterrows():
            dfRezThis = pd.read_hdf(h5fname, row['key'])

            # Merge Unique1 and Unique2 for this plot
            dfRezThis = preprocess_unique(dfRezThis)
            dfRezThis = preprocess_drop_negative(dfRezThis)

            for iPid, pidType in enumerate(pidTypes):
                df1 = dfRezThis[dfRezThis['PID'] == pidType]

                ax[iPid].plot(df1['effSize'], df1['muTrue'], '.', label=row['mousename'])

        for i in range(3):
            ax[i].set_title(pidTypes[i])
            ax[i].set_xlabel('effSize')
            ax[i].set_ylabel('Bits')
            ax[i].legend()

        plt.show()


def plot_triplets(dataDB, h5fname, dfSummary, nTop=20, dropChannels=None):
    lmap = dataDB.map_channel_labels_canon()

    pidTypes = ['unique', 'syn', 'red']

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):
        # fig, ax = plt.subplots(ncols=3, figsize=(12, 4), tight_layout=True)
        # fig.suptitle('_'.join(key))

        mice = list(sorted(set(dataMouse['mousename'])))
        dfJointDict = read_parse_joint_dataframe(dataMouse, h5fname, mice, pidTypes, dropChannels=dropChannels)

        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        for iPid, pidType in enumerate(pidTypes):
            dfJoint = dfJointDict[pidType]
            # meanColNames = list(set(dfJoint.columns) - set(dfJointDict[pidType].keys()))
            meanColNames = list(set(dfJoint.columns) - {'S1', 'S2', 'T'})

            dfJoint['bits_mean'] = dfJoint[meanColNames].mean(axis=1)

            dfJoint = dfJoint.sort_values('bits_mean', axis=0, ascending=False)
            dfJointHead = dfJoint.head(nTop)

            labels = [str((lmap[s1],lmap[s2],lmap[t])) for s1,s2,t in zip(dfJointHead['S1'], dfJointHead['S2'], dfJointHead['T'])]
            rezDict = {mousename : np.array(dfJointHead['muTrue_' + mousename]) for mousename in mice}

            # 4) Plot stacked barplot with absolute numbers. Set ylim_max to total number of sessions
            barplot_stacked_indexed(ax[iPid], rezDict, xTickLabels=labels, xLabel='triplet',
                                    yLabel='bits', title=pidType, iMax=None, rotation=90)

        fig.suptitle('_'.join(key))
        plt.show()


def plot_singlets(dataDB, h5fname, dfSummary, nTop=20, dropChannels=None):
    lmap = dataDB.map_channel_labels_canon()
    dataLabels = dataDB.get_channel_labels()
    dataLabelsCanon = [lmap[l] for l in dataLabels]

    pidTypes = ['unique', 'syn', 'red']

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):
        mice = list(sorted(set(dataMouse['mousename'])))
        dfJointDict = read_parse_joint_dataframe(dataMouse, h5fname, mice, pidTypes, dropChannels=dropChannels)

        fig, ax = plt.subplots(nrows=len(pidTypes), figsize=(len(pidTypes) * 6, 12), tight_layout=True)
        for iPid, pidType in enumerate(pidTypes):
            rezDict = {}
            for mousename in mice:
                rezTmp = []
                for label in dataLabels:
                    rezTmp += [np.mean(pd_query(dfJointDict[pidType], {'T': label})['muTrue_' + mousename])]
                rezDict[mousename] = rezTmp

            barplot_stacked_indexed(ax[iPid], rezDict, xTickLabels=dataLabelsCanon, xLabel='singlet',
                                    yLabel='bits', title=pidType, iMax=None, rotation=90)

        fig.suptitle('_'.join(key))
        plt.show()


def plot_unique_top_pairs(dataDB, h5fname, dfSummary, nTop=20, dropChannels=None):
    lmap = dataDB.map_channel_labels_canon()
    dataLabels = dataDB.get_channel_labels()
    dataLabelsCanon = [lmap[l] for l in dataLabels]

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):
        mice = list(sorted(set(dataMouse['mousename'])))
        dfJointDict = read_parse_joint_dataframe(dataMouse, h5fname, mice, ['unique'],
                                                   dropChannels=dropChannels)

        fig, ax = plt.subplots(nrows=2, ncols=len(mice), figsize=(6*len(mice), 8))
        for iMouse, mousename in enumerate(mice):
            Mrez3D = _bitdict_to_3Dmat(dataDB, dfJointDict, 'unique', [mousename])
            nChannel = Mrez3D.shape[0]
            Mrez2D = np.sum(Mrez3D, axis=1) / (nChannel - 2)  # Target can't be either of the sources

            imshow(fig, ax[0][iMouse], Mrez2D, title=mousename, ylabel='Unique2D', haveColorBar=True, cmap='jet')

            # Find 2D indices of nTop strongest links
            vals1D = Mrez2D.flatten()
            idxsMax1D = np.argsort(vals1D)[::-1][:2*nTop]
            idxsMax2D = np.vstack([idxsMax1D // nChannel, idxsMax1D % nChannel]).T
            valsMax1D = vals1D[idxsMax1D]
            labelsMax2D = [(dataLabelsCanon[i], dataLabelsCanon[j]) for i,j in idxsMax2D]

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


def plot_2D_avg(dataDB, h5fname, dfSummary, dropChannels=None, avgAxis=2):
    pidTypes = ['unique', 'syn', 'red']

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):
        mice = list(sorted(set(dataMouse['mousename'])))
        dfJointDict = read_parse_joint_dataframe(dataMouse, h5fname, mice, pidTypes, dropChannels=dropChannels)

        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        for iPid, pidType in enumerate(pidTypes):
            Mrez3D = _bitdict_to_3Dmat(dataDB, dfJointDict, pidType, mice)

            nChannel = Mrez3D.shape[0]
            Mrez2D = np.sum(Mrez3D, axis=avgAxis) / (nChannel-2)   # Target can't be either of the sources

            print(np.max(Mrez2D))

            imshow(fig, ax[iPid], Mrez2D, title=pidType, haveColorBar=True, cmap='jet')

        fig.suptitle('_'.join(key))
        plt.show()


def plot_2D_bytarget(dataDB, h5fname, dfSummary, trgChName, dropChannels=None):
    pidTypes = ['unique', 'syn', 'red']
    labels = dataDB.get_channel_labels()
    trgIdx = labels.index(trgChName)

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):
        mice = list(sorted(set(dataMouse['mousename'])))
        dfJointDict = read_parse_joint_dataframe(dataMouse, h5fname, mice, pidTypes, dropChannels=dropChannels)

        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        for iPid, pidType in enumerate(pidTypes):
            Mrez3D = _bitdict_to_3Dmat(dataDB, dfJointDict, pidType, mice)

            nChannel = Mrez3D.shape[0]
            Mrez2D = Mrez3D[:, :, trgIdx]

            print(np.max(Mrez2D))

            imshow(fig, ax[iPid], Mrez2D, title=pidType, haveColorBar=True, cmap='jet')

        fig.suptitle('_'.join(key))
        plt.show()


def plot_2D_bytarget_synergy_cluster(dataDB, h5fname, dfSummary, trgChName, dropChannels=None,
                                     clusterParam=1.0, dropWeakChannelThr=None):
    labels = dataDB.get_channel_labels()
    trgIdx = labels.index(trgChName)

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):
        mice = list(sorted(set(dataMouse['mousename'])))
        dfJointDict = read_parse_joint_dataframe(dataMouse, h5fname, mice, ['syn'], dropChannels=dropChannels)

        nMice = len(mice)
        fig, ax = plt.subplots(nrows=2, ncols=nMice, figsize=(4*nMice, 8))

        for iMouse, (idx, row) in enumerate(dataMouse.iterrows()):
            Mrez3D = _bitdict_to_3Dmat(dataDB, dfJointDict, 'syn', [row['mousename']])
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

def plot_consistency_bymouse(h5fname, dfSummary, dropChannels=None, performance=None, kind='point',
                             fisherThr=0.1, limits=None):

    pidTypes = ['unique', 'syn', 'red']
    limitKWargs = {'vmin': limits[0], 'vmax': limits[1]} if limits is not None else {}

    dfSummaryEff = dfSummary if performance is None else pd_query(dfSummary, {'performance' : performance})
    groupLst = sorted(list(set(dfSummaryEff.columns) - {'key', 'mousename', 'performance', 'trialType'}))
    dfColumns = groupLst+['consistency']

    for trialType, dfTrialType in dfSummaryEff.groupby(['trialType']):
        dfConsistencyDict = {pidType : pd.DataFrame(columns=dfColumns) for pidType in pidTypes}
        for key, dataMouse in dfTrialType.groupby(groupLst):
            fnameSuffix = '_'.join(list(key) + [trialType, str(performance)])
            mice = list(sorted(set(dataMouse['mousename'])))
            nMice = len(mice)
            dfJointDict = read_parse_joint_dataframe(dataMouse, h5fname, mice, pidTypes, dropChannels=dropChannels)

            for iPid, pidType in enumerate(pidTypes):
                maxRange = 0.35 if pidType == 'syn' else 1.0

                dfFilter = pd.DataFrame()
                for mousename in mice:
                    dfFilter[mousename] = dfJointDict[pidType]['muTrue_'+mousename]

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


def plot_consistency_byphase(h5fname, dfSummary, dropChannels=None, performance=None, datatype=None,
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
        for iPid, pidType in enumerate(pidTypes):
            dfPhaseDict[pidType] = pd.DataFrame()

        for phase, dfPhase in df1.groupby(['phase']):
            dfTmp = read_parse_joint_dataframe(dfPhase, h5fname, [mousename], pidTypes, dropChannels=dropChannels)
            for iPid, pidType in enumerate(pidTypes):
                dfPhaseDict[pidType][phase] = dfTmp[pidType]['muTrue_'+mousename]

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


def plot_consistency_bytrialtype(h5fname, dfSummary, dropChannels=None, performance=None, datatype=None,
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

        for trialType, dfTrialType in df1.groupby(['trialType']):
            if trialType in trialTypes:
                dfTmp = read_parse_joint_dataframe(dfTrialType, h5fname, [mousename], pidTypes, dropChannels=dropChannels)
                for iPid, pidType in enumerate(pidTypes):
                    dfTrialTypeDict[pidType][trialType] = dfTmp[pidType]['muTrue_'+mousename]

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