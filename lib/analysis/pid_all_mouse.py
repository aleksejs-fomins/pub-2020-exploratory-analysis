import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, fisher_exact
from sklearn.metrics import cohen_kappa_score

from IPython.display import display

from mesostat.utils.system import make_path
from mesostat.utils.matrix import drop_channels, offdiag_1D, matrix_copy_triangle_symmetric
from mesostat.stat.stat import continuous_empirical_CDF
from mesostat.stat.classification import confusion_matrix
from mesostat.stat.clustering import cluster_dist_matrix_min, cluster_plot

from mesostat.utils.pandas_helper import pd_query, pd_merge_multiple, pd_is_one_row, pd_append_row, pd_pivot
from mesostat.visualization.mpl_barplot import barplot_stacked_indexed, barplot_labeled, sns_barplot
from mesostat.visualization.mpl_matrix import imshow


def _vals_to_df(vals, valKey, keyDict):
    df = pd.DataFrame()
    df[valKey] = vals
    for k, v in keyDict.items():
        df[k] = v
    return df


# Compute average PID over two out of three triplets
def mean_vals_axis(idxs, vals2D, axis, nChannel):
    return np.array([np.mean(vals2D[idxs[:, axis] == iCh]) for iCh in range(nChannel)])


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
        return np.concatenate([labelsU1, labelsU2], axis=0), np.concatenate([valsU1, valsU2], axis=0)
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


def plot_cdf(h5fname, dfSummary, fontsize=20):
    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):
        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))

        for iPID, pidType in enumerate(['unique', 'syn', 'red']):
            for idx, row in dataMouse.iterrows():
                idxs, vals = read_computed_3D(h5fname, row['key'], pidType=pidType)
                x, y = continuous_empirical_CDF(vals)

                ax[iPID].plot(x, y, label=row['mousename'])
                ax[iPID].set_title(pidType, fontsize=fontsize)

            ax[iPID].legend()

        prefixPath = 'pics/pid_avg/pid_cdf/'
        make_path(prefixPath)
        plt.savefig(prefixPath + 'PID_cdf_'+'_'.join(key)+'.png')
        plt.close()


def plot_violin_test(h5fname, h5fnameRand, dfSummary, dfSummaryRand, fontsize=20):
    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):
        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))

        for iPID, pidType in enumerate(['unique', 'syn', 'red']):

            dfPID = pd.DataFrame()

            for idx, row in dataMouse.iterrows():
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
                dfPID = dfPID.append(dfTrue)
                dfPID = dfPID.append(dfRand)

            sns.violinplot(ax=ax[iPID], x="mousename", y=pidType, hue="type", data=dfPID, scale='width', cut=0)
            ax[iPID].set_yscale('log')
            ax[iPID].set_title(pidType, fontsize=fontsize)

        prefixPath = 'pics/pid_avg/pid_test_violin/'
        make_path(prefixPath)
        plt.savefig(prefixPath + 'pics/PID_violin_'+'_'.join(key) + '.png')
        plt.close()


def barplot_avg(dataDB, h5fname, dfSummary, paramName, paramVals, fontsize=20):
    mice = sorted(dataDB.mice)

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename', paramName}))
    for groupVals, dataTmp in dfSummary.groupby(groupLst):
        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))

        for iPID, pidType in enumerate(['unique', 'syn', 'red']):
            rezDF = pd.DataFrame()

            for mousename, dataMouse in dataTmp.groupby(['mousename']):
                for paramNameThis, dfParam in dataMouse.groupby([paramName]):
                    idx, row = pd_is_one_row(dfParam)
                    idxs, vals = read_computed_3D(h5fname, row['key'], pidType=pidType)

                    keyDict = {'mousename': mousename, paramName: row[paramName], 'PID': pidType}
                    dfTmp = _vals_to_df(vals, 'bits', keyDict)
                    rezDF = rezDF.append(dfTmp)

            sns_barplot(ax[iPID], rezDF, "mousename", 'bits', paramName, annotHue=False, xOrd=mice, hOrd=paramVals)
            ax[iPID].set_title(pidType, fontsize=fontsize)

        prefixPath = 'pics/pid_avg/pid_avg_barplot_' + paramName + '/'
        make_path(prefixPath)
        plt.savefig(prefixPath + 'PID_barplot_' + paramName + '_' + '_'.join(groupVals) + '.png')
        plt.close()


def plot_top_triplets(dataDB, h5fname, dfSummary, nTop=20, fontsize=20):
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    pidTypes = ['unique', 'syn', 'red']

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):
        print(key)

        fig, ax = plt.subplots(ncols=3, figsize=(12, 4), tight_layout=True)
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
                                    yLabel='bits', title=pidType, iMax=None, rotation=90, fontsize=fontsize)

        prefixPath = 'pics/pid_3D/triplets_barplot/'
        make_path(prefixPath)
        plt.savefig(prefixPath + 'pid_triplets_barplot' + '_'.join(key) + '.png', dpi=300)
        plt.close()


def plot_top_singlets(dataDB, h5fname, dfSummary, fontsize=20):
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    nChannel = len(labelsCanon)
    pidTypes = ['unique', 'syn', 'red']

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):
        print(key)

        fig, ax = plt.subplots(ncols=3, figsize=(12, 4), tight_layout=True)
        for iPid, pidType in enumerate(pidTypes):
            ax[iPid].set_title(pidType)

            rezDict = {}
            for idx, row in dataMouse.iterrows():
                idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                rezDict[row['mousename']] = mean_vals_axis(idxs, vals, 2, nChannel)

            # Plot stacked barplot with absolute numbers. Set ylim_max to total number of sessions
            barplot_stacked_indexed(ax[iPid], rezDict, xTickLabels=labelsCanon, xLabel='singlet',
                                    yLabel='bits', title=pidType, iMax=None, rotation=90, fontsize=fontsize)

        prefixPath = 'pics/pid_1D/pid_singlets_top/'
        make_path(prefixPath)
        plt.savefig(prefixPath + 'pid_singlets_barplot' + '_'.join(key) + '.png', dpi=300)
        plt.close()


def plot_singlets_brainplot(dataDB, h5fname, dfSummary, paramKey, paramNames, fontsize=20):
    pidTypes = ['unique', 'syn', 'red']
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    nChannel = len(labelsCanon)
    nParam = len(paramNames)
    mice = sorted(dataDB.mice)
    nMice = len(mice)

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename', paramKey}))
    for key, dataTmp in dfSummary.groupby(groupLst):
        for pidType in pidTypes:
            vmax = 1.0 if pidType == 'red' else 0.25

            fig, ax = plt.subplots(nrows=nMice, ncols=nParam, figsize=(4 * nParam, 4 * nMice))
            for iMouse, mousename in enumerate(mice):
                ax[iMouse, 0].set_ylabel(mousename, fontsize=fontsize)

                for iParam, paramName in enumerate(paramNames):
                    ax[0, iParam].set_title(paramName, fontsize=fontsize)

                    idx, row = pd_is_one_row(pd_query(dataTmp, {'mousename': mousename, paramKey: paramName}))
                    if row is not None:
                        idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                        valsMeanTrg = mean_vals_axis(idxs, vals, 2, nChannel)

                        haveColorBar = iParam == nParam - 1
                        dataDB.plot_area_values(fig, ax[iMouse][iParam], valsMeanTrg,
                                                vmin=0, vmax=vmax, cmap='jet', haveColorBar=haveColorBar)

            prefixPath = 'pics/pid_1D/pid_singlets_brainplot_' + paramKey + '/' + pidType + '/'
            make_path(prefixPath)
            plotSuffix = '_'.join(list(key) + [pidType])
            plt.savefig(prefixPath + 'pid_brainplot_signlets_mouse' + paramKey + '_' + plotSuffix + '.png')
            plt.close()


def plot_singlets_brainplot_mousephase_subpre(dataDB, h5fname, dfSummary, fontsize=20):
    pidTypes = ['unique', 'syn', 'red']
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    nChannel = len(labelsCanon)
    intervNames = dataDB.get_interval_names()
    nInterv = len(intervNames)
    mice = sorted(dataDB.mice)
    nMice = len(mice)

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename', 'phase'}))
    dfBnSession = pd_query(dfSummary, {'datatype': 'bn_session'})

    for key, dataTmp in dfBnSession.groupby(groupLst):
        for pidType in pidTypes:
            vmax = 1.0 if pidType == 'red' else 0.25

            fig, ax = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4*nInterv, 4*nMice))
            for iMouse, mousename in enumerate(mice):
                ax[iMouse, 0].set_ylabel(mousename, fontsize=fontsize)

                rezDict = {}
                for iInterv, intervName in enumerate(intervNames):
                    ax[0, iInterv].set_title(intervName, fontsize=fontsize)

                    idx, row = pd_is_one_row(pd_query(dataTmp, {'mousename': mousename, 'phase': intervName}))
                    if row is not None:
                        idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                        rezDict[intervName] = mean_vals_axis(idxs, vals, 2, nChannel)

                for iInterv, intervName in enumerate(intervNames):
                    if (intervName != 'PRE') and (intervName in rezDict.keys()):
                        haveColorBar = iInterv == nInterv - 1
                        dataDB.plot_area_values(fig, ax[iMouse][iInterv], rezDict[intervName] - rezDict['PRE'],
                                                vmin=-vmax, vmax=vmax, cmap='jet', haveColorBar=haveColorBar)

            prefixPath = 'pics/pid_1D/pid_singlets_brainplot_subpre/' + pidType + '/'
            make_path(prefixPath)
            plotSuffix = '_'.join(list(key) + [pidType])
            plt.savefig(prefixPath + 'pid_brainplot_signlets_mousephase_subpre_' + '_' + plotSuffix + '.png')
            plt.close()


def plot_singlets_brainplot_mousephase_submouse(dataDB, h5fname, dfSummary, fontsize=20):
    pidTypes = ['unique', 'syn', 'red']
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    nChannel = len(labelsCanon)
    intervNames = dataDB.get_interval_names()
    nInterv = len(intervNames)
    mice = sorted(dataDB.mice)
    nMice = len(mice)

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename', 'phase'}))
    for key, dataTmp in dfSummary.groupby(groupLst):
        for pidType in pidTypes:
            vmax = 1.0 if pidType == 'red' else 0.25

            fig, ax = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4*nInterv, 4*nMice))
            for iInterv, intervName in enumerate(intervNames):
                ax[0, iInterv].set_title(intervName, fontsize=fontsize)

                rezDict = {}
                for iMouse, mousename in enumerate(mice):
                    ax[iMouse, 0].set_ylabel(mousename, fontsize=fontsize)

                    idx, row = pd_is_one_row(pd_query(dataTmp, {'mousename': mousename, 'phase': intervName}))
                    if row is not None:
                        idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                        rezDict[mousename] = mean_vals_axis(idxs, vals, 2, nChannel)

                rezMean = np.mean(list(rezDict.values()), axis=0)
                for iMouse, mousename in enumerate(mice):
                    if mousename in rezDict.keys():
                        haveColorBar = iMouse == nMice - 1
                        dataDB.plot_area_values(fig, ax[iMouse][iInterv], rezDict[mousename] - rezMean,
                                                vmin=-vmax, vmax=vmax, cmap='jet', haveColorBar=haveColorBar)

            prefixPath = 'pics/pid_1D/pid_singlets_brainplot_submouse/' + pidType + '/'
            make_path(prefixPath)
            plotSuffix = '_'.join(list(key) + [pidType])
            plt.savefig('pid_brainplot_signlets_mousephase_submouse_' + '_' + plotSuffix + '.png')
            plt.close()


def plot_singlets_barplot_2DF(dataDB1, dataDB2, labelDB1, labelDB2, h5fname1, h5fname2, dfSummary1, dfSummary2,
                              intervNameMap, intervOrdMap, fontsize=20):
    pidTypes = ['unique', 'syn', 'red']
    labelsCanon = list(dataDB1.map_channel_labels_canon().values())
    nChannel = len(labelsCanon)
    mice = sorted(dataDB1.mice)
    nMice = len(mice)

    groupLst1 = sorted(list(set(dfSummary1.columns) - {'key', 'mousename'}))
    for key, dataTmp in dfSummary1.groupby(groupLst1):
        for iPid, pidType in enumerate(pidTypes):
            fig, ax = plt.subplots(nrows=2, ncols=nMice, figsize=(4 * nMice, 8))
            ax[0, 0].set_ylabel(labelDB1)
            ax[1, 0].set_ylabel(labelDB2)

            plotSuffix = '_'.join([pidType] + list(key))
            print(plotSuffix)

            for idx1, row1 in dataTmp.iterrows():
                iMouse = mice.index(row1['mousename'])
                ax[0, iMouse].set_title(row1['mousename'], fontsize=fontsize)

                # Replace interval name with analog from the other dataset
                queryDict2 = dict(row1)
                intervName2 = queryDict2['phase']
                intervName2 = intervName2 if (labelDB2, intervName2) not in intervOrdMap else intervOrdMap[(labelDB2, intervName2)]
                queryDict2['phase'] = intervName2

                idx2, row2 = pd_is_one_row(pd_query(dfSummary2, queryDict2))
                assert row2 is not None

                idxs1, vals1 = read_computed_3D(h5fname1, row1['key'], pidType)
                idxs2, vals2 = read_computed_3D(h5fname2, row2['key'], pidType)
                valsMeanTrg1 = mean_vals_axis(idxs1, vals1, 2, nChannel)
                valsMeanTrg2 = mean_vals_axis(idxs2, vals2, 2, nChannel)

                haveColorBar = iMouse == nMice - 1
                vmax = 1.0 if pidType == 'red' else 0.25
                dataDB1.plot_area_values(fig, ax[0][iMouse], valsMeanTrg1,
                                         vmin=0, vmax=vmax, cmap='jet', haveColorBar=haveColorBar)
                dataDB2.plot_area_values(fig, ax[1][iMouse], valsMeanTrg2,
                                         vmin=0, vmax=vmax, cmap='jet', haveColorBar=haveColorBar)

            plt.savefig('pid_brainplot_signlets_bymodality_' + plotSuffix + '.png')
            plt.close()


# FIXME: Averaging is not aware of dropped channels
def plot_unique_top_pairs(dataDB, h5fname, dfSummary, nTop=20, fontsize=20):
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    nChannel = len(labelsCanon)
    mice = dataDB.mice
    pidTypes = ['unique', 'syn', 'red']

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename'}))
    for key, dataMouse in dfSummary.groupby(groupLst):

        fig, ax = plt.subplots(nrows=2, ncols=len(mice), figsize=(6*len(mice), 8))
        for iPid, pidType in enumerate(pidTypes):
            ax[iPid].set_title(pidType, fontsize=fontsize)

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
def plot_2D_avg(dataDB, h5fname, dfSummary, paramKey, paramNames, dropChannels=None, avgAxis=2, fontsize=20):
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    nChannel = len(labelsCanon)
    pidTypes = ['unique', 'syn', 'red']

    mice = sorted(dataDB.mice)
    nMice = len(mice)
    nParam = len(paramNames)

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename', paramKey}))
    for key, dataTmp in dfSummary.groupby(groupLst):
        for pidType in pidTypes:
            fig, ax = plt.subplots(nrows=nMice, ncols=nParam, figsize=(4*nParam, 4*nMice))
            for iMouse, mousename in enumerate(mice):
                ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

                for iParam, paramName in enumerate(paramNames):
                    ax[0][iParam].set_title(paramName, fontsize=fontsize)

                    idx, row = pd_is_one_row(pd_query(dataTmp, {'mousename': mousename, paramKey: paramName}))
                    if row is not None:
                        idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                        Mrez3D = list_to_3Dmat(idxs, vals, nChannel)
                        Mrez2D = np.nansum(Mrez3D, axis=avgAxis) / (nChannel - 2)  # Target can't be either of the sources
                        Mrez2D = drop_channels(Mrez2D, dropIdxs=dropChannels)

                        if pidType != 'unique':
                            Mrez2D = matrix_copy_triangle_symmetric(Mrez2D, source='U')

                        vmax = 1.0 if pidType == 'red' else 0.5
                        imshow(fig, ax[iMouse][iParam], Mrez2D, cmap='jet',
                               haveColorBar=iParam == nParam - 1, limits=[0, vmax])

            prefixPath = 'pics/pid_2D/pid_2D_avg_' + paramKey + '/' + pidType + '/'
            make_path(prefixPath)
            pltSuffix = '_'.join([paramKey] + list(key) + [pidType])
            plt.savefig(prefixPath + 'pid_2D_avg_mouse' + pltSuffix + '.png')
            plt.close()


# FIXME: Averaging is not aware of dropped channels
# TODO: Impl mouse-specific
def plot_2D_target(dataDB, h5fname, dfSummary, trgChName, paramKey, paramNames, dropChannels=None, fontsize=20):
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    nChannel = len(labelsCanon)
    trgIdx = labelsCanon.index(trgChName)
    pidTypes = ['unique', 'syn', 'red']

    mice = sorted(dataDB.mice)
    nMice = len(mice)
    nParam = len(paramNames)

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename', paramKey}))
    for key, dataTmp in dfSummary.groupby(groupLst):
        for pidType in pidTypes:
            fig, ax = plt.subplots(nrows=nMice, ncols=nParam, figsize=(4*nParam, 4*nMice))
            for iMouse, mousename in enumerate(mice):
                ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

                for iParam, paramName in enumerate(paramNames):
                    ax[0][iParam].set_title(paramName, fontsize=fontsize)

                    idx, row = pd_is_one_row(pd_query(dataTmp, {'mousename': mousename, paramKey: paramName}))
                    if row is not None:
                        idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                        Mrez3D = list_to_3Dmat(idxs, vals, nChannel)
                        Mrez2D = Mrez3D[:, :, trgIdx]
                        Mrez2D = drop_channels(Mrez2D, dropIdxs=dropChannels)

                        if pidType != 'unique':
                            Mrez2D = matrix_copy_triangle_symmetric(Mrez2D, source='U')

                        vmax = 1.0 if pidType != 'syn' else 0.5
                        imshow(fig, ax[iMouse][iParam], Mrez2D, cmap='jet',
                               haveColorBar=iParam == nParam - 1, limits=[0, vmax])

            prefixPath = 'pics/pid_2D/pid_2D_' + trgChName + '_' + paramKey + '/' + pidType + '/'
            make_path(prefixPath)
            pltSuffix = '_'.join([paramKey, trgChName] + list(key) + [pidType])
            plt.savefig(prefixPath + 'pid_2D_bytrg_mouse' + pltSuffix + '.png')
            plt.close()


def plot_2D_target_mousephase_subpre(dataDB, h5fname, dfSummary, trgChName, dropChannels=None, fontsize=20):
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    nChannel = len(labelsCanon)
    trgIdx = labelsCanon.index(trgChName)
    pidTypes = ['unique', 'syn', 'red']

    intervNames = dataDB.get_interval_names()
    mice = sorted(dataDB.mice)
    nMice = len(mice)
    nInterv = len(intervNames)

    dfBnSession = pd_query(dfSummary, {'datatype': 'bn_session'})
    groupLst = sorted(list(set(dfBnSession.columns) - {'key', 'mousename', 'phase'}))
    for key, dataTmp in dfBnSession.groupby(groupLst):
        for pidType in pidTypes:
            fig, ax = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4*nInterv, 4*nMice))
            for iMouse, mousename in enumerate(mice):
                ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
                rezDict = {}
                for iInterv, intervName in enumerate(intervNames):
                    ax[0][iInterv].set_title(intervName, fontsize=fontsize)

                    idx, row = pd_is_one_row(pd_query(dataTmp, {'mousename': mousename, 'phase': intervName}))
                    if row is not None:
                        idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                        Mrez3D = list_to_3Dmat(idxs, vals, nChannel)
                        Mrez2D = Mrez3D[:, :, trgIdx]
                        Mrez2D = drop_channels(Mrez2D, dropIdxs=dropChannels)

                        if pidType != 'unique':
                            Mrez2D = matrix_copy_triangle_symmetric(Mrez2D, source='U')

                        rezDict[intervName] = Mrez2D

                for iInterv, intervName in enumerate(intervNames):
                    if (intervName != 'PRE') and (intervName in rezDict.keys()):
                        vmax = 1.0 if pidType != 'syn' else 0.5
                        imshow(fig, ax[iMouse][iInterv], rezDict[intervName] - rezDict['PRE'],
                               cmap='jet', haveColorBar=iInterv == nInterv - 1, limits=[-vmax, vmax])

            prefixPath = 'pics/pid_2D/pid_2D_' + trgChName + '_subpre/' + pidType + '/'
            make_path(prefixPath)
            pltSuffix = '_'.join([trgChName] + list(key) + [pidType])
            plt.savefig(prefixPath + 'pid_2D_bytrg_mousephase_subpre_' + pltSuffix + '.png')
            plt.close()


def plot_2D_target_mousephase_submouse(dataDB, h5fname, dfSummary, trgChName, dropChannels=None, fontsize=20):
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    nChannel = len(labelsCanon)
    trgIdx = labelsCanon.index(trgChName)
    pidTypes = ['unique', 'syn', 'red']

    intervNames = dataDB.get_interval_names()
    mice = sorted(dataDB.mice)
    nMice = len(mice)
    nInterv = len(intervNames)

    groupLst = sorted(list(set(dfSummary.columns) - {'key', 'mousename', 'phase'}))
    for key, dataTmp in dfSummary.groupby(groupLst):
        for pidType in pidTypes:
            fig, ax = plt.subplots(nrows=nMice, ncols=nInterv, figsize=(4*nInterv, 4*nMice))
            for iInterv, intervName in enumerate(intervNames):
                ax[0][iInterv].set_title(intervName, fontsize=fontsize)

                rezDict = {}
                for iMouse, mousename in enumerate(mice):
                    ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)

                    idx, row = pd_is_one_row(pd_query(dataTmp, {'mousename': mousename, 'phase': intervName}))
                    if row is not None:
                        idxs, vals = read_computed_3D(h5fname, row['key'], pidType)
                        Mrez3D = list_to_3Dmat(idxs, vals, nChannel)
                        Mrez2D = Mrez3D[:, :, trgIdx]
                        Mrez2D = drop_channels(Mrez2D, dropIdxs=dropChannels)

                        if pidType != 'unique':
                            Mrez2D = matrix_copy_triangle_symmetric(Mrez2D, source='U')

                        rezDict[mousename] = Mrez2D

                rezAvg = np.mean(list(rezDict.values()), axis=0)
                for iMouse, mousename in enumerate(mice):
                    if mousename in rezDict.keys():
                        vmax = 1.0 if pidType != 'syn' else 0.5
                        imshow(fig, ax[iMouse][iInterv], rezDict[mousename] - rezAvg,
                               cmap='jet', haveColorBar=iInterv == nInterv - 1, limits=[-vmax, vmax])

            prefixPath = 'pics/pid_2D/pid_2D_' + trgChName + '_submouse/' + pidType + '/'
            make_path(prefixPath)
            pltSuffix = '_'.join([trgChName] + list(key) + [pidType])
            plt.savefig(prefixPath + 'pid_2D_bytrg_mousephase_submouse_' + pltSuffix + '.png')
            plt.close()


def plot_2D_bytarget_synergy_cluster(dataDB, h5fname, dfSummary, trgChName,
                                     clusterParam=1.0, dropWeakChannelThr=None, fontsize=20):
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

            imshow(fig, ax[0, iMouse], Mrez2D, title=row['mousename'], haveColorBar=True,
                   cmap='jet', fontsize=fontsize, limits=[0, 0.5])
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