import os
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from scipy.stats import mannwhitneyu, fisher_exact
# from sklearn.metrics import cohen_kappa_score
from pathlib import Path

from IPython.display import display
from ipywidgets import IntProgress

from mesostat.visualization.mpl_matrix import imshow

from lib.common.param_sweep import param_vals_to_suffix

# from mesostat.utils.matrix import drop_channels, offdiag_1D, matrix_copy_triangle_symmetric
# from mesostat.stat.stat import continuous_empirical_CDF
# from mesostat.stat.classification import confusion_matrix
# from mesostat.stat.clustering import cluster_dist_matrix_min, cluster_plot
#
# from mesostat.utils.pandas_helper import pd_query, pd_merge_multiple, pd_is_one_row, pd_append_row, pd_pivot
# from mesostat.visualization.mpl_barplot import barplot_stacked_indexed, barplot_labeled, sns_barplot

################################
#  Converting target-sweep into stacked
################################


def trgsweep_all_parse_key(key):
    lst = key.split('_')
    rez = {
        'mousename': '_'.join(lst[1:3]),
        'datatype': '_'.join(lst[3:5]),
        'trialType': lst[5],
        'trg': lst[6]
    }

    if len(lst) == 7:
        return rez
    elif len(lst) == 8:
        rez['performance'] = lst[7]
        return rez
    else:
        raise ValueError('Unexpected key', key)


def trgsweep_all_summary_df(h5fname):
    with h5py.File(h5fname, 'r') as f:
        keys = set(f.keys()) - {'lock'}

    # Only keep value keys, ignore labels for now
    keys = [key for key in keys if key[:5] != 'Label']

    summaryDF = pd.DataFrame()
    for key in keys:
        summaryDF = summaryDF.append(pd.DataFrame({**{'key': key}, **trgsweep_all_parse_key(key)}, index=[0]))

    return summaryDF.reset_index(drop=True)


def trgsweep_read_data(h5fname, keyVals, nChannelTrg, nTimeTrg):
    keyLabels = 'Label_' + keyVals[4:]

    with h5py.File(h5fname, 'r') as f:
        labels = np.copy(f[keyLabels])
        vals = np.copy(f[keyVals])

    nPairSrc = (nChannelTrg - 1)*(nChannelTrg - 2) / 2
    assert labels.shape == (nPairSrc, 3)
    assert vals.shape == (nPairSrc, nTimeTrg, 4)

    # # Drop negatives
    # vals = np.clip(vals, 0, None)

    return labels, vals


def trgsweep_merge_data(h5fname, dfSummary, nChannelTrg, nTimeTrg):
    cols = ['mousename', 'datatype', 'trialType']
    if 'performance' in dfSummary.columns:
        cols += ['performance']

    for vals, dfSub in dfSummary.groupby(cols):
        rezSuffix = '_'.join(vals)
        print(rezSuffix)

        labelsLst = []
        valsLst = []
        for idx, row in dfSub.iterrows():
            # trg = row['trg']

            labels, vals = trgsweep_read_data(h5fname, row['key'], nChannelTrg, nTimeTrg)
            # labels = np.concatenate([labels, np.full((276,), trg)], axis=1)

            labelsLst += [labels]
            valsLst += [vals]

        with h5py.File(h5fname, 'a') as f:
            f['Label_' + rezSuffix] = np.concatenate(labelsLst, axis=0)
            f['PID_' + rezSuffix] =  np.concatenate(valsLst, axis=0)


def trgsweep_purge_old(h5fname, dfSummary):
    with h5py.File(h5fname, 'a') as f:
        for idx, row in dfSummary.iterrows():
            keyVal = row['key']
            keyLabel = 'Label_' + keyVal[4:]

            if keyVal in f.keys():
                print(row)

                del f[keyVal]
                del f[keyLabel]


################################
#  Read stacked data
################################

def pid_all_parse_key(key):
    lst = key.split('_')
    rez = {
        'mousename': '_'.join(lst[1:3]),
        'datatype': '_'.join(lst[3:5]),
        'trialType': lst[5]
    }

    if len(lst) == 6:
        return rez
    elif len(lst) == 7:
        rez['performance'] = lst[6]
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


def pid_read_data(h5fname, key):
    with h5py.File(h5fname, 'r') as f:
        data = np.array(f[key])
        labels = np.array(f['Label_' + key[4:]])

    return labels, data


################################
#  1D Movies
################################

def get_pid_type(labels, data, pidType):
    if pidType == 'unique':
        dataL = data[:, :, 0]
        dataR = data[:, :, 1]
    elif pidType == 'red':
        dataL = data[:, :, 2]
        dataR = data[:, :, 2]
    elif pidType == 'syn':
        dataL = data[:, :, 3]
        dataR = data[:, :, 3]
    else:
        raise ValueError('Unexpected pid type', pidType)

    labelsL = labels
    labelsR = labels[:, [1,0,2]]  # Swap sources, keep target
    labelsEff = np.concatenate([labelsL, labelsR], axis=0)
    dataEff = np.concatenate([dataL, dataR], axis=0)
    return labelsEff, dataEff


# Map channels from index list to 2D matrix
def get_2D_map(labelIdxsPairS, dataPairS, nChannel):
    nTime = dataPairS.shape[1]
    rezSPP = np.full((nTime, nChannel, nChannel), np.nan)
    rezSPP[:, labelIdxsPairS[:, 0], labelIdxsPairS[:, 1]] = dataPairS.T
    return rezSPP


def pid_movie_1D_brainplot_mousephase(dataDB, h5fname, dfSummary, trialTypes, pidType, vmin=None, vmax=None, fontsize=20):
    mice = sorted(dataDB.mice)
    nMice = len(mice)
    nTrialType = len(trialTypes)

    sweepCols = list(set(dfSummary.columns) - {'mousename', 'trialType', 'key'})
    for paramVals, dfTmp in dfSummary.groupby(sweepCols):
        plotSuffix = param_vals_to_suffix(paramVals)

        # Store all preprocessed data first
        dataDict = {}
        for mousename, dfMouse in dfTmp.groupby(['mousename']):
            nChannel = dataDB.get_nchannels(mousename)
            for idx, row in dfMouse.iterrows():
                trialType = row['trialType']
                print('Reading data, ', plotSuffix, mousename, trialType)

                with h5py.File(h5fname, 'r') as f:
                    labels, data = pid_read_data(h5fname, row['key'])

                labels, data = get_pid_type(labels, data, pidType)

                # # Drop negatives
                # vals = np.clip(vals, 0, None)

                nTimeTrg = data.shape[1]
                dataSP = np.full((nTimeTrg, nChannel), np.nan)

                for iCh in range(nChannel):
                    if iCh in labels[:, 2]:
                        idxs = labels[:, 2] == iCh
                        dataSP[:, iCh] = np.mean(data[idxs], axis=0)

                dataDict[(mousename, trialType)] = dataSP

        # Test that all datasets have the same duration
        shapeSet = set([v.shape for v in dataDict.values()])
        assert len(shapeSet) == 1
        nTimes = shapeSet.pop()[0]

        progBar = IntProgress(min=0, max=nTimes, description=plotSuffix)
        display(progBar)  # display the bar
        for iTime in range(nTimes):
            outfname = 'pid1D_brainplot_mousetrialtype_' + plotSuffix + '_' + str(iTime) + '.png'
            if os.path.isfile(outfname):
                print('Already calculated', iTime, 'skipping')
                progBar.value += 1
                continue

            fig, ax = plt.subplots(nrows=nMice, ncols=nTrialType, figsize=(4 * nTrialType, 4 * nMice), tight_layout=True)

            for iMouse, mousename in enumerate(mice):
                ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
                for iTT, trialType in enumerate(trialTypes):
                    ax[0][iTT].set_title(trialType, fontsize=fontsize)
                    # print(datatype, mousename)

                    dataP = dataDict[(mousename, trialType)][iTime]

                    haveColorBar = iTT == nTrialType - 1
                    dataDB.plot_area_values(fig, ax[iMouse][iTT], dataP, vmin=vmin, vmax=vmax, cmap='jet',
                                            haveColorBar=haveColorBar)

            plt.savefig(outfname)
            # plt.close()
            plt.cla()
            plt.clf()
            plt.close('all')
            progBar.value += 1


def pid_movie_2D_mousephase_bytrg(dataDB, h5fname, dfSummary, trialTypes, pidType, trgChName, vmin=None, vmax=None, fontsize=20):
    labelsCanon = list(dataDB.map_channel_labels_canon().values())
    nChannel = len(labelsCanon)
    trgIdx = labelsCanon.index(trgChName)

    mice = sorted(dataDB.mice)
    nMice = len(mice)
    nTrialType = len(trialTypes)

    sweepCols = list(set(dfSummary.columns) - {'mousename', 'trialType', 'key'})
    for paramVals, dfTmp in dfSummary.groupby(sweepCols):
        plotSuffix = param_vals_to_suffix(paramVals)

        # Store all preprocessed data first
        dataDict = {}
        for mousename, dfMouse in dfTmp.groupby(['mousename']):
            for idx, row in dfMouse.iterrows():
                trialType = row['trialType']
                print('Reading data, ', plotSuffix, mousename, trialType)

                with h5py.File(h5fname, 'r') as f:
                    labels, data = pid_read_data(h5fname, row['key'])

                # # Drop negatives
                # vals = np.clip(vals, 0, None)

                labels, data = get_pid_type(labels, data, pidType)
                idxs = labels[:, 2] == trgIdx
                dataDict[(mousename, trialType)] = get_2D_map(labels[idxs, :2], data[idxs], nChannel)

        # Test that all datasets have the same duration
        shapeSet = set([v.shape for v in dataDict.values()])
        assert len(shapeSet) == 1
        nTimes = shapeSet.pop()[0]

        progBar = IntProgress(min=0, max=nTimes, description=plotSuffix)
        display(progBar)  # display the bar
        for iTime in range(nTimes):
            outfname = 'pid2D_byTrg_mousetrialtype_' + trgChName + '_' + plotSuffix + '_' + str(iTime) + '.png'
            if os.path.isfile(outfname):
                print('Already calculated', iTime, 'skipping')
                progBar.value += 1
                continue

            fig, ax = plt.subplots(nrows=nMice, ncols=nTrialType, figsize=(4 * nTrialType, 4 * nMice), tight_layout=True)

            for iMouse, mousename in enumerate(mice):
                ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
                for iTT, trialType in enumerate(trialTypes):
                    ax[0][iTT].set_title(trialType, fontsize=fontsize)
                    # print(datatype, mousename)

                    dataPP = dataDict[(mousename, trialType)][iTime]

                    haveColorBar = iTT == nTrialType - 1
                    imshow(fig, ax[iMouse][iTT], dataPP, limits=[vmin, vmax], cmap='jet', haveColorBar=haveColorBar)

            plt.savefig(outfname)
            plt.cla()
            plt.clf()
            plt.close('all')
            progBar.value += 1
