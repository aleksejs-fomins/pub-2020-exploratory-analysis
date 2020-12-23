import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import combine_pvalues

from mesostat.metric.idtxl_pid import bivariate_pid_3D
from mesostat.utils.pandas_helper import pd_append_row, pd_query
from mesostat.utils.signals import bin_data
from mesostat.stat.permtests import percentile_twosided, perm_test_resample
from mesostat.utils.plotting import imshowAddColorBar


def pid(dataDB, mc, sources, targets, mousename, datatype, zscoreDim, cropTime, trialType, performance, nPerm=1000, nBin=4):
    channelNames = dataDB.get_channel_labels(mousename)
    sourceIdxs = [channelNames.index(s) for s in sources]
    targetIdxs = [channelNames.index(t) for t in targets]

    dataLst = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype,
                                    zscoreDim=zscoreDim, cropTime=cropTime,
                                    trialType=trialType, performance=performance)

    # Concatenate all sessions
    data = np.concatenate(dataLst, axis=0)
    data = np.mean(data, axis=1)  # Average out time
    data = bin_data(data, nBin, axis=1)

    settings_estimator = {'pid_estimator': 'TartuPID', 'lags_pid': [0, 0]}

    mc.set_data(data, 'rp')
    rez = mc.metric3D('BivariatePID', '',
                      metricSettings={'settings_estimator': settings_estimator,
                                      'src': sourceIdxs},
                      sweepSettings={'trg': targetIdxs})

    # Since all channels are binned to the same quantiles,
    # the permutation test is exactly the same for all of them, so we need any three channels as input
    settings_test = {'src': [0, 1], 'trg': 2, 'settings_estimator': settings_estimator}
    fTest = lambda x: bivariate_pid_3D(x, settings_test)
    dataTest = data[:, :3][..., None]  # Add fake 1D sample dimension

    fRand = perm_test_resample(fTest, dataTest, nPerm, iterAxis=1)

    rezTest = [
        percentile_twosided(fTrue, fRand, settings={"haveEffectSize": True, "haveMeans": True})
        for fTrue in rez
    ]

    rezTest = np.array(rezTest)
    df = pd.DataFrame(columns=['mousename', 'S1', 'S2', 'T', 'PID', 'p', 'effSize', 'muTrue', 'muRand'])
    for iTrg, trgName in enumerate(targets):
        for iType, infType in enumerate(['U1', 'U2', 'red', 'syn']):
            rowLst = [mousename, *sources, trgName, infType, *rezTest[iTrg, 1:, iType]]
            df = pd_append_row(df, rowLst, skip_repeat=False)

    return df


def info3D_plot_slice_trg(dataDB, sources, targets, datatype, zscoreDim, cropTime, trialType, performance, nBin=4):
    channelNames = dataDB.get_channel_labels('mvg_4')
    sourceIdxs = [channelNames.index(s) for s in sources]
    targetIdxs = [channelNames.index(t) for t in targets]

    s1Idx, s2Idx = sourceIdxs
    s1Name, s2Name = channelNames[s1Idx], channelNames[s2Idx]
    for targetIdx in targetIdxs:
        nMice = len(dataDB.mice)
        fig, ax = plt.subplots(nrows=nMice, ncols=nBin, figsize=(4 * nBin, 4 * nMice), tight_layout=True)

        for iMouse, mousename in enumerate(sorted(dataDB.mice)):

            dataLst = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype,
                                            zscoreDim=zscoreDim, cropTime=cropTime,
                                            trialType=trialType, performance=performance)

            # Concatenate all sessions
            data = np.concatenate(dataLst, axis=0)

            # Average out time
            data = np.mean(data, axis=1)

            # Binarize data over channels
            data = bin_data(data, nBin, axis=1)

            trgName = channelNames[targetIdx]
            h3d = np.histogramdd(data[:, [targetIdx, s1Idx, s2Idx]], bins = (nBin,)*3)[0]
            h3d /= np.sum(h3d)  # Normalize

            for iTrgBin in range(nBin):
                img = ax[iMouse][iTrgBin].imshow(h3d[iTrgBin], vmin=0, vmax=10/nBin**3, cmap='jet')
                ax[iMouse][iTrgBin].set_xlabel(s2Name)
                ax[iMouse][iTrgBin].set_ylabel(s1Name)
                ax[iMouse][iTrgBin].set_title(trgName + '=' + str(iTrgBin))
                imshowAddColorBar(fig, ax[iMouse][iTrgBin], img)

        plt.savefig('pics/info3D_' + s1Name + '_' + s2Name + '_' + trgName + '.pdf')
        plt.close(fig)


def plot_pval_aggr(h5fname, groupkey):
    df = pd.read_hdf(h5fname, key=groupkey)
    pidTypes = ['U1', 'U2', 'red', 'syn']
    mice = set(df['mousename'])

    fig, ax = plt.subplots()
    for s1 in sorted(set(df['S1'])):
        for s2 in sorted(set(df['S2'])):
            if s1 != s2:
                for t in sorted(set(df['T'])):
                    label = '(' + s1 + ',' + s2 + ')->' + t
                    pLst = []

                    for pidType in pidTypes:
                        rows = pd_query(df, {'S1': s1, 'S2': s2, 'T': t, 'PID': pidType})
                        assert len(rows) == len(mice)
                        p = combine_pvalues(list(rows['p']))[1]

                        pLst += [-np.log10(p)]

                    ax.plot(pLst, label=label)

    ax.legend()
    ax.set_ylabel('-log10(p)')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Unique1', 'Unique2', 'Redundancy', 'Synergy'])
    plt.axhline(y=2, linestyle='--', color='r')
    plt.savefig('pics/PID_PVAL_AGGR_'+groupkey+'.pdf')
    plt.close(fig)


def plot_metric_bymouse(h5fname, groupkey, metricKey, metricName, yscale=None, clip=None, ylim=None):
    df = pd.read_hdf(h5fname, key=groupkey)
    mice = set(df['mousename'])

    pidTypes = ['U1', 'U2', 'red', 'syn']

    nMice = len(mice)
    fig, ax = plt.subplots(ncols=nMice, figsize=(6 * nMice, 6))

    for iMouse, mousename in enumerate(sorted(mice)):
        for s1 in sorted(set(df['S1'])):
            for s2 in sorted(set(df['S2'])):
                if s1 != s2:
                    for t in sorted(set(df['T'])):
                        label = '(' + s1 + ',' + s2 + ')->' + t
                        metricLst = []

                        for pidType in pidTypes:
                            rows = pd_query(df, {'mousename': mousename, 'S1': s1, 'S2': s2, 'T': t, 'PID': pidType})
                            assert len(rows) == 1

                            metricLst += [list(rows[metricKey])[0]]

                        if clip is not None:
                            metricLst = np.clip(metricLst, *clip)

                        ax[iMouse].plot(metricLst, label=label)

        ax[iMouse].legend()
        ax[iMouse].set_title(mousename)
        ax[iMouse].set_ylabel(metricName)
        ax[iMouse].set_xticks([0, 1, 2, 3])
        ax[iMouse].set_xticklabels(['Unique1', 'Unique2', 'Redundancy', 'Synergy'])
        if yscale is not None:
            ax[iMouse].set_yscale(yscale)
        if ylim is not None:
            ax[iMouse].set_ylim(ylim)

    plt.savefig('pics/PID_BY_MOUSE_'+groupkey+'.pdf')
    plt.close(fig)