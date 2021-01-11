import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import combine_pvalues

from mesostat.metric.idtxl_pid import bivariate_pid_3D
from mesostat.utils.pandas_helper import pd_append_row, pd_query
from mesostat.utils.signals import bin_data
from mesostat.stat.permtests import percentile_twosided, perm_test_resample
from mesostat.visualization.mpl_colorbar import imshow_add_color_bar


# Calculate 3D PID with two sources and 1 target. If more than one target is provided,
def pid(dataLst, mc, labelsAll, labelsSrc, labelsTrg, nPerm=1000, nBin=4):
    '''
    :param dataLst:     List of data over sessions, each dataset is of shape 'rsp'
    :param mc:          MetricCalculator
    :param labelsAll:   List of labels of all the channels. Needed to identify indices of source and target channels
    :param labelsSrc:   List of labels of the source channels. Must be exactly two
    :param labelsTrg:   List of labels of the target channels. Each target is analysed separately
    :param nPerm:       Number of permutations to use for permutation testing
    :param nBin:        Number of bins to use to bin the data
    :return:            Dataframe containing PID results for each combination of sources and targets
    '''

    assert len(labelsSrc) == 2
    sourceIdxs = [labelsAll.index(s) for s in labelsSrc]
    targetIdxs = [labelsAll.index(t) for t in labelsTrg]

    # Concatenate all sessions
    data = np.concatenate(dataLst, axis=0)   # Concatenate trials and sessions
    data = np.mean(data, axis=1)             # Average out time
    data = bin_data(data, nBin, axis=1)      # Bin data separately for each channel

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
    df = pd.DataFrame(columns=['S1', 'S2', 'T', 'PID', 'p', 'effSize', 'muTrue', 'muRand'])
    for iTrg, trgName in enumerate(labelsTrg):
        for iType, infType in enumerate(['U1', 'U2', 'red', 'syn']):
            rowLst = [*labelsSrc, trgName, infType, *rezTest[iTrg, 1:, iType]]
            df = pd_append_row(df, rowLst, skip_repeat=False)

    return df


# Plot PID slices by target
def info3D_plot_slice_trg(dataDB, labelsSrc, labelsTrg, datakwargs, nBin=4):
    s1Label, s2Label = labelsSrc

    for labelTrg in labelsTrg:
        nMice = len(dataDB.mice)
        fig, ax = plt.subplots(nrows=nMice, ncols=nBin, figsize=(4 * nBin, 4 * nMice), tight_layout=True)

        for iMouse, mousename in enumerate(sorted(dataDB.mice)):
            channelNames = dataDB.get_channel_labels(mousename)
            s1Idx, s2Idx = [channelNames.index(s) for s in labelsSrc]
            targetIdx = channelNames.index(labelTrg)

            dataLst = dataDB.get_neuro_data({'mousename': mousename}, **datakwargs)

            data = np.concatenate(dataLst, axis=0)   # Concatenate all sessions
            data = np.mean(data, axis=1)             # Average out time
            data = bin_data(data, nBin, axis=1)      # Binarize data over channels

            h3d = np.histogramdd(data[:, [targetIdx, s1Idx, s2Idx]], bins = (nBin,)*3)[0]
            h3d /= np.sum(h3d)  # Normalize

            for iTrgBin in range(nBin):
                img = ax[iMouse][iTrgBin].imshow(h3d[iTrgBin], vmin=0, vmax=10/nBin**3, cmap='jet')
                ax[iMouse][iTrgBin].set_xlabel(s1Label)
                ax[iMouse][iTrgBin].set_ylabel(s2Label)
                ax[iMouse][iTrgBin].set_title(labelTrg + '=' + str(iTrgBin))
                imshow_add_color_bar(fig, ax[iMouse][iTrgBin], img)

        plt.savefig('pics/info3D_' + s1Label + '_' + s2Label + '_' + labelTrg + '.pdf')
        plt.close(fig)


def plot_pval_aggr(h5fname, groupkey):
    df = pd.read_hdf(h5fname, key=groupkey)
    mice = set(df['mousename'])
    fig, ax = plt.subplots()

    # Loop over sources and targets
    for colVals1, dfSub1 in df.groupby(['S1', 'S2', 'T']):
        label = '(' + colVals1[0] + ',' + colVals1[1] + ')->' + colVals1[2]
        pLst = []

        for colVals2, dfSub2 in dfSub1.groupby(['PID']):
            assert len(dfSub2) == len(mice)
            p = combine_pvalues(list(dfSub2['p']))[1]
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