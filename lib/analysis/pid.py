import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import combine_pvalues
from collections import defaultdict

from mesostat.metric.idtxl_pid import bivariate_pid_3D
from mesostat.utils.pandas_helper import pd_append_row, pd_query, merge_df_from_dict
from mesostat.utils.signals import bin_data
from mesostat.stat.permtests import percentile_twosided, perm_test_resample
from mesostat.visualization.mpl_colorbar import imshow_add_color_bar
from mesostat.visualization.mpl_violin import violins_labeled
from mesostat.visualization.mpl_cdf import cdf_labeled
from mesostat.visualization.mpl_barplot import barplot_labeled

'''
Hypothesis-plots:
Plots for specific subsets of sources and target constituting a hypothesis
'''


# Return a list of all pairs of elements in a list, excluding flips.
def _pairs_unordered(lst):
    n = len(lst)
    assert n >= 2
    rez = []
    for i in range(n):
        for j in range(i+1, n):
            rez += [[lst[i], lst[j]]]
    return rez


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

    # Find indices of channel labels
    sourceIdxs = [labelsAll.index(s) for s in labelsSrc]
    targetIdxs = [labelsAll.index(t) for t in labelsTrg]

    # Find combinations of all source pairs
    sourceIdxPairs = _pairs_unordered(sourceIdxs)

    # Concatenate all sessions
    data = np.concatenate(dataLst, axis=0)   # Concatenate trials and sessions
    data = np.mean(data, axis=1)             # Average out time
    data = bin_data(data, nBin, axis=1)      # Bin data separately for each channel

    settings_estimator = {'pid_estimator': 'TartuPID', 'lags_pid': [0, 0]}

    mc.set_data(data, 'rp')
    rez = mc.metric3D('BivariatePID', '',
                      metricSettings={'settings_estimator': settings_estimator},
                      sweepSettings={'src': sourceIdxPairs, 'trg': targetIdxs})
    # rez = mc.metric3D('BivariatePID', '',
    #                   metricSettings={'settings_estimator': settings_estimator, 'src': sourceIdxs},
    #                   sweepSettings={'trg': targetIdxs})

    # Since all channels are binned to the same quantiles,
    # the permutation test is exactly the same for all of them, so we need any three channels as input
    settings_test = {'src': [0, 1], 'trg': 2, 'settings_estimator': settings_estimator}
    fTest = lambda x: bivariate_pid_3D(x, settings_test)
    dataTest = data[:, :3][..., None]  # Add fake 1D sample dimension
    fRand = perm_test_resample(fTest, dataTest, nPerm, iterAxis=1)

    df = pd.DataFrame(columns=['S1', 'S2', 'T', 'PID', 'p', 'effSize', 'muTrue', 'muRand'])
    for iSrcPair, (iS1, iS2) in enumerate(sourceIdxPairs):
        for iTrg, trgName in enumerate(labelsTrg):
            rezThis = rez[iSrcPair, iTrg]
            pvalSummary = percentile_twosided(rezThis, fRand, settings={"haveEffectSize": True, "haveMeans": True})
            labelS1, labelS2 = labelsAll[iS1], labelsAll[iS2]

            for iType, infType in enumerate(['U1', 'U2', 'red', 'syn']):
                rowLst = [labelS1, labelS2, trgName, infType, *pvalSummary[1:, iType]]
                df = pd_append_row(df, rowLst, skip_repeat=False)
    return df


def hypotheses_calc_pid(dataDB, mc, hDict, intervDict, h5outname, datatypes=None, **kwargs):
    if datatypes is None:
        datatypes = dataDB.get_data_types()

    for datatype in datatypes:
        for hLabel, (intervKey, sources, targets) in hDict.items():
            print(hLabel)

            # Calculate PID
            rezDict = {}
            for mousename in dataDB.mice:
                channelNames = dataDB.get_channel_labels(mousename)
                dataLst = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype,
                                                zscoreDim=None, cropTime=intervDict[intervKey], **kwargs)

                rezDict[(mousename,)] = pid(dataLst, mc, channelNames, sources, targets, nPerm=2000, nBin=4)

            rezDF = merge_df_from_dict(rezDict, ['mousename'])

            #     # Display resulting dataframe
            #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            #         display(rezDF.sort_values(by=['S1', 'S2', 'T', 'PID', 'mousename']))

            # Save to file
            dataLabel = '_'.join(['PID', datatype, hLabel])
            rezDF.to_hdf(h5outname, dataLabel, mode='a', format='table', data_columns=True)


def hypotheses_plot_pid(dataDB, hDict, h5outname, datatypes=None):
    if datatypes is None:
        datatypes = dataDB.get_data_types()

    for datatype in datatypes:
        for hLabel, (intervKey, sources, targets) in hDict.items():
            print(hLabel)

            dataLabel = '_'.join(['PID', datatype, hLabel])
            plot_pval_aggr(h5outname, dataLabel)
            plot_metric_bymouse(h5outname, dataLabel, 'muTrue', 'Info(Bits)', yscale='log', clip=[1.0E-3, 1],
                                    ylim=[1.0E-3, 1])


def hypotheses_calc_plot_info3D(dataDB, hDict, intervDict, datatypes=None, nBin=4, **kwargs):
    if datatypes is None:
        datatypes = dataDB.get_data_types()

    for datatype in datatypes:
        for hLabel, (intervKey, sources, targets) in hDict.items():
            print(hLabel)

            dataLabel = '_'.join(['PID', datatype, hLabel])

            sourcePairs = _pairs_unordered(sources)
            for s1Label, s2Label in sourcePairs:
                for labelTrg in targets:
                    nMice = len(dataDB.mice)
                    fig, ax = plt.subplots(nrows=nMice, ncols=nBin, figsize=(4 * nBin, 4 * nMice), tight_layout=True)

                    for iMouse, mousename in enumerate(sorted(dataDB.mice)):
                        channelNames = dataDB.get_channel_labels(mousename)
                        s1Idx = channelNames.index(s1Label)
                        s2Idx = channelNames.index(s2Label)
                        targetIdx = channelNames.index(labelTrg)

                        dataLst = dataDB.get_neuro_data({'mousename': mousename},
                                                        datatype=datatype, cropTime=intervDict[intervKey], **kwargs)

                        data = np.concatenate(dataLst, axis=0)  # Concatenate all sessions
                        data = np.mean(data, axis=1)  # Average out time
                        data = bin_data(data, nBin, axis=1)  # Binarize data over channels

                        h3d = np.histogramdd(data[:, [targetIdx, s1Idx, s2Idx]], bins=(nBin,) * 3)[0]
                        h3d /= np.sum(h3d)  # Normalize

                        for iTrgBin in range(nBin):
                            img = ax[iMouse][iTrgBin].imshow(h3d[iTrgBin], vmin=0, vmax=10 / nBin ** 3, cmap='jet')
                            ax[iMouse][iTrgBin].set_xlabel(s1Label)
                            ax[iMouse][iTrgBin].set_ylabel(s2Label)
                            ax[iMouse][iTrgBin].set_title(labelTrg + '=' + str(iTrgBin))
                            imshow_add_color_bar(fig, ax[iMouse][iTrgBin], img)

                    plt.savefig('pics/info3D_' + dataLabel + '_'  + s1Label + '_' + s2Label + '_' + labelTrg + '.pdf')
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


'''
All-plots:
Plots for all-to-all PID, pre-computed using a separate script
'''


# Parse data key from H5 storage of all-to-all pid file
def pid_all_parse_key(key):
    lst = key.split('_')
    return {
        'mousename' : '_'.join(lst[1:3]),
        'datatype' : '_'.join(lst[3:5]),
        'session' : '_'.join(lst[5:11]),
        'phase' : lst[-1]
    }


# Parse data keys from H5 storage of all-to-all pid file, return as dataframe
def pid_all_summary_df(h5fname):
    with h5py.File(h5fname, 'r') as f:
        keys = list(f.keys())

    summaryDF = pd.DataFrame()
    for key in keys:
        summaryDF = summaryDF.append(pd.DataFrame({**{'key': key}, **pid_all_parse_key(key)}, index=[0]))

    return summaryDF


# Plot fraction of significant PID's for each session from H5 storage of all-to-all pid file
def plot_all_frac_significant_bysession(h5fname):
    pidTypes = {'unique', 'red', 'syn'}
    summaryDF = pid_all_summary_df(h5fname)

    for keyLst, dfSession in summaryDF.groupby(['mousename', 'datatype', 'phase']):
        keyLabel = '_'.join(keyLst)
        print(keyLabel)

        pidDict = defaultdict(list)
        for idx, row in dfSession.iterrows():
            df1 = pd.read_hdf(h5fname, key=row['key'])

            # Merge Unique1 and Unique2 for this plot
            df1.replace({'PID' : {'U1' : 'unique', 'U2' : 'unique'}}, inplace=True)

            for pidType in pidTypes:
                pVal = df1[df1['PID'] == pidType]['p']
                fracSig = np.mean(pVal < 0.01)
                pidDict[pidType] += [fracSig]

        plt.figure()
        for pidType in pidTypes:
            plt.plot(pidDict[pidType], label=pidType)

        plt.legend()
        plt.xlabel('session')
        plt.ylabel('Fraction Significant')
        plt.savefig('PID_Freq_Significant_' + keyLabel + '.png')
        plt.close()


# Plot distribution of results by PID, session and other params - for all-to-all pid file
def plot_all_results_distribution(h5fname, plotstyle='cdf'):
    pidTypes = {'unique', 'red', 'syn'}
    summaryDF = pid_all_summary_df(h5fname)

    for keyLst, dfSession in summaryDF.groupby(['mousename', 'datatype', 'phase']):
        keyLabel = '_'.join(keyLst)
        print(keyLabel)

        for idx, row in dfSession.iterrows():
            df1 = pd.read_hdf(h5fname, key=row['key'])

            # Merge Unique1 and Unique2 for this plot
            df1.replace({'PID' : {'U1' : 'unique', 'U2' : 'unique'}}, inplace=True)

            fig, ax = plt.subplots(ncols=3, figsize=(12, 4))

            pVals = []
            effSizes = []
            values = []
            for pidType in pidTypes:
                dfThis = df1[df1['PID'] == pidType]

                pVals += [np.array(dfThis['p'])]
                effSizes += [np.array(dfThis['effSize'])]
                values += [np.array(dfThis['muTrue'])]

            if plotstyle == 'violin':
                ax[0].axhline(y=0.01, linestyle='--', label='significant')
                violins_labeled(ax[0], pVals, pidTypes, 'pidType', 'pVal', haveLog=True, violinScale='width')
                violins_labeled(ax[1], effSizes, pidTypes, 'pidType', 'effSize', haveLog=False, violinScale='width')
                violins_labeled(ax[2], values, pidTypes, 'pidType', 'Bits', haveLog=False, violinScale='width')
                plt.savefig('PID_Violin_' + keyLabel + '.png')
            elif plotstyle == 'cdf':
                ax[0].axvline(x=0.01, linestyle='--', label='significant')
                cdf_labeled(ax[0], pVals, pidTypes, 'pidType', 'pVal', haveLog=True)
                cdf_labeled(ax[1], effSizes, pidTypes, 'pidType', 'effSize', haveLog=False)
                cdf_labeled(ax[2], values, pidTypes, 'pidType', 'Bits', haveLog=False)
                plt.savefig('PID_CDF_' + keyLabel + '.png')
            else:
                raise ValueError('Unexpected plot style', plotstyle)

            plt.close()


# FIXME: Why perf=2.5 in one case? How did we deal with this in Yaro's case (e.g. shared-links)
def plot_all_frac_significant_performance_scatter(dataDB, h5fname):
    pidTypes = {'unique', 'red', 'syn'}
    summaryDF = pid_all_summary_df(h5fname)

    for keyLst, dfSession in summaryDF.groupby(['datatype', 'phase']):    # 'mousename'
        keyLabel = '_'.join(keyLst)

        pidDictNaive = defaultdict(list)
        pidDictExpert = defaultdict(list)
        fig, ax = plt.subplots(nrows=2, ncols=len(pidTypes), figsize=(len(pidTypes) * 4, 8), tight_layout=True)
        for keyLst2, dfSession2 in dfSession.groupby(['mousename']):  # 'mousename'
            print(keyLabel, keyLst2)

            pidDictSession = defaultdict(list)
            for idx, row in dfSession2.iterrows():
                perf = dataDB.get_performance(row['session'])

                df1 = pd.read_hdf(h5fname, key=row['key'])

                # Merge Unique1 and Unique2 for this plot
                df1.replace({'PID' : {'U1' : 'unique', 'U2' : 'unique'}}, inplace=True)

                for pidType in pidTypes:
                    pVal = df1[df1['PID'] == pidType]['p']

                    fracSig = np.mean(pVal < 0.01)
                    pidDictSession[pidType] += [(perf, fracSig)]
                    if perf < 0.7:
                        pidDictNaive[pidType] += [fracSig]
                    else:
                        pidDictExpert[pidType] += [fracSig]

            for iFig, pidType in enumerate(pidTypes):
                ax[0, iFig].set_title(pidType)
                ax[0, iFig].set_xlabel('Performance')
                ax[0, iFig].set_ylabel('Fraction Significant')
                ax[0, iFig].plot(*np.array(pidDictSession[pidType]).T, '.', label=str(keyLst2))

        for iFig, pidType in enumerate(pidTypes):
            ax[0, iFig].legend()
            ax[0, iFig].set_xlim([0, 1])
            ax[0, iFig].axvline(x=0.7, linestyle='--', color='gray', alpha=0.5)

            violins_labeled(ax[1, iFig],
                            [pidDictNaive[pidType], pidDictExpert[pidType]],
                            ['naive', 'expert'],
                            'performance',
                            'Fraction Significant',
                            violinInner='box',
                            style='bar',
                            sigTestPairs=[[0, 1]]
                            )
            ax[1, iFig].set_ylim([0, 1])

        # plt.show()
        plt.savefig('PID_Freq_Significant_vs_perf_' + keyLabel + '.png')
        plt.close()



# FIXME: Pointless with current bug - wait for new data.
# TODO: Assembling matrix from labels too slow. Figure out from matrix decomposition
#   -> Matrix decomposition too complicated. Slow better
def plot_all_top_10_frac_significant(dataDB, h5fname):
    pidTypes = {'unique', 'red', 'syn'}
    summaryDF = pid_all_summary_df(h5fname)

    for keyLst, dfSession in summaryDF.groupby(['datatype', 'phase']):    # 'mousename'
        keyLabel = '_'.join(keyLst)

        mat3DmouseDict = defaultdict(dict)

        for mousename, dfSession2 in dfSession.groupby(['mousename']):  # 'mousename'
            print(keyLabel, mousename)
            channelLabels = dataDB.get_channel_labels(mousename)
            nChannels = len(channelLabels)


            for pidType in pidTypes:
                mat3DmouseDict[pidType][mousename] = np.zeros((nChannels, nChannels, nChannels))

            for idx, row in dfSession2.iterrows():
                df1 = pd.read_hdf(h5fname, key=row['key'])

                # Merge Unique1 and Unique2 for this plot
                df1.replace({'PID' : {'U1' : 'unique', 'U2' : 'unique'}}, inplace=True)

                # Convert all channel labels to indices
                for iCh, ch in enumerate(channelLabels):
                    df1.replace({'S1' : {ch : iCh}, 'S2' : {ch : iCh}, 'T' : {ch : iCh}})

                for pidType in pidTypes:
                    df1PID = df1[df1['PID'] == pidType]
                    df1Sig = df1PID[df1PID['p'] < 0.01]

                    for idx, row in df1Sig.iterrows():
                        x,y,z = row['S1'], row['S2'], row['T']
                        mat3DmouseDict[pidType][mousename][x, y, z] += 1

        for pidType in pidTypes:
            # 1) Sum all mouse arrays
            # 2) Find indices of largest values
            # 3) For each index, get channel names and values by mouse
            # 4) Plot stacked barplot with absolute numbers. Set ylim_max to total number of sessions
            # 5) Consider some significance test maybe

            pass





        # fig, ax = plt.subplots(nrows=2, ncols=len(pidTypes), figsize=(len(pidTypes) * 4, 8), tight_layout=True)