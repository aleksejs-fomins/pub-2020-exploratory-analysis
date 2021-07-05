import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import combine_pvalues
from collections import defaultdict
import seaborn as sns

from mesostat.metric.idtxl_pid import bivariate_pid_3D
from mesostat.utils.pandas_helper import pd_append_row, pd_query, merge_df_from_dict
from mesostat.utils.signals.resample import bin_data
from mesostat.utils.signals.filter import drop_PCA
from mesostat.visualization.mpl_colorbar import imshow_add_color_bar
from mesostat.visualization.mpl_violin import violins_labeled
from mesostat.visualization.mpl_cdf import cdf_labeled
from mesostat.visualization.mpl_barplot import barplot_stacked, barplot_stacked_indexed
from mesostat.visualization.mpl_font import update_fonts_axis
from mesostat.stat.permtests import percentile_twosided, perm_test_resample
# from mesostat.stat.moments import n_largest_indices
from mesostat.utils.iterators.matrix import iter_g_2D, iter_gn_3D


'''
Hypothesis-plots:
Plots for specific subsets of sources and target constituting a hypothesis
'''


def _update_pid_rez_df_rand(df, rezThis, fRand, labelS1, labelS2, labelTrg):
    if df is None:
        df = pd.DataFrame(columns=['S1', 'S2', 'T', 'PID', 'p', 'effSize', 'muTrue', 'muRand'])

    pvalSummary = percentile_twosided(rezThis, fRand, settings={"haveEffectSize": True, "haveMeans": True})

    for iType, infType in enumerate(['U1', 'U2', 'red', 'syn']):
        rowLst = [labelS1, labelS2, labelTrg, infType, *pvalSummary[1:, iType]]
        df = pd_append_row(df, rowLst, skip_repeat=False)
    return df




# Calculate 3D PID with two sources and 1 target. If more than one target is provided,
def pid(dataLst, mc, labelsAll=None, labelsSrc=None, labelsTrg=None, nPerm=1000, nBin=4, nDropPCA=None, verbose=True,
        permuteTarget=False):
    '''
    :param dataLst:     List of data over sessions, each dataset is of shape 'rsp'
    :param mc:          MetricCalculator
    :param labelsAll:   List of labels of all the channels. Needed to identify indices of source and target channels
    :param labelsSrc:   List of labels of the source channels. Must be exactly two
    :param labelsTrg:   List of labels of the target channels. Each target is analysed separately
    :param nPerm:       Number of permutations to use for permutation testing
    :param nBin:        Number of bins to use to bin the data
    :param nDropPCA:    Number of primary principal components to drop prior to analysis
    :param verbose:     Verbosity of output
    :param permuteTarget:  Whether to permute data by target, resulting in shuffled estimator
    :return:            Dataframe containing PID results for each combination of sources and targets
    '''

    ############################
    # Prepare and set data
    ############################
    # Concatenate all sessions
    dataRSP = np.concatenate(dataLst, axis=0)  # Concatenate trials and sessions
    dataRP = np.mean(dataRSP, axis=1)  # Average out time

    if nDropPCA is not None:
        dataRP = drop_PCA(dataRP, nDropPCA)

    dataBin = bin_data(dataRP, nBin, axis=1)  # Bin data separately for each channel
    nTrial, nChannel = dataBin.shape

    if verbose:
        print("Data shape:", dataBin.shape)

    settings_estimator = {'pid_estimator': 'TartuPID', 'lags_pid': [0, 0]}
    mc.set_data(dataBin, 'rp')

    ############################
    # Prepare and set data
    ############################

    if verbose:
        print("Permutation-Testing...")

    # Since all channels are binned to the same quantiles,
    # the permutation test is exactly the same for all of them, so we need any three channels as input
    if nPerm > 0:
        settings_test = {'src': [0, 1], 'trg': 2, 'settings_estimator': settings_estimator}
        fTest = lambda x: bivariate_pid_3D(x, settings_test)
        dataTest = dataBin[:, :3][..., None]  # Add fake 1D sample dimension
        fRand = perm_test_resample(fTest, dataTest, nPerm, iterAxis=1)
    else:
        fRand = None

    if verbose:
        print("Computing PID...")

    if (labelsSrc is None) and (labelsTrg is None) and (labelsAll is None):
        ###############################
        # Loop over all possible targets excluding sources
        ###############################

        # Find indices of channel labels
        # channelIdxs = np.arange(len(labelsAll)).astype(int)

        # Find combinations of all source pairs
        channelIdxTriplets = list(iter_gn_3D(nChannel))

        rez = mc.metric3D('BivariatePID', '',
                          metricSettings={'settings_estimator': settings_estimator, 'shuffle': permuteTarget},
                          sweepSettings={'channels': channelIdxTriplets})

        if nPerm == 0:
            # 2D result: (triplets, pidType)
            return rez
        else:
            # 3D result: (triplets, pidType, statType);;  statType == {'p', 'effSize', 'muTrue', 'muRand'}
            return np.array([
                percentile_twosided(rezThis, fRand, settings={"haveEffectSize": True, "haveMeans": True})[1:].T
                for rezThis in rez
            ])

    elif (labelsSrc is not None) and (labelsTrg is not None) and (labelsAll is not None):
        ###############################
        # Loop over targets in target list
        ###############################

        # Find indices of channel labels
        sourceIdxs = [labelsAll.index(s) for s in labelsSrc]
        targetIdxs = [labelsAll.index(t) for t in labelsTrg]

        # Find combinations of all source pairs
        nSrc = len(sourceIdxs)
        nTrg = len(targetIdxs)
        sourceIdxPairs = [sourceIdxs[it] for it in iter_g_2D(nSrc)]
        nPairs = len(sourceIdxPairs)

        rez = mc.metric3D('BivariatePID', '',
                          metricSettings={'settings_estimator': settings_estimator, 'shuffle': permuteTarget},
                          sweepSettings={'src': sourceIdxPairs, 'trg': targetIdxs})
        # rez = mc.metric3D('BivariatePID', '',
        #                   metricSettings={'settings_estimator': settings_estimator, 'src': sourceIdxs},
        #                   sweepSettings={'trg': targetIdxs})

        df = None
        for iSrcPair, (iS1, iS2) in enumerate(sourceIdxPairs):
            for iTrg, labelTrg in enumerate(labelsTrg):
                if (nPairs == 1) and (nTrg == 1):
                    rezThis = rez
                else:
                    rezThis = rez[iSrcPair, iTrg]

                df = _update_pid_rez_df(df, rezThis, fRand, labelsAll[iS1], labelsAll[iS2], labelTrg)
        return df
    else:
        raise ValueError('Must provide both source and target indices or neither')


def hypotheses_calc_pid(dataDB, mc, hDict, h5outname, datatypes=None, nDropPCA=None, **kwargs):
    if datatypes is None:
        datatypes = dataDB.get_data_types()

    for datatype in datatypes:
        for hLabel, (intervName, sources, targets) in hDict.items():
            print(hLabel)

            # Calculate PID
            rezDict = {}
            for mousename in dataDB.mice:
                channelNames = dataDB.get_channel_labels(mousename)
                dataLst = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype,
                                                zscoreDim='rs', intervName=intervName, **kwargs)

                rezDict[(mousename,)] = pid(dataLst, mc, channelNames, sources, targets,
                                            nPerm=2000, nBin=4, nDropPCA=nDropPCA)

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


def hypotheses_calc_plot_info3D(dataDB, hDict, datatypes=None, nBin=4, nDropPCA=None, **kwargs):
    if datatypes is None:
        datatypes = dataDB.get_data_types()

    for datatype in datatypes:
        for hLabel, (intervName, sources, targets) in hDict.items():
            print(hLabel)

            dataLabel = '_'.join(['PID', datatype, hLabel, intervName])

            # Find combinations of all source pairs
            nSrc = len(sources)
            sourcePairs = [sources[it] for it in iter_g_2D(nSrc)]
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
                                                        datatype=datatype, intervName=intervName, **kwargs)

                        dataRSP = np.concatenate(dataLst, axis=0)  # Concatenate all sessions
                        dataRP = np.mean(dataRSP, axis=1)  # Average out time

                        if nDropPCA is not None:
                            dataRP = drop_PCA(dataRP, nDropPCA)

                        dataBin = bin_data(dataRP, nBin, axis=1)  # Binarize data over channels

                        h3d = np.histogramdd(dataBin[:, [targetIdx, s1Idx, s2Idx]], bins=(nBin, nBin, nBin))[0]
                        h3d /= np.sum(h3d)  # Normalize

                        for iTrgBin in range(nBin):
                            img = ax[iMouse][iTrgBin].imshow(h3d[iTrgBin], vmin=0, vmax=10 / nBin ** 3, cmap='jet')
                            ax[iMouse][iTrgBin].set_ylabel(s1Label)  # First label is rows a.k.a Y-AXIS!!!
                            ax[iMouse][iTrgBin].set_xlabel(s2Label)
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
        'session' : '_'.join(lst[5:-1]),
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


# Replace U1 and U2 with unique, swap sources for U2
def preprocess_unique(df):
    dfRez = df.copy()
    u2idxs = dfRez['PID'] == 'U2'

    # tmp = dfRez[u2idxs]['S1']
    # dfRez.loc[u2idxs, 'S1'] = dfRez[u2idxs]['S2']
    # dfRez.loc[u2idxs, 'S2'] = tmp

    # Swap sources so that first unique is always the significant one
    dfRez.loc[u2idxs, ['S1', 'S2']] = dfRez.loc[u2idxs, ['S2', 'S1']].values

    dfRez.replace({'PID': {'U1': 'unique', 'U2': 'unique'}}, inplace=True)
    return dfRez


def preprocess_drop_negative(df):
    df['muTrue'] = np.clip(df['muTrue'], 0, None)
    df['muRand'] = np.clip(df['muRand'], 0, None)
    return df


def preprocess_drop_channels(df, channelLst):
    dfRez = df.copy()

    for ch in channelLst:
        dfRez = dfRez[~(dfRez['S1'] == ch)]
        dfRez = dfRez[~(dfRez['S2'] == ch)]
        dfRez = dfRez[~(dfRez['T'] == ch)]
    return dfRez


# Plot fraction of significant PID's for each session from H5 storage of all-to-all pid file
def plot_all_frac_significant_bysession(dataDB, h5fname, minTrials=50, trialType='iGO'):
    pidTypes = {'unique', 'red', 'syn'}
    summaryDF = pid_all_summary_df(h5fname)

    for keyLst, dfSession in summaryDF.groupby(['mousename', 'datatype', 'phase']):
        keyLabel = '_'.join(keyLst)
        print(keyLabel)

        pidDict = defaultdict(list)
        for idx, row in dfSession.iterrows():
            nTrialsThis = dataDB.get_ntrial_bytype({'session': row['session']}, trialType=trialType)
            if nTrialsThis < minTrials:
                print('Skipping session', row['session'], 'because it has too few trials', nTrialsThis)

                for pidType in pidTypes:
                    pidDict[pidType] += [np.nan]

            else:
                df1 = pd.read_hdf(h5fname, key=row['key'])

                # Merge Unique1 and Unique2 for this plot
                df1 = preprocess_unique(df1)

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
def plot_all_results_distribution(dataDB, h5fname, plotstyle='cdf', minTrials=50, trialType='iGO'):
    pidTypes = {'unique', 'red', 'syn'}
    summaryDF = pid_all_summary_df(h5fname)

    for idx, row in summaryDF.iterrows():
        nTrialsThis = dataDB.get_ntrial_bytype({'session': row['session']}, trialType=trialType)
        if nTrialsThis < minTrials:
            print('Skipping session', row['session'], 'because it has too few trials', nTrialsThis)
        else:
            keyLabel = '_'.join([row['mousename'], row['session'], row['datatype'], row['phase']])
            print(keyLabel)

            df1 = pd.read_hdf(h5fname, key=row['key'])

            # Merge Unique1 and Unique2 for this plot
            df1 = preprocess_unique(df1)
            df1 = preprocess_drop_negative(df1)

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
                cdf_labeled(ax[0], pVals, pidTypes, 'pVal', haveLog=True)
                cdf_labeled(ax[1], effSizes, pidTypes, 'effSize', haveLog=False)
                cdf_labeled(ax[2], values, pidTypes, 'Bits', haveLog=False)
                plt.savefig('PID_CDF_' + keyLabel + '.png')
            else:
                raise ValueError('Unexpected plot style', plotstyle)

            plt.close()


# Scatter fraction of significant PID's vs performance from H5 storage of all-to-all pid file
def plot_all_frac_significant_performance_scatter(dataDB, h5fname, minTrials=50):
    pidTypes = ['unique', 'red', 'syn']
    summaryDF = pid_all_summary_df(h5fname)

    for keyLst, dfSession in summaryDF.groupby(['datatype', 'phase']):    # 'mousename'
        keyLabel = '_'.join(keyLst)

        pidDFAll = pd.DataFrame(columns=['pid', 'x', 'y', 'mouse'])
        pidDictNaive = defaultdict(list)
        pidDictExpert = defaultdict(list)
        fig, ax = plt.subplots(nrows=2, ncols=len(pidTypes), figsize=(len(pidTypes) * 4, 8), tight_layout=True)
        for keyLst2, dfSession2 in dfSession.groupby(['mousename']):  # 'mousename'
            print(keyLabel, keyLst2)

            for idx, row in dfSession2.iterrows():
                nTrialsThis = dataDB.get_ntrial_bytype({'session': dfSession['session']}, trialType='iGO')
                if nTrialsThis < minTrials:
                    print('Skipping session', dfSession['session'], 'because it has too few trials', nTrialsThis)
                else:
                    perf = dataDB.get_performance(row['session'])

                    df1 = pd.read_hdf(h5fname, key=row['key'])

                    # Merge Unique1 and Unique2 for this plot
                    df1 = preprocess_unique(df1)

                    for pidType in pidTypes:
                        pVal = df1[df1['PID'] == pidType]['p']

                        fracSig = np.mean(pVal < 0.01)
                        pidDFAll = pd_append_row(pidDFAll, [pidType, perf, fracSig, keyLst2])
                        # pidDFAll[pidType] += [(perf, fracSig, keyLst2)]
                        if perf < 0.7:
                            pidDictNaive[pidType] += [fracSig]
                        else:
                            pidDictExpert[pidType] += [fracSig]

        for iFig, pidType in enumerate(pidTypes):
            ax[0, iFig].set_title(pidType)
            ax[0, iFig].set_xlabel('Performance')
            ax[0, iFig].set_ylabel('Fraction Significant')

            dataThis = pidDFAll[pidDFAll['pid'] == pidType]
            sns.scatterplot(ax=ax[0, iFig], data=dataThis, x='x', y='y', hue='mouse')
            sns.regplot(ax=ax[0, iFig], data=dataThis, x='x', y='y', scatter=False)
            ax[0, iFig].set_ylim(-0.01, 1.01)

            # sns.regplot(x=x, y=y, ax=ax[0, iFig])
            # ax[0, iFig].plot(*np.array(pidDictSession[pidType]).T, '.', label=str(keyLst2))

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


# Compute a list of significant triplets for each pidType and mouse
# FIXME: Impl original name label convertion
def _get_pid_sign_dict(dataDB, keyLabel, dfSession, h5fname, pidTypes, minTrials=50, trialType='iGO'):
    # Init dictionary
    mouseSignDict = {}
    for pidType in pidTypes:
        mouseSignDict[pidType] = {}

    for mousename, dfSession2 in dfSession.groupby(['mousename']):
        print(keyLabel, mousename)
        channelLabels = np.array(dataDB.get_channel_labels(mousename))

        # For each session and each PIDtype, find significant triples and stack their indices
        signPidDict = {pidType : np.zeros((0, 3), dtype=int) for pidType in pidTypes}
        nSessionGood = 0
        for idx, row in dfSession2.iterrows():
            nTrialsThis = dataDB.get_ntrial_bytype({'session': row['session']}, trialType=trialType)
            if nTrialsThis < minTrials:
                print('Skipping session', row['session'], 'because it has too few trials', nTrialsThis)
            else:
                nSessionGood += 1
                df1 = pd.read_hdf(h5fname, key=row['key'])

                # Merge Unique1 and Unique2
                df1 = preprocess_unique(df1)

                # Convert all channel labels to indices
                # for iCh, ch in enumerate(channelLabels):
                #     df1.replace({'S1': {ch: iCh}, 'S2': {ch: iCh}, 'T': {ch: iCh}})
                chMap = {ch: iCh for iCh, ch in enumerate(channelLabels)}
                df1.replace({'S1': chMap, 'S2': chMap, 'T': chMap}, inplace=True)

                for pidType in pidTypes:
                    df1PID = df1[df1['PID'] == pidType]
                    df1Sig = df1PID[df1PID['p'] < 0.01]

                    x, y, z = df1Sig['S1'], df1Sig['S2'], df1Sig['T']
                    x2D = np.array([x,y,z], dtype=int).T

                    signPidDict[pidType] = np.vstack([signPidDict[pidType], x2D])

        # For each PIDtype, for each significant triple, count in how many sessions it is significant
        for pidType in pidTypes:
            x2D = signPidDict[pidType]
            idxs, counts = np.unique(x2D, return_counts=True, axis=0)
            fractions = counts.astype(float) * 100 / nSessionGood       # Convert to % to account for diff session number in each mouse
            idxs = np.array(idxs)

            idxNamesX = channelLabels[idxs[:, 0]]
            idxNamesY = channelLabels[idxs[:, 1]]
            idxNamesZ = channelLabels[idxs[:, 2]]
            idxNames = list(zip(idxNamesX, idxNamesY, idxNamesZ))

            mouseSignDict[pidType][mousename] = [idxNames, fractions]

    return mouseSignDict


# Plot top N most significant triplets over all mice. Stack barplots for individual mice
# FIXME: Impl original name label convertion
def plot_all_frac_significant_3D_top_n(dataDB, mouseSignDict, keyLabel, pidTypes, nTop=10):
    fig, ax = plt.subplots(ncols=len(pidTypes), figsize=(len(pidTypes) * 4, 4), tight_layout=True)
    for iPid, pidType in enumerate(pidTypes):
        '''
          1. Find all nonzero triplets over all mice
          2. Append zeros for zero triplets for each mouse
          3. For each mouse, sort all triplets in alphabetic order
          4. Add counts for each mouse, get total
          5. Sort total in descending order
          6. Sort each mouse in descending order
          7. Barplot
        '''

        keysAsStr = {mousename : [str(t) for t in data[0]] for mousename, data in mouseSignDict[pidType].items()}

        setNonZeroKeys = set.union(*[set(key) for key in keysAsStr.values()])
        lstNonZeroKeysAlphabetical = sorted([str(t) for t in setNonZeroKeys])

        countsAlphabetical = {}
        for mousename, mouseData in mouseSignDict[pidType].items():
            idxNames = keysAsStr[mousename]
            idxCounts = mouseData[1]
            missingNames = list(setNonZeroKeys - set(idxNames))
            idxNamesNew = np.concatenate([idxNames, missingNames], axis=0)
            idxCountsNew = np.concatenate([idxCounts, np.zeros(len(missingNames))])
            sortIdxs = np.argsort(idxNamesNew, axis=0)
            countsAlphabetical[mousename] = idxCountsNew[sortIdxs]

        countsTot = np.sum(list(countsAlphabetical.values()), axis=0)
        sortIdxs = np.argsort(countsTot)[::-1]
        # countsTotSorted = countsTot[sortIdxs]
        lstNonZeroKeysSorted = np.array(lstNonZeroKeysAlphabetical)[sortIdxs]
        countsSorted = {k: v[sortIdxs] for k, v in countsAlphabetical.items()}

        barplot_stacked_indexed(ax[iPid], countsSorted, xTickLabels=lstNonZeroKeysSorted, xLabel='triplet',
                                yLabel='4*Percent', title=pidType, iMax=nTop, rotation=90)

        # 5) Consider some significance test maybe

    plt.savefig('PID_top_' + str(nTop) + '_triplets_frac_significant_' + keyLabel + '.pdf')
    plt.close()


# Plot top N targets with most total significant connections
def plot_all_frac_significant_1D_top_n(dataDB, mouseSignDict, keyLabel, pidTypes, nTop=10):
    # FIXME: Currently relies on same dimension ordering for all mice
    lmap = dataDB.map_channel_labels_canon()
    channelLabels = dataDB.get_channel_labels()
    channelLabelsCanon = [lmap[l] for l in channelLabels]

    nChannel = len(channelLabels)

    fig, ax = plt.subplots(nrows=len(pidTypes), figsize=(len(pidTypes) * 6, 12), tight_layout=True)
    for iPid, pidType in enumerate(pidTypes):
        factor = (nChannel - 1) * (nChannel - 2) / 2     # Exclude itself
        if pidType == 'unique':
            factor *= 2.0

        rezDict = {}
        for mousename in dataDB.mice:
            print(pidType, mousename, len(mouseSignDict[pidType][mousename][0]), factor)

            # Convert to dataframe
            df = pd.DataFrame(mouseSignDict[pidType][mousename][0], columns=['S1', 'S2', 'T'])
            df['Val'] = mouseSignDict[pidType][mousename][1]

            # For each target, average fraction of significant sessions over all possible source pairs
            rez = []
            for label in channelLabels:
                rez += [np.sum(df[df['T'] == label]['Val']) / factor]

            rezDict[mousename] = rez

        # 4) Plot stacked barplot with absolute numbers. Set ylim_max to total number of sessions
        barplot_stacked_indexed(ax[iPid], rezDict, xTickLabels=channelLabelsCanon, xLabel='singlet',
                                yLabel='fraction of connections', title=pidType, iMax=None, rotation=90)

        update_fonts_axis(ax[iPid], 20)

    plt.savefig('PID_top_' + str(nTop) + '_singlets_frac_significant_' + keyLabel + '.png')
    plt.close()


def _sign_dict_to_3D_mat(dataDB, mouseSignDict, pidType):
    labels = dataDB.get_channel_labels()
    labelDict = {l: i for i, l in enumerate(labels)}
    nChannel = len(labels)

    Mrez = np.zeros((nChannel, nChannel, nChannel))
    for mousename in dataDB.mice:
        # Convert stored list to dataframe
        df = pd.DataFrame(mouseSignDict[pidType][mousename][0], columns=['S1', 'S2', 'T'])
        df['fr'] = mouseSignDict[pidType][mousename][1]

        # Select target channel
        # df = df[df['T'] == trgChName].drop('T', axis=1)

        # Rename channels back to indices
        # df.replace({'S1': labelDict, 'S2': labelDict}, inplace=True)
        df.replace({'S1': labelDict, 'S2': labelDict, 'T': labelDict}, inplace=True)

        # Construct as matrix
        M = np.zeros((nChannel, nChannel, nChannel))
        M[df['S1'], df['S2'], df['T']] = df['fr']

        if pidType != 'unique':
            # M += M.T
            M += M.transpose((1,0,2))
        Mrez += M

    Mrez = Mrez / len(dataDB.mice) / 100.0
    return Mrez


# Plot a matrix of fraction of significant connections for a target channel
def plot_all_frac_significant_2D_avg(dataDB, mouseSignDict, keylabel, pidTypes):
    for pidType in pidTypes:
        Mrez3D = _sign_dict_to_3D_mat(dataDB, mouseSignDict, pidType)
        nChannel = Mrez3D.shape[0]
        Mrez2D = np.sum(Mrez3D, axis=2) / (nChannel-2)   # Target can't be either of the sources

        print(np.max(Mrez2D))

        plt.figure()
        plt.imshow(Mrez2D, cmap='jet', vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig('PID_2D_' + '_'.join([pidType, 'AVG', keylabel]) + '.png')
        plt.close()


# Plot a matrix of fraction of significant connections for a target channel
def plot_all_frac_significant_2D_by_target(dataDB, mouseSignDict, keylabel, pidType, trgChName, vmax=0.5):
    labels = dataDB.get_channel_labels()
    trgIdx = labels.index(trgChName)
    Mrez3D = _sign_dict_to_3D_mat(dataDB, mouseSignDict, pidType)
    Mrez2D = Mrez3D[:, :, trgIdx]

    print(np.max(Mrez2D))

    plt.figure()
    plt.imshow(Mrez2D, cmap='jet', vmin=0, vmax=vmax)
    plt.colorbar()
    plt.savefig('PID_2D_' + '_'.join([pidType, trgChName, keylabel]) + '.png')
    plt.close()

