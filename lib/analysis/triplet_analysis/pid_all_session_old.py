import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

from mesostat.utils.pandas_helper import pd_append_row, pd_query, merge_df_from_dict
from mesostat.visualization.mpl_violin import violins_labeled
from mesostat.visualization.mpl_cdf import cdf_labeled
from mesostat.visualization.mpl_barplot import barplot_stacked, barplot_stacked_indexed
from mesostat.visualization.mpl_font import update_fonts_axis


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
