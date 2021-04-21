import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from mesostat.stat.stat import continuous_empirical_CDF
from mesostat.utils.pandas_helper import pd_query, pd_merge_multiple

from mesostat.visualization.mpl_barplot import barplot_stacked_indexed

from lib.analysis.pid import preprocess_unique, preprocess_drop_channels


def pid_all_parse_key(key):
    lst = key.split('_')
    return {
        'mousename': '_'.join(lst[1:3]),
        'datatype': '_'.join(lst[3:5]),
        'phase': lst[-2],
        'trialType': lst[-1]
    }


def pid_all_summary_df(h5fname):
    with h5py.File(h5fname, 'r') as f:
        keys = list(f.keys())

    summaryDF = pd.DataFrame()
    for key in keys:
        summaryDF = summaryDF.append(pd.DataFrame({**{'key': key}, **pid_all_parse_key(key)}, index=[0]))

    return summaryDF.reset_index(drop=True)


def cdfplot(h5fname, dfSummary):
    pidTypes = ['unique', 'syn', 'red']
    dataLabels = ['p', 'effSize', 'muTrue']
    haveLog = [True, False, False]

    for key, dataMouse in dfSummary.groupby(['datatype', 'phase', 'trialType']):
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


def scatter_effsize_bits(h5fname, dfSummary):
    pidTypes = ['unique', 'syn', 'red']
    dataLabels = ['p', 'effSize', 'muTrue']
    haveLog = [True, False, False]

    for key, dataMouse in dfSummary.groupby(['datatype', 'phase', 'trialType']):
        fig, ax = plt.subplots(ncols=3, figsize=(12, 4), tight_layout=True)
        fig.suptitle('_'.join(key))

        for idx, row in dataMouse.iterrows():
            dfRezThis = pd.read_hdf(h5fname, row['key'])

            # Merge Unique1 and Unique2 for this plot
            dfRezThis = preprocess_unique(dfRezThis)

            for iPid, pidType in enumerate(pidTypes):
                df1 = dfRezThis[dfRezThis['PID'] == pidType]

                ax[iPid].plot(df1['effSize'], df1['muTrue'], '.', label=row['mousename'])

        for i in range(3):
            ax[i].set_title(pidTypes[i])
            ax[i].set_xlabel('effSize')
            ax[i].set_ylabel('Bits')
            ax[i].legend()

        plt.show()


def plot_triplets(h5fname, dfSummary, nTop=20, dropChannels=None):
    pidTypes = ['unique', 'syn', 'red']

    for key, dataMouse in dfSummary.groupby(['datatype', 'phase', 'trialType']):
        # fig, ax = plt.subplots(ncols=3, figsize=(12, 4), tight_layout=True)
        # fig.suptitle('_'.join(key))

        mice = list(sorted(set(dataMouse['mousename'])))
        dfDict = defaultdict(list)
        for idx, row in dataMouse.sort_values('mousename', axis=0).iterrows():
            df = pd.read_hdf(h5fname, row['key'])
            df = preprocess_unique(df)
            if dropChannels is not None:
                df = preprocess_drop_channels(df, dropChannels)

            for pidType in pidTypes:
                dfDict[pidType] += [df[df['PID'] == pidType].drop(['PID', 'p', 'effSize', 'muRand'], axis=1)]

        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        for iPid, pidType in enumerate(pidTypes):
            dfJoint = pd_merge_multiple(mice, dfDict[pidType], ["S1", "S2", "T"])
            print(pidType, len(dfDict[pidType][0]), len(dfJoint))

            meanColNames = list(set(dfJoint.columns) - set(dfDict.keys()))
            dfJoint['bits_mean'] = dfJoint[meanColNames].mean(axis=1)

            dfJoint = dfJoint.sort_values('bits_mean', axis=0, ascending=False)
            dfJointHead = dfJoint.head(nTop)


            labels = [str((s1,s2,t)) for s1,s2,t in zip(dfJointHead['S1'], dfJointHead['S2'], dfJointHead['T'])]
            rezDict = {mousename : np.array(dfJointHead['muTrue_' + mousename]) for mousename in mice}



            # 4) Plot stacked barplot with absolute numbers. Set ylim_max to total number of sessions
            ax[iPid].set_title(pidType)

            barplot_stacked_indexed(ax[iPid], rezDict, xTickLabels=labels, xLabel='triplet',
                                    yLabel='bits', title=pidType, iMax=None, rotation=90)

        fig.suptitle('_'.join(key))
        plt.show()


def plot_singlets(dataDB, h5fname, dfSummary, nTop=20, dropChannels=None):
    pidTypes = ['unique', 'syn', 'red']

    for key, dataMouse in dfSummary.groupby(['datatype', 'phase', 'trialType']):
        # fig, ax = plt.subplots(ncols=3, figsize=(12, 4), tight_layout=True)
        # fig.suptitle('_'.join(key))

        mice = list(sorted(set(dataMouse['mousename'])))
        dfDict = defaultdict(list)
        for idx, row in dataMouse.sort_values('mousename', axis=0).iterrows():
            df = pd.read_hdf(h5fname, row['key'])
            df = preprocess_unique(df)
            if dropChannels is not None:
                df = preprocess_drop_channels(df, dropChannels)

            for pidType in pidTypes:
                dfDict[pidType] += [df[df['PID'] == pidType].drop(['PID', 'p', 'effSize', 'muRand'], axis=1)]

        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        for iPid, pidType in enumerate(pidTypes):
            dfJoint = pd_merge_multiple(mice, dfDict[pidType], ["S1", "S2", "T"])
            print(pidType, len(dfDict[pidType][0]), len(dfJoint))

            print(dfJoint.columns)

            # dataLabels = dataDB.get_channel_labels()
            #
            # rez = []
            # for label in dataLabels:
            #     rez += [np.mean(pd_query(dfJoint, {'T', label})['muTrue_'])]
            #
            # print(rez)

            #
            # meanColNames = list(set(dfJoint.columns) - set(dfDict.keys()))
            # dfJoint['bits_mean'] = dfJoint[meanColNames].mean(axis=1)
            #
            # dfJoint = dfJoint.sort_values('bits_mean', axis=0, ascending=False)
            # dfJointHead = dfJoint.head(nTop)
            #
            #
            # labels = [str((s1,s2,t)) for s1,s2,t in zip(dfJointHead['S1'], dfJointHead['S2'], dfJointHead['T'])]
            # rezDict = {mousename : np.array(dfJointHead['muTrue_' + mousename]) for mousename in mice}
            #
            #
            #
            # # 4) Plot stacked barplot with absolute numbers. Set ylim_max to total number of sessions
            # ax[iPid].set_title(pidType)
            #
            # barplot_stacked_indexed(ax[iPid], rezDict, xTickLabels=labels, xLabel='triplet',
            #                         yLabel='bits', title=pidType, iMax=None, rotation=90)

        fig.suptitle('_'.join(key))
        plt.show()