import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import mannwhitneyu

from mesostat.stat.stat import continuous_empirical_CDF
from mesostat.utils.pandas_helper import pd_query, pd_merge_multiple

from mesostat.visualization.mpl_barplot import barplot_stacked_indexed
from mesostat.visualization.mpl_matrix import imshow

from lib.analysis.pid import preprocess_unique, preprocess_drop_channels, pid


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


def read_parse_joint_dataframe(df, h5fname, mice, pidTypes, dropChannels=None):
    dfDict = defaultdict(list)
    for idx, row in df.sort_values('mousename', axis=0).iterrows():
        df = pd.read_hdf(h5fname, row['key'])
        df = preprocess_unique(df)
        if dropChannels is not None:
            df = preprocess_drop_channels(df, dropChannels)

        for pidType in pidTypes:
            dfDict[pidType] += [df[df['PID'] == pidType].drop(['PID', 'p', 'effSize', 'muRand'], axis=1)]

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


def test_avg_bits(dataDB, mc, h5fname, dfSummary):
    channelLabels = dataDB.get_channel_labels()
    nChannel = len(channelLabels)
    pidTypes = ['unique', 'syn', 'red']

    for key, dataMouse in dfSummary.groupby(['datatype', 'phase', 'trialType']):
        datatype, phase, trialType = key
        if trialType == 'None':
            trialType = None

        print(key)
        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        fig.suptitle('_'.join(key))

        dfTot = pd.DataFrame()
        for idx, row in dataMouse.iterrows():
            dfRezTrue = pd.read_hdf(h5fname, row['key'])
            dfRezTrue.replace({'PID': {'U1': 'unique', 'U2': 'unique'}}, inplace=True)
            dfRezTrue['type'] = 'Measured'
            dfRezTrue['mousename'] = row['mousename']
            dfTot = dfTot.append(dfRezTrue)

            # 1. Find how many trials for these parameters
            dataLst = dataDB.get_neuro_data({'mousename': row['mousename']}, datatype=datatype, trialType=trialType)

            nTrial = np.concatenate(dataLst, axis=0).shape[0]

            # 2. Generate random dataset of same shape
            dataRPRand = np.random.uniform(0, 1, (nTrial, 1, nChannel))  # Add fake time dimension

            # 3. Evaluate PID for random dataset
            dfRezRand = pid([dataRPRand], mc, channelLabels, nPerm=0)
            dfRezRand.replace({'PID': {'U1': 'unique', 'U2': 'unique'}}, inplace=True)
            dfRezRand['type'] = 'Random'
            dfRezRand['mousename'] = row['mousename']
            dfTot = dfTot.append(dfRezRand)

        # 4. Perform t-test between true and random. Barplot by PID and Mouse
        for iPid, pidType in enumerate(pidTypes):
            dfPID = dfTot[dfTot['PID'] == pidType]
            sns.barplot(ax=ax[iPid], x="mousename", y="muTrue", hue="type", data=dfPID)
            ax[iPid].set_ylabel('Bits')

            # pVal = mannwhitneyu(pidTrue, pidRand, alternative='greater')[1]


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
        dfJointDict = read_parse_joint_dataframe(dataMouse, h5fname, mice, pidTypes, dropChannels=dropChannels)

        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        for iPid, pidType in enumerate(pidTypes):
            dfJoint = dfJointDict[pidType]
            meanColNames = list(set(dfJoint.columns) - set(dfJointDict[pidType].keys()))
            dfJoint['bits_mean'] = dfJoint[meanColNames].mean(axis=1)

            dfJoint = dfJoint.sort_values('bits_mean', axis=0, ascending=False)
            dfJointHead = dfJoint.head(nTop)

            labels = [str((s1,s2,t)) for s1,s2,t in zip(dfJointHead['S1'], dfJointHead['S2'], dfJointHead['T'])]
            rezDict = {mousename : np.array(dfJointHead['muTrue_' + mousename]) for mousename in mice}

            # 4) Plot stacked barplot with absolute numbers. Set ylim_max to total number of sessions
            barplot_stacked_indexed(ax[iPid], rezDict, xTickLabels=labels, xLabel='triplet',
                                    yLabel='bits', title=pidType, iMax=None, rotation=90)

        fig.suptitle('_'.join(key))
        plt.show()


def plot_singlets(dataDB, h5fname, dfSummary, nTop=20, dropChannels=None):
    pidTypes = ['unique', 'syn', 'red']

    for key, dataMouse in dfSummary.groupby(['datatype', 'phase', 'trialType']):
        mice = list(sorted(set(dataMouse['mousename'])))
        dfJointDict = read_parse_joint_dataframe(dataMouse, h5fname, mice, pidTypes, dropChannels=dropChannels)

        fig, ax = plt.subplots(nrows=len(pidTypes), figsize=(len(pidTypes) * 6, 12), tight_layout=True)
        for iPid, pidType in enumerate(pidTypes):
            dataLabels = dataDB.get_channel_labels()

            rezDict = {}
            for mousename in mice:
                rezTmp = []
                for label in dataLabels:
                    rezTmp += [np.mean(pd_query(dfJointDict[pidType], {'T': label})['muTrue_' + mousename])]
                rezDict[mousename] = rezTmp

            barplot_stacked_indexed(ax[iPid], rezDict, xTickLabels=dataLabels, xLabel='singlet',
                                    yLabel='bits', title=pidType, iMax=None, rotation=90)

        fig.suptitle('_'.join(key))
        plt.show()


def plot_2D_avg(dataDB, h5fname, dfSummary, dropChannels=None):
    pidTypes = ['unique', 'syn', 'red']

    for key, dataMouse in dfSummary.groupby(['datatype', 'phase', 'trialType']):
        mice = list(sorted(set(dataMouse['mousename'])))
        dfJointDict = read_parse_joint_dataframe(dataMouse, h5fname, mice, pidTypes, dropChannels=dropChannels)

        fig, ax = plt.subplots(ncols=3, figsize=(12, 4))
        for iPid, pidType in enumerate(pidTypes):
            Mrez3D = _bitdict_to_3Dmat(dataDB, dfJointDict, pidType, mice)

            nChannel = Mrez3D.shape[0]
            Mrez2D = np.sum(Mrez3D, axis=2) / (nChannel-2)   # Target can't be either of the sources

            print(np.max(Mrez2D))

            imshow(fig, ax[iPid], Mrez2D, title=pidType, haveColorBar=True, cmap='jet')

        fig.suptitle('_'.join(key))
        plt.show()


def plot_2D_bytarget(dataDB, h5fname, dfSummary, trgChName, dropChannels=None):
    pidTypes = ['unique', 'syn', 'red']
    labels = dataDB.get_channel_labels()
    trgIdx = labels.index(trgChName)

    for key, dataMouse in dfSummary.groupby(['datatype', 'phase', 'trialType']):
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