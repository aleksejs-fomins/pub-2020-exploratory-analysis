import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
from idtxl.bivariate_pid import BivariatePID
from idtxl.data import Data

from mesostat.utils.arrays import unique_ordered
from mesostat.utils.pandas_helper import merge_df_from_dict
from mesostat.utils.decorators import redirect_stdout
from mesostat.visualization.mpl_matrix import imshow


##############################
# Auxiliary Functions
##############################

def shuffle(x):
    x1 = x.copy()
    np.random.shuffle(x1)
    return x1


def bin_data_1D(data, nBins):
    boundaries = np.quantile(data, np.linspace(0, 1, nBins + 1))
    boundaries[-1] += 1.0E-10
    return np.digitize(data, boundaries, right=False) - 1


def pid_bin(x, y, z, nBins=4):
    dataEff = np.array([
        bin_data_1D(x, nBins),
        bin_data_1D(y, nBins),
        bin_data_1D(z, nBins)
    ])
    return pid(dataEff)


@redirect_stdout
def pid(dataPS):
    settings = {'pid_estimator': 'TartuPID', 'lags_pid': [0, 0]}

    dataIDTxl = Data(dataPS, dim_order='ps', normalise=False)
    pid = BivariatePID()
    rez = pid.analyse_single_target(settings=settings, data=dataIDTxl, target=2, sources=[0, 1])
    return rez.get_single_target(2)


def effect_size_by_method(df, dfRand):
    dfEffSize = pd.DataFrame()
    for method in unique_ordered(df['Method']):
        dfMethodTrue = df[df['Method'] == method]
        dfMethodRand = dfRand[dfRand['Method'] == method]

        muRand = np.mean(dfMethodRand['Value'])
        stdRand = np.std(dfMethodRand['Value'])

        dfMethodEff = dfMethodTrue.copy()
        dfMethodEff['Value'] = (dfMethodEff['Value'] - muRand) / stdRand

        dfEffSize = dfEffSize.append(dfMethodEff)
    return dfEffSize

##############################
# Functions
##############################


def run_tests(datagen_func, decompFunc, decompLabels, nTest=100, haveShuffle=False):
    rezDict = {k: [] for k in decompLabels}
    for iTest in range(nTest):
        x, y, z = datagen_func()
        zEff = z if not haveShuffle else shuffle(z)

        rez = decompFunc(x, y, zEff)

        for k in rezDict.keys():
            rezDict[k] += [rez[k]]

    rezDF = pd.DataFrame()

    for iLabel, label in enumerate(decompLabels):
        rezTmp = pd.DataFrame({'Method': [label] * nTest, 'Value': rezDict[label]})
        rezDF = rezDF.append(rezTmp)

    return rezDF


def plot_test_summary(df, dfRand, suptitle=None, haveEff=True, logEff=False):
    nFig = 3 if haveEff else 2
    fig, ax = plt.subplots(ncols=nFig, figsize=(4*nFig, 4), tight_layout=True)
    if suptitle is not None:
        fig.suptitle(suptitle)

    # Plot 1: True vs Random
    dfMerged = merge_df_from_dict({'True': df, 'Random': dfRand}, columnNames=['Kind'])
    sns.violinplot(ax=ax[0], x="Method", y="Value", hue="Kind", data=dfMerged, scale='width')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Metric Value')

    # Calculate effect sizes
    dfEffSize = effect_size_by_method(df, dfRand)

    # Plot 2: Effect Sizes
    if haveEff:
        sns.violinplot(ax=ax[1], x="Method", y="Value", data=dfEffSize, scale='width')
        if logEff:
            ax[1].set_yscale('log')
        # ax[1].axhline(y='2', color='pink', linestyle='--')
        ax[1].set_xlabel('')
        ax[1].set_ylabel('Effect Size')

    # Calculate fraction significant
    sigDict = {}
    for method in unique_ordered(df['Method']):
        dfEffMethod = dfEffSize[dfEffSize['Method'] == method]
        sigDict[method] = [np.mean(dfEffMethod['Value'] > 2)]

    # Plot 3: Fraction significant
    idx3 = 2 if haveEff else 1
    sns.barplot(ax=ax[idx3], data=pd.DataFrame(sigDict))
    ax[idx3].set_ylim([0, 1])
    ax[idx3].set_xlabel('')
    ax[idx3].set_ylabel('Fraction Significant')


def run_plot_param_effect(datagen_func, decompFunc, decompLabels, nTest=1000, alphaRange=(0, 1)):
    alphaLst = []
    rezLst = []
    for iTest in range(nTest):
        alpha = np.random.uniform(*alphaRange)
        x, y, z = datagen_func(alpha)
        rez = decompFunc(x, y, z)
        rez = [rez[k] for k in decompLabels]

        alphaLst += [alpha]
        rezLst += [rez]

    rezLst = np.array(rezLst)

    fig, ax = plt.subplots(ncols=4, figsize=(16, 4))
    for iKind, kindLabel in enumerate(decompLabels):
        ax[iKind].set_title(kindLabel)

        ax[iKind].semilogy(alphaLst, rezLst[:, iKind], '.')
        ax[iKind].set_ylim([1.0E-7, 1])


def run_plot_param_effect_test(datagen_func, decompFunc, decompLabels, nStep=10, nTest=1000, alphaRange=(0, 1)):
    alphaLst = np.linspace(*alphaRange, nStep)

    dfRezDict = {}
    for alpha in alphaLst:
        gen_data_eff = lambda: datagen_func(alpha)

        rezDF   = run_tests(gen_data_eff, decompFunc, decompLabels, nTest=nTest)
        rezDFsh = run_tests(gen_data_eff, decompFunc, decompLabels, nTest=nTest, haveShuffle=True)
        dfEffSize = effect_size_by_method(rezDF, rezDFsh)

        dfRezDict[(np.round(alpha, 2), )] = dfEffSize

    dfRez = merge_df_from_dict(dfRezDict, ['alpha'])

    nMethods = len(decompLabels)
    fig, ax = plt.subplots(nrows=2, ncols=nMethods, figsize=(4*nMethods, 8), tight_layout=True)
    for iMethod, methodName in enumerate(decompLabels):
        # Compute plot effect sizes
        dfRezMethod = dfRez[dfRez['Method'] == methodName]
        sns.violinplot(ax=ax[0, iMethod], x="alpha", y="Value", data=dfRezMethod, scale='width')
        ax[0, iMethod].set_xticklabels(ax[0, iMethod].get_xticklabels(), rotation = 90)
        ax[0, iMethod].set_xlabel('')
        ax[0, iMethod].set_title(methodName)

        # Compute plot thresholded effect sizes
        valDict = {}
        for alpha, dfSig in dfRezDict.items():
            dfSigMethod = dfSig[dfSig['Method'] == methodName]
            valDict[alpha[0]] = [np.mean(dfSigMethod['Value'] > 2)]

        valDF = pd.DataFrame(valDict)
        sns.barplot(ax=ax[1, iMethod], data=valDF)
        ax[1, iMethod].set_xticklabels(ax[1, iMethod].get_xticklabels(), rotation=90)
        ax[1, iMethod].set_ylim(0, 1.05)
        ax[0, iMethod].set_xlabel('$\sigma$')

    ax[0, 0].set_ylabel('Effect Size')
    ax[1, 0].set_ylabel('Fraction Significant')


def run_plot_data_effect_test(datagen_func, decompFunc, decompLabels, nStep=10, nTest=1000):
    nSampleLst = (10 ** np.linspace(2, 5, nStep)).astype(int)

    dfRezDict = {}
    for nSample in nSampleLst:
        gen_data_eff = lambda: datagen_func(nSample)

        rezDF   = run_tests(gen_data_eff, decompFunc, decompLabels, nTest=nTest)
        rezDFsh = run_tests(gen_data_eff, decompFunc, decompLabels, nTest=nTest, haveShuffle=True)
        dfEffSize = effect_size_by_method(rezDF, rezDFsh)

        dfRezDict[(nSample, )] = dfEffSize

    dfRez = merge_df_from_dict(dfRezDict, ['nSample'])

    nMethods = len(decompLabels)
    fig, ax = plt.subplots(nrows=2, ncols=nMethods, figsize=(4*nMethods, 8), tight_layout=True)
    for iMethod, methodName in enumerate(decompLabels):
        ax[0, iMethod].set_title(methodName)

        # Compute plot effect sizes
        dfRezMethod = dfRez[dfRez['Method'] == methodName]
        sns.violinplot(ax=ax[0, iMethod], x="nSample", y="Value", data=dfRezMethod, scale='width')
        ax[0, iMethod].set_xticklabels(ax[0, iMethod].get_xticklabels(), rotation = 90)
        ax[0, iMethod].set_xlabel('')

        # Compute plot thresholded effect sizes
        valDict = {}
        for nSampleTuple, dfSample in dfRezDict.items():
            dfSigMethod = dfSample[dfSample['Method'] == methodName]
            valDict[nSampleTuple[0]] = [np.mean(dfSigMethod['Value'] > 2)]

        valDF = pd.DataFrame(valDict)
        sns.barplot(ax=ax[1, iMethod], data=valDF)
        ax[1, iMethod].set_xticklabels(ax[1, iMethod].get_xticklabels(), rotation=90)
        ax[1, iMethod].set_ylim(0, 1.05)
        ax[0, iMethod].set_xlabel('$\sigma$')

    ax[0, 0].set_ylabel('Effect Size')
    ax[1, 0].set_ylabel('Fraction Significant')


##############################
# Synergy-Redundancy distribution search
##############################

def run_plot_scatter_explore(datagen_func, decompFunc, labelA, labelB, nVars, varLimits=(0, 1), nSample=1000, nTestDim=10):
    rezALst = []
    rezBLst = []

    sTmp = 0
    sVars = 0

    x1 = np.linspace(*varLimits, nTestDim)
    prodIt = itertools.product(*[x1]*nVars)

    for vars in prodIt:
        # vars = np.random.uniform(*varLimits, nVars)
        x, y, z = datagen_func(nSample, *vars)
        rez = decompFunc(x, y, z)

        rezALst += [rez[labelA]]
        rezBLst += [rez[labelB]]

        # if rez[labelA] >= 1:
        #     print(vars)

        if rez[labelB] > sTmp:
            sTmp = rez[labelB]
            sVars = vars

    print('maxSyn', sTmp, sVars)

    plt.figure()
    plt.plot(rezALst, rezBLst, '.')
    plt.xlabel(labelA)
    plt.ylabel(labelB)
    # plt.title('Synergy-Redundancy relationship for noisy redundant model')
    plt.show()


def run_plot_scatter_exact(datagen_func, decompFunc, labelA, labelB, vars, nSample=1000, nTest=1000):
    rezALst = []
    rezBLst = []

    for iTest in range(nTest):
        x, y, z = datagen_func(nSample, *vars)
        rez = decompFunc(x, y, z)

        rezALst += [rez[labelA]]
        rezBLst += [rez[labelB]]

    plt.figure()
    plt.plot(rezALst, rezBLst, '.')
    plt.xlabel(labelA)
    plt.ylabel(labelB)
    # plt.title('Synergy-Redundancy relationship for noisy redundant model')
    plt.show()


def run_plot_2D_scan(datagen_func, decompFunc, labelA, labelB, varLimits=(0, 1), nSample=1000, nStep=10, nTest=20):
    rezAMat = np.zeros((nStep, nStep))
    rezBMat = np.zeros((nStep, nStep))

    alphaLst = np.linspace(*varLimits, nStep)

    for iAlpha, alphaX in enumerate(alphaLst):
        for jAlpha, alphaY in enumerate(alphaLst):

            tmpA = []
            tmpB = []
            for iTest in range(nTest):
                x, y, z = datagen_func(nSample, alphaX, alphaY, 0)
                rez = decompFunc(x, y, z)

                tmpA += [rez[labelA]]
                tmpB += [rez[labelB]]

            rezAMat[iAlpha][jAlpha] = np.mean(tmpA)
            rezBMat[iAlpha][jAlpha] = np.mean(tmpB)

    # Find and report maximal synergy point
    iAlphaMax, jAlphaMax = np.unravel_index(np.argmax(rezBMat), rezBMat.shape)
    print('maxSyn', np.max(rezBMat), 'red', rezAMat[iAlphaMax][jAlphaMax], 'alpha', alphaLst[iAlphaMax], alphaLst[jAlphaMax])

    # Find distribution at maximal synergy point
    rezDict = {labelA: [], labelB: []}
    for iTest in range(1000):
        x, y, z = datagen_func(nSample, alphaLst[iAlphaMax], alphaLst[jAlphaMax], 0)
        rez = decompFunc(x, y, z)
        rezDict[labelA] += [rez[labelA]]
        rezDict[labelB] += [rez[labelB]]
    dfMax = pd.DataFrame(rezDict)

    print('1% quantile max synergy', np.quantile(rezDict[labelB], 0.99))

    fig, ax = plt.subplots(ncols=3, figsize=(12,4), tight_layout=True)
    imshow(fig, ax[0], rezAMat, title=labelA, haveColorBar=True)
    imshow(fig, ax[1], rezBMat, title=labelB, haveColorBar=True)
    sns.violinplot(ax=ax[2], data=dfMax)
    plt.show()


def run_plot_1D_scan(datagen_func, decompFunc, labelA, labelB, varLimits=(0, 1), nSample=1000, nStep=100, nTest=20):
    rezAMuLst = []
    rezBMuLst = []
    rezAStdLst = []
    rezBStdLst = []

    alphaLst = np.linspace(*varLimits, nStep)
    for alpha in alphaLst:
        aTmp = []
        bTmp = []
        for iTest in range(nTest):
            x, y, z = datagen_func(nSample, alpha, alpha, 0)
            rez = decompFunc(x, y, z)

            aTmp += [rez[labelA]]
            bTmp += [rez[labelB]]

        rezAMuLst += [np.mean(aTmp)]
        rezBMuLst += [np.mean(bTmp)]
        rezAStdLst += [np.std(aTmp)]
        rezBStdLst += [np.std(bTmp)]

    print('maxSyn', np.max(rezBMuLst), np.argmax(rezBMuLst))

    plt.figure()
    plt.errorbar(alphaLst, rezAMuLst, rezAStdLst, label=labelA)
    plt.errorbar(alphaLst, rezBMuLst, rezBStdLst, label=labelA)
    plt.xlabel('Parameter values')
    plt.ylabel('Function values')
    # plt.title('Synergy-Redundancy relationship for noisy redundant model')
    plt.show()