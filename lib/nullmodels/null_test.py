import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
# from collections import defaultdict

from mesostat.utils.arrays import unique_ordered
from mesostat.utils.pandas_helper import merge_df_from_dict

# from mesostat.visualization.mpl_matrix import imshow


##############################
# Auxiliary Functions
##############################

def shuffle(x):
    x1 = x.copy()
    np.random.shuffle(x1)
    return x1


def fraction_significant(df, dfRand, pVal, valThrDict=None):
    sigDict = {}
    for method in unique_ordered(df['Method']):
        dataTrueMethod = df[df['Method'] == method]
        dataRandMethod = dfRand[dfRand['Method'] == method]

        # print(method,
        #       np.min(dataTrueMethod['Value']),
        #       np.max(dataTrueMethod['Value']),
        #       np.min(dataRandMethod['Value']),
        #       np.max(dataRandMethod['Value'])
        #       )

        # Compute threshold based on shuffled data
        thr = np.quantile(dataRandMethod['Value'], 1 - pVal)

        # If available, also apply constant threshold to data magnitude
        # Choose bigger of the two thresholds
        if valThrDict is not None and valThrDict[method] is not None:
            thr = max(thr, valThrDict[method])

        sigDict[method] = [np.mean(dataTrueMethod['Value'] - thr > 1.0E-6)]

    return sigDict


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

def run_tests(datagen_func, decomp_func, decompLabels, nTest=100, haveShuffle=False):
    rezDict = {k: [] for k in decompLabels}
    for iTest in range(nTest):
        x, y, z = datagen_func()
        zEff = z if not haveShuffle else shuffle(z)

        rez = decomp_func(x, y, zEff)

        for k in rezDict.keys():
            rezDict[k] += [rez[k]]

    rezDF = pd.DataFrame()

    for iLabel, label in enumerate(decompLabels):
        rezTmp = pd.DataFrame({'Method': [label] * nTest, 'Value': rezDict[label]})
        rezDF = rezDF.append(rezTmp)

    return rezDF


def plot_test_summary(df, dfRand, suptitle=None, haveEff=True, logTrue=True, logEff=False, valThrDict=None):
    nFig = 3 if haveEff else 2
    fig, ax = plt.subplots(ncols=nFig, figsize=(4*nFig, 4), tight_layout=True)
    if suptitle is not None:
        fig.suptitle(suptitle)

    # # Clip data
    # df['Value'] = np.clip(df['Value'], 1.0E-6, None)
    # dfRand['Value'] = np.clip(dfRand['Value'], 1.0E-6, None)

    # Plot 1: True vs Random
    dfMerged = merge_df_from_dict({'True': df, 'Random': dfRand}, columnNames=['Kind'])
    sns.violinplot(ax=ax[0], x="Method", y="Value", hue="Kind", data=dfMerged, scale='width', cut=0)
    if logTrue:
        ax[0].set_yscale('log')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Metric Value')

    # Calculate effect sizes
    dfEffSize = effect_size_by_method(df, dfRand)

    # Plot 2: Effect Sizes
    if haveEff:
        sns.violinplot(ax=ax[1], x="Method", y="Value", data=dfEffSize, scale='width', cut=0)
        if logEff:
            ax[1].set_yscale('log')
        # ax[1].axhline(y='2', color='pink', linestyle='--')
        ax[1].set_xlabel('')
        ax[1].set_ylabel('Effect Size')

    # Calculate fraction significant
    sigDict = fraction_significant(df, dfRand, 0.01, valThrDict=valThrDict)

    # Plot 3: Fraction significant
    idx3 = 2 if haveEff else 1
    sns.barplot(ax=ax[idx3], data=pd.DataFrame(sigDict))
    ax[idx3].set_ylim([0, 1])
    ax[idx3].set_xlabel('')
    ax[idx3].set_ylabel('Fraction Significant')


def _stratify_range(x, eta=1):
    return (1 - np.exp(eta*x)) / (1 - np.exp(eta))


def run_plot_param_effect(datagen_func, decomp_func, decompLabels, nTest=1000, alphaRange=(0, 1)):
    alphaLst = np.linspace(0, 1, nTest)
    alphaLst = _stratify_range(alphaLst, eta=2)
    alphaLst = alphaRange[0] + (alphaRange[1] - alphaRange[0]) * alphaLst

    rezTrueLst = []
    rezRandLst = []
    for iTest in range(nTest):
        # alpha = np.random.uniform(*alphaRange)
        x, y, z = datagen_func(alphaLst[iTest])
        rezTrue = decomp_func(x, y, z)
        rezRand = decomp_func(x, y, shuffle(z))
        rezTrue = [rezTrue[k] for k in decompLabels]
        rezRand = [rezRand[k] for k in decompLabels]

        # alphaLst += [alpha]
        rezTrueLst += [rezTrue]
        rezRandLst += [rezRand]

    rezTrueLst = np.array(rezTrueLst)
    rezRandLst = np.array(rezRandLst)

    nMethods = len(decompLabels)
    fig, ax = plt.subplots(ncols=nMethods, figsize=(4*nMethods, 4), tight_layout=True)
    for iKind, kindLabel in enumerate(decompLabels):
        ax[iKind].set_title(kindLabel)

        ax[iKind].semilogy(alphaLst, rezRandLst[:, iKind], '.', label='Rand')
        ax[iKind].semilogy(alphaLst, rezTrueLst[:, iKind], '.', label='True')
        ax[iKind].set_ylim([1.0E-7, 10])
        ax[iKind].legend()


def run_plot_param_effect_test(datagen_func, decomp_func, decompLabels,
                               nStep=10, nTest=1000, alphaRange=(0, 1), valThrDict=None):
    # alphaLst = np.linspace(*alphaRange, nStep)
    alphaLst = np.linspace(0, 1, nStep)
    alphaLst = _stratify_range(alphaLst, eta=3)
    alphaLst = alphaRange[0] + (alphaRange[1] - alphaRange[0]) * alphaLst

    dfTrueDict = {}
    dfRandDict = {}
    dfEffDict = {}
    for alpha in alphaLst:
        gen_data_eff = lambda: datagen_func(alpha)

        rezDF   = run_tests(gen_data_eff, decomp_func, decompLabels, nTest=nTest)
        rezDFsh = run_tests(gen_data_eff, decomp_func, decompLabels, nTest=nTest, haveShuffle=True)
        dfEffSize = effect_size_by_method(rezDF, rezDFsh)

        dfTrueDict[(np.round(alpha, 2), )] = rezDF
        dfRandDict[(np.round(alpha, 2), )] = rezDFsh
        dfEffDict[(np.round(alpha, 2), )] = dfEffSize

    dfRez = merge_df_from_dict(dfEffDict, ['alpha'])

    nMethods = len(decompLabels)
    fig, ax = plt.subplots(nrows=2, ncols=nMethods, figsize=(4*nMethods, 8), tight_layout=True)
    for iMethod, methodName in enumerate(decompLabels):
        # Compute plot effect sizes
        dfRezMethod = dfRez[dfRez['Method'] == methodName]
        sns.violinplot(ax=ax[0, iMethod], x="alpha", y="Value", data=dfRezMethod, scale='width')
        ax[0, iMethod].set_xticklabels(ax[0, iMethod].get_xticklabels(), rotation = 90)
        ax[0, iMethod].set_xlabel('')
        ax[0, iMethod].set_title(methodName)

        # # Compute plot thresholded effect sizes
        # sigDict = {}
        # for alpha, dfTrue in dfTrueDict.items():
        #     dfRand = dfRandDict[alpha]
        #     dfTrueMethod = dfTrue[dfTrue['Method'] == methodName]
        #     dfRandMethod = dfRand[dfRand['Method'] == methodName]
        #
        #     thr = np.quantile(dfRandMethod['Value'], 0.99)
        #     sigDict[alpha[0]] = [np.mean(dfTrueMethod['Value'] > thr)]

        # Compute plot thresholded effect sizes
        sigDict = {}
        for alpha, dfTrue in dfTrueDict.items():
            dfRand = dfRandDict[alpha]
            sigDict[alpha[0]] = fraction_significant(dfTrue, dfRand, 0.01, valThrDict=valThrDict)[methodName]

        valDF = pd.DataFrame(sigDict)
        sns.barplot(ax=ax[1, iMethod], data=valDF)
        ax[1, iMethod].set_xticklabels(ax[1, iMethod].get_xticklabels(), rotation=90)
        ax[1, iMethod].set_ylim(0, 1.05)
        ax[0, iMethod].set_xlabel('$\sigma$')

    ax[0, 0].set_ylabel('Effect Size')
    ax[1, 0].set_ylabel('Fraction Significant')


def run_plot_param_effect_test_single(datagen_func, decomp_func, decompLabels, alpha, nTest=1000, valThrDict=None):
    gen_data_eff = lambda: datagen_func(alpha)

    rezDF   = run_tests(gen_data_eff, decomp_func, decompLabels, nTest=nTest)
    rezDFsh = run_tests(gen_data_eff, decomp_func, decompLabels, nTest=nTest, haveShuffle=True)
    dfEffSize = effect_size_by_method(rezDF, rezDFsh)

    print(fraction_significant(rezDF, rezDFsh, 0.01, valThrDict=valThrDict))

    fig, ax = plt.subplots(ncols=3, figsize=(12,4))
    sns.violinplot(ax=ax[0], data=rezDF, x="Method", y="Value", scale='width', cut=0)
    sns.violinplot(ax=ax[1], data=rezDFsh, x="Method", y="Value", scale='width', cut=0)
    sns.violinplot(ax=ax[2], data=dfEffSize, x="Method", y="Value", scale='width', cut=0)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    # ax[0].set_ylim([1.0E-7, 10])
    # ax[1].set_ylim([1.0E-7, 10])
    plt.show()


def run_plot_data_effect_test(datagen_func, decomp_func, decompLabels, nStep=10, nTest=1000, valThrDict=None):
    nSampleLst = (10 ** np.linspace(2, 5, nStep)).astype(int)

    dfTrueDict = {}
    dfRandDict = {}
    dfEffDict = {}
    for nSample in nSampleLst:
        gen_data_eff = lambda: datagen_func(nSample)

        rezDF   = run_tests(gen_data_eff, decomp_func, decompLabels, nTest=nTest)
        rezDFsh = run_tests(gen_data_eff, decomp_func, decompLabels, nTest=nTest, haveShuffle=True)
        dfEffSize = effect_size_by_method(rezDF, rezDFsh)

        dfTrueDict[(nSample, )] = rezDF
        dfRandDict[(nSample, )] = rezDFsh
        dfEffDict[(nSample, )] = dfEffSize


    dfRez = merge_df_from_dict(dfEffDict, ['nSample'])

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
        sigDict = {}
        for nSampleTuple, dfTrue in dfTrueDict.items():
            dfRand = dfRandDict[nSampleTuple]
            sigDict[nSampleTuple[0]] = fraction_significant(dfTrue, dfRand, 0.01, valThrDict=valThrDict)[methodName]

            # dfTrueMethod = dfTrue[dfTrue['Method'] == methodName]
            # dfRandMethod = dfRand[dfRand['Method'] == methodName]
            #
            # thr = np.quantile(dfRandMethod['Value'], 0.99)
            # sigDict[nSampleTuple[0]] = [np.mean(dfTrueMethod['Value'] > thr)]

        valDF = pd.DataFrame(sigDict)
        sns.barplot(ax=ax[1, iMethod], data=valDF)
        ax[1, iMethod].set_xticklabels(ax[1, iMethod].get_xticklabels(), rotation=90)
        ax[1, iMethod].set_ylim(0, 1.05)
        ax[0, iMethod].set_xlabel('$\sigma$')

    ax[0, 0].set_ylabel('Effect Size')
    ax[1, 0].set_ylabel('Fraction Significant')


##############################
# Max-Synergy-Parameter Search
##############################

def _in_limits(x, varLim):
    return np.all(x >= varLim[0]) and np.all(x <= varLim[1]) and (x[1] <= x[0])


def run_sgd_3D(datagen_func, decomp_func, labelTrg, varLimits=(0, 1), nSample=1000, maxStep=100, sgdSig=0.2):
    def _est(p):
        x, y, z = datagen_func(nSample, *p)
        rez = decomp_func(x, y, z)
        return rez[labelTrg]

    p = np.random.uniform(varLimits[0], varLimits[1], 3)
    while not _in_limits(p, varLimits):
        p = np.random.uniform(varLimits[0], varLimits[1], 3)

    v = _est(p)

    print('+', p, ':', v)
    for i in range(maxStep):
        pNew = p + np.random.normal(0, sgdSig, 3)
        while not _in_limits(pNew, varLimits):
            pNew = p + np.random.normal(0, sgdSig, 3)

        vNew = _est(pNew)
        if vNew > v:
            print('+', pNew, ':', vNew)
            p=pNew
            v=vNew
        else:
            pass
            # if np.random.uniform(0,1) < 0.1:
            #     # print('-', pNew, ':', vNew)
            #     p = pNew
            #     # v = vNew


def run_gridsearch_3D(datagen_func, decomp_func, labelTrg, varLimits=(0, 1), nSample=1000, nStep=10):
    rngLst = np.linspace(*varLimits, nStep)

    rezLst = []
    for p1 in rngLst:
        for p2 in rngLst:
            if p2 <= p1:
                for p3 in rngLst:
                    x, y, z = datagen_func(nSample, p1, p2, p3)
                    rez = decomp_func(x, y, z)[labelTrg]
                    rezLst += [[p1, p2, p3, rez]]

    rezLst = np.array(rezLst)
    rezLst = rezLst[np.argsort(rezLst[:, -1])]  # Sort by result
    print(rezLst[-10:])


def run_plot_1D_scan(datagen_func, decomp_func, labelA, labelB, varLimits=(0, 1),
                     nSample=1000, nStep=100, nTest=20, nTestResample=1000):
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
            rez = decomp_func(x, y, z)

            aTmp += [rez[labelA]]
            bTmp += [rez[labelB]]

        rezAMuLst += [np.mean(aTmp)]
        rezBMuLst += [np.mean(bTmp)]
        rezAStdLst += [np.std(aTmp)]
        rezBStdLst += [np.std(bTmp)]

    # Find and report maximal synergy point
    iAlphaMax = np.argmax(rezBMuLst)
    alphaMax = alphaLst[iAlphaMax]

    # Find distribution at maximal synergy point
    synDistr = []
    for iTest in range(nTestResample):
        x, y, z = datagen_func(nSample, alphaMax, alphaMax, 0)
        rez = decomp_func(x, y, z)
        synDistr += [rez[labelB]]

    synThrMax = np.quantile(synDistr, 0.99)
    print('alpha', alphaMax, 'thr', synThrMax)

    plt.figure()
    plt.errorbar(alphaLst, rezAMuLst, rezAStdLst, label=labelA)
    plt.errorbar(alphaLst, rezBMuLst, rezBStdLst, label=labelB)
    plt.axhline(synThrMax, color='red', alpha=0.3, linestyle='--')
    plt.axvline(alphaMax, color='red', alpha=0.3, linestyle='--')

    plt.xlabel('Parameter values')
    plt.ylabel('Function values')
    # plt.title('Synergy-Redundancy relationship for noisy redundant model')
    plt.legend()

    return alphaMax, synThrMax


##############################
# Synergy-Redundancy Relation
##############################

def run_plot_scatter_explore(datagen_func, decomp_func, labelA, labelB, nVars, varLimits=(0, 1), nSample=1000, nTestDim=10):
    rezALst = []
    rezBLst = []

    sTmp = 0
    sVars = 0

    x1 = np.linspace(*varLimits, nTestDim)
    prodIt = itertools.product(*[x1]*nVars)

    for vars in prodIt:
        # vars = np.random.uniform(*varLimits, nVars)
        x, y, z = datagen_func(nSample, *vars)
        rez = decomp_func(x, y, z)

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


def run_plot_scatter_exact(datagen_func, decomp_func, labelA, labelB, vars, nSample=1000, nTest=1000):
    rezALst = []
    rezBLst = []

    for iTest in range(nTest):
        x, y, z = datagen_func(nSample, *vars)
        rez = decomp_func(x, y, z)

        rezALst += [rez[labelA]]
        rezBLst += [rez[labelB]]

    plt.figure()
    plt.plot(rezALst, rezBLst, '.')
    plt.xlabel(labelA)
    plt.ylabel(labelB)
    # plt.title('Synergy-Redundancy relationship for noisy redundant model')
    plt.show()

#
# def run_plot_2D_scan(datagen_func, decomp_func, labelA, labelB, varLimits=(0, 1), nSample=1000, nStep=10, nTest=20):
#     rezAMat = np.zeros((nStep, nStep))
#     rezBMat = np.zeros((nStep, nStep))
#
#     alphaLst = np.linspace(*varLimits, nStep)
#
#     for iAlpha, alphaX in enumerate(alphaLst):
#         for jAlpha, alphaY in enumerate(alphaLst):
#
#             tmpA = []
#             tmpB = []
#             for iTest in range(nTest):
#                 x, y, z = datagen_func(nSample, alphaX, alphaY, 0)
#                 rez = decomp_func(x, y, z)
#
#                 tmpA += [rez[labelA]]
#                 tmpB += [rez[labelB]]
#
#             rezAMat[iAlpha][jAlpha] = np.mean(tmpA)
#             rezBMat[iAlpha][jAlpha] = np.mean(tmpB)
#
#     # Find and report maximal synergy point
#     iAlphaMax, jAlphaMax = np.unravel_index(np.argmax(rezBMat), rezBMat.shape)
#     print('maxSyn', np.max(rezBMat), 'red', rezAMat[iAlphaMax][jAlphaMax], 'alpha', alphaLst[iAlphaMax], alphaLst[jAlphaMax])
#
#     # Find distribution at maximal synergy point
#     rezDict = {labelA: [], labelB: []}
#     for iTest in range(1000):
#         x, y, z = datagen_func(nSample, alphaLst[iAlphaMax], alphaLst[jAlphaMax], 0)
#         rez = decomp_func(x, y, z)
#         rezDict[labelA] += [rez[labelA]]
#         rezDict[labelB] += [rez[labelB]]
#     dfMax = pd.DataFrame(rezDict)
#
#     print('1% quantile max synergy', np.quantile(rezDict[labelB], 0.99))
#
#     fig, ax = plt.subplots(ncols=3, figsize=(12,4), tight_layout=True)
#     imshow(fig, ax[0], rezAMat, title=labelA, haveColorBar=True)
#     imshow(fig, ax[1], rezBMat, title=labelB, haveColorBar=True)
#     sns.violinplot(ax=ax[2], data=dfMax)
#     plt.show()






