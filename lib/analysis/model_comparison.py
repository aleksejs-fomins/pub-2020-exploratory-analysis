import numpy as np
import matplotlib.pyplot as plt

from mesostat.utils.matrix import tril_1D
from mesostat.metric.corr import corr_2D
from mesostat.visualization.mpl_violin import violins_labeled

from lib.common.param_sweep import DataParameterSweep, pd_row_to_kwargs


def corr_evaluation(dataDB, mc, estimator, exclQueryLst=None, minTrials=50, **kwargs):
    resultsDict = {'corr' : {}, 'pval' : {}}

    dps = DataParameterSweep(dataDB, exclQueryLst, mousename='auto', **kwargs)

    for idx, row in dps.sweepDF.iterrows():
        kwargsThis = pd_row_to_kwargs(row, parseNone=True, dropKeys=['mousename'])

        results = []
        for session in dataDB.get_sessions(row['mousename']):
            dataRSP = dataDB.get_neuro_data({'session' : session}, **kwargsThis)[0]

            nTrials, nTime, nChannel = dataRSP.shape

            if nTrials < minTrials:
                print('Too few trials =', nTrials, ' for', session, kwargs, ': skipping')
            else:
                mc.set_data(dataRSP, 'rsp')
                metricSettings={'timeAvg' : True, 'havePVal' : True, 'estimator' : estimator}
                rez2D = mc.metric3D('corr', '', metricSettings=metricSettings)
                rez1D = np.array([tril_1D(rez2D[..., 0]), tril_1D(rez2D[..., 1])])
                results += [rez1D]

        if results != []:
            dictKey = '_'.join([row['mousename'], *kwargs.values()])
            results = np.hstack(results)
            resultsDict['corr'][dictKey] = results[0]
            resultsDict['pval'][dictKey] = results[1]
    return resultsDict


def plot_fc_explore(dictMag, dictPval, metricName, withBonferroni=False):
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    keys = list(dictMag.keys())
    valsC = [mag for mag in dictMag.values()]
    valsP = list(dictPval.values())
    valsP = [np.clip(p, 1.0E-20, 1) for p in valsP]

    #     print([c.shape for c in valsC])

    violins_labeled(ax[0], valsC, keys, 'stuff', metricName, joinMeans=False)
    violins_labeled(ax[1], valsP, keys, 'stuff', 'pval', joinMeans=False, haveLog=True)
    ax[0].set_xticklabels(keys, rotation=90)
    ax[1].set_xticklabels(keys, rotation=90)

    if withBonferroni:
        nChannel = 48
        pvalThr = 0.01 / (nChannel * (nChannel - 1) / 2)
        valsPthr = [np.mean(p < pvalThr) for p in valsP]
        ax[1].axhline(y=pvalThr, linestyle='--')
        ax[2].plot(np.arange(len(valsPthr)), valsPthr, '.')
        ax[2].set_xticks(np.arange(len(valsPthr)))
        ax[2].set_xticklabels(keys, rotation=90)

    plt.show()


def empirical_corr_spr(resultsDictCorr, resultsDictSpr):
    ###############################
    # Generate data with random correlation
    ###############################
    pvalCorr = []
    pvalSpr = []

    aVals = np.linspace(0.01, 0.99, 1000)
    for a in aVals:
        x = np.linspace(0, 1, 100)
        x = x[:, None].T
        y = a * x + (1 - a) * np.random.normal(0, 1, 100)

        pvalCorr += [corr_2D(x, y, {'havePVal': True, 'estimator': 'corr'})[0, 1, 1]]
        pvalSpr += [corr_2D(x, y, {'havePVal': True, 'estimator': 'spr'})[0, 1, 1]]

    nLogCorr = -np.log10(pvalCorr)
    nLogSpr = -np.log10(pvalSpr)

    ###############################
    # Linear Fit log(p) vs log(p) for corr vs spr
    ###############################
    a, b = np.polyfit(nLogCorr, nLogSpr, 1)
    t = np.linspace(np.min(nLogCorr), np.max(nLogCorr), 100)
    v = a * t + b

    ###############################
    # Plot results for real data. Use estimated corr vs spr to test if spr > corr
    ###############################
    corrCValues = list(resultsDictCorr['corr'].values())
    sprCValues = list(resultsDictSpr['corr'].values())
    corrPValues = list(resultsDictCorr['pval'].values())
    sprPValues = list(resultsDictSpr['pval'].values())

    fig, ax = plt.subplots(ncols=3, figsize=(12, 4), tight_layout=True)
    ax[0].plot(nLogCorr, nLogSpr, '.')
    ax[0].plot(t, v)
    ax[1].plot(corrCValues[0], sprCValues[0], '.')
    ax[2].plot(-np.log10(corrPValues[0]), -np.log10(sprPValues[0]), '.', label='real')
    ax[2].plot(t, v, label='simulated')
    ax[2].legend()

    ax[0].set_title("Simulated pVal for Corr vs Spr")
    ax[2].set_title("Real Corr vs Spr")
    ax[2].set_title("Real pVal for Corr vs Spr")
    ax[0].set_xlabel('$-\log_{10}(p_{corr})$')
    ax[0].set_ylabel('$-\log_{10}(p_{spr})$')
    ax[1].set_xlabel('Corr')
    ax[1].set_ylabel('Spr')
    ax[2].set_xlabel('$-\log_{10}(p_{corr})$')
    ax[2].set_ylabel('$-\log_{10}(p_{spr})$')

    plt.show()
