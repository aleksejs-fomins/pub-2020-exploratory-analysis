import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import combine_pvalues

from mesostat.stat.connectomics import offdiag_idx
from mesostat.metric.corr import corr_2D
from mesostat.utils.signals import polyfit_transform
from mesostat.visualization.mpl_matrix import imshow
from mesostat.utils.pandas_helper import outer_product_df


def _sweep_iter(dataDB, mousename, intervDict=None, trialTypes=None):
    paramDict = {'session' : dataDB.get_sessions(mousename)}
    if trialTypes is not None:
        paramDict['trialType'] = trialTypes
    if intervDict is not None:
        paramDict['intervName'] = list(intervDict.keys())

    paramDF = outer_product_df(paramDict)

    for idx, row in paramDF.iterrows():
        kwargs = {}
        if 'trialType' in row.keys():
            kwargs['trialType'] = row['trialType']
        if 'intervName' in row.keys():
            kwargs['cropTime'] = intervDict[row['intervName']]

        yield row, kwargs


def correlation_by_session(dataDB, datatype, intervDict=None, trialTypes=None):
    for mousename in sorted(dataDB.mice):
        for row, kwargs in _sweep_iter(dataDB, mousename, intervDict=intervDict, trialTypes=trialTypes):
            dataRSP = dataDB.get_neuro_data({'session': row['session']}, datatype=datatype, **kwargs)[0]

            dataRP = np.mean(dataRSP, axis=1)
            nTrials, nChannel = dataRP.shape
            results = np.zeros((nChannel, nChannel))

            if nTrials < 50:
                print('Too few trials =', nTrials, ' for', row.values, ': skipping')
            else:
                corr = corr_2D(dataRP.T)

                fig, ax = plt.subplots()
                imshow(fig, ax, corr, cmap='jet', haveColorBar=True, limits=[-1,1])

                plt.savefig('pics/corr_'+'_'.join(list(row.values))+'.png')
                plt.close()


def linear_fit_correlation(dataDB, datatype, intervDict=None, trialTypes=None):
    for mousename in sorted(dataDB.mice):
        for row, kwargs in _sweep_iter(dataDB, mousename, intervDict=intervDict, trialTypes=trialTypes):
            dataRSP = dataDB.get_neuro_data({'session': row['session']}, datatype=datatype, **kwargs)[0]

            dataRP = np.mean(dataRSP, axis=1)
            nTrials, nChannel = dataRP.shape
            results = np.zeros((nChannel, nChannel))

            if nTrials < 50:
                print('Too few trials =', nTrials, ' for', row.values, ': skipping')
            else:
                for iChannel in range(nChannel):
                    dataA = dataRP[:, iChannel]
                    dataOther = np.delete(dataRP, iChannel, axis=1)
                    dataSub = dataOther.copy()

                    # Part 1: Fit-subtract A from all other channels
                    for iOther in range(nChannel - 1):
                        dataSub[:, iOther] -= polyfit_transform(dataA, dataSub[:, iOther])

                    # corrOther = corr_2D(dataSub.T, settings={'havePVal': True})[..., 0]
                    # corrMean = np.array([np.mean(np.delete(corrOther[i], i)) for i in range(nChannel - 1)])
                    # results[iChannel] = np.insert(corrMean, iChannel, 1)

                    # Part 2: Compute correlation and its p-value
                    pValsOther = corr_2D(dataSub.T, settings={'havePVal': True})[..., 1]

                    # Part 3: Combine pvalues over all pairs
                    pValsCombined = np.array([combine_pvalues(np.delete(pValsOther[i], i))[1] for i in range(nChannel - 1)])

                    results[iChannel] = np.insert(pValsCombined, iChannel, 0)

                results = -np.log10(results, where=offdiag_idx(nChannel))

                plt.figure()
                plt.imshow(results)
                plt.title('_'.join(list(row.values)))
                plt.colorbar()

                plt.savefig('pics/corr_subtracted_' + '_'.join(list(row.values)) + '.png')
                plt.close()
