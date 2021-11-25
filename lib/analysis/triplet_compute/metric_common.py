import numpy as np

import mesostat.utils.iterators.matrix as matiter
from mesostat.utils.arrays import numpy_merge_dimensions
from mesostat.utils.signals.resample import bin_data
from mesostat.utils.signals.filter import drop_PCA


# We want the sweep over source and target channels to be on the first place
# But the method returns the sweep over dataset dimensions on the first place
def _transpose_result(rez, dimOrdTrg):
    dimMove = len(dimOrdTrg)
    if dimMove == 0:
        return rez
    else:
        dimRez = len(rez.shape)
        shapeNew = list(range(dimRez))
        shapeNew = shapeNew[dimMove:dimMove+2] + shapeNew[:dimMove] + shapeNew[dimMove+2:]
        return rez.transpose(shapeNew)


def preprocess_data(dataRSPLst, nDropPCA=None, nBin=None, timeAvg=False):
    dataRSP = np.concatenate(dataRSPLst, axis=0)  # Concatenate trials and sessions

    if timeAvg:
        dataRP = np.mean(dataRSP, axis=1)  # Average out time
    else:
        dataRP = numpy_merge_dimensions(dataRSP, 0, 2)

    if nDropPCA is not None:
        dataRP = drop_PCA(dataRP, nDropPCA)

    if nBin is not None:
        dataRP = bin_data(dataRP, nBin, axis=1)  # Bin data separately for each channel

    return dataRP if timeAvg else dataRP.reshape(dataRSP.shape)


# Loop over all possible targets excluding sources
def metric_all(mc, data, dimOrdSrc, dimOrdTrg, metricName, settingsEstimator, dim=3, dropChannels=None, shuffle=False):
    # Set data
    mc.set_data(data, dimOrdSrc)

    # Find number of channels
    idxOrdChannel = dimOrdSrc.index('p')
    nChannel = data.shape[idxOrdChannel]

    # Find combinations of all source pairs
    if dim == 3:
        channelIdxCombinations = list(matiter.iter_gn_3D(nChannel))  # Triplets
    else:
        channelIdxCombinations = list(matiter.iter_ggn_4D(nChannel))  # Quadruplets

    if dropChannels is not None:
        dropSet = set(dropChannels)
        channelIdxCombinations = [idx for idx in channelIdxCombinations if len(set(idx).intersection(dropSet)) == 0]

    print('--nComb', len(channelIdxCombinations))

    rez = mc.metric3D(metricName, dimOrdTrg,
                      metricSettings={'settings_estimator': settingsEstimator, 'shuffle': shuffle},
                      sweepSettings={'channels': channelIdxCombinations})

    rez = _transpose_result(rez, dimOrdTrg)

    return channelIdxCombinations, rez


# Loop over targets in target list
def metric_specific(mc, data, dimOrdSrc, dimOrdTrg, labelsAll, labelsSrc, labelsTrg, metricName, settingsEstimator,
                    dim=3, shuffle=False, labelsAsText=False):
    # Set data
    mc.set_data(data, dimOrdSrc)

    # Find indices of channel labels
    sourceIdxs = [labelsAll.index(s) for s in labelsSrc]
    targetIdxs = [labelsAll.index(t) for t in labelsTrg]

    # Find combinations of all source pairs
    nSrc = len(sourceIdxs)
    nTrg = len(targetIdxs)

    if dim == 3:
        sourceIdxCombinations = matiter.sample_list(sourceIdxs, matiter.iter_g_2D(nSrc))  # Pairs of sources
    else:
        sourceIdxCombinations = matiter.sample_list(sourceIdxs, matiter.iter_gg_3D(nSrc))  # Triplets of sources
    nCombinations = len(sourceIdxCombinations)

    rez = mc.metric3D(metricName, dimOrdTrg,
                      metricSettings={'settings_estimator': settingsEstimator, 'shuffle': shuffle},
                      sweepSettings={'src': sourceIdxCombinations, 'trg': targetIdxs})

    if (nCombinations == 1) and (nTrg == 1):
        rez = [[rez]]  # Add extra sweep brackets accidentally removed by mesostat
    else:
        rez = _transpose_result(rez, dimOrdTrg)

    rezLst = []
    labelsRezLst = []
    for iSrcPair, srcPair in enumerate(sourceIdxCombinations):
        for iTrg, labelTrg in enumerate(labelsTrg):
            rezLst += [rez[iSrcPair][iTrg]]

            if labelsAsText:
                labelsRezLst += [[labelsAll[iSrc] for iSrc in srcPair] + [labelTrg]]
            else:
                labelsRezLst += [srcPair + [targetIdxs[iTrg]]]

    return labelsRezLst, np.array(rezLst)
