import numpy as np

from mesostat.utils.arrays import numpy_merge_dimensions
from mesostat.utils.signals.resample import bin_data
from mesostat.utils.signals.filter import drop_PCA
import mesostat.utils.iterators.matrix as matiter


# Loop over all possible targets excluding sources
def _pid_all(mc, nChannel, dimOrdTrg, metricName, settingsEstimator, dim=3, dropChannels=None, shuffle=False):

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

    return channelIdxCombinations, rez
    # 3D result: (triplets, pidType, statType);;  statType == {'p', 'effSize', 'muTrue', 'muRand'}
    # return np.array([percentile_twosided(rezThis, fRand, settings={"haveEffectSize": True, "haveMeans": True})[1:].T for rezThis in rez])


# Loop over targets in target list
def _pid_specific(mc, labelsAll, labelsSrc, labelsTrg, dimOrdTrg, metricName, settingsEstimator, dim=3, shuffle=False):
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

    rezLst = []
    labelsRezLst = []
    for iSrcPair, iSrcTuple in enumerate(sourceIdxCombinations):
        for iTrg, labelTrg in enumerate(labelsTrg):
            labelsRezLst += [[labelsAll[iSrc] for iSrc in iSrcTuple] + [labelTrg]]
            rezLst += [rez[iSrcPair][iTrg]]

    return labelsRezLst, np.array(rezLst)


def _pid_estimator_settings(metric, dim=3):
    if metric == 'BivariatePID':
        settingsEstimator = {'pid_estimator': 'TartuPID'}
    elif metric == 'MultivariatePID':
        settingsEstimator = {'pid_estimator': 'SxPID'}
    else:
        raise ValueError('Unexpected metric', metric)

    assert dim in [3, 4]  # Only 3D and 4D estimators currently supported
    settingsEstimator['lags_pid'] = [0] * (dim - 1)

    return settingsEstimator


def _pid_prepare_data_avg(dataLst, nDropPCA=None, nBin=4):
    # Concatenate all sessions
    dataRSP = np.concatenate(dataLst, axis=0)  # Concatenate trials and sessions
    dataRP = np.mean(dataRSP, axis=1)  # Average out time

    if nDropPCA is not None:
        dataRP = drop_PCA(dataRP, nDropPCA)

    dataBin = bin_data(dataRP, nBin, axis=1)  # Bin data separately for each channel
    return dataBin, dataBin.shape[1]


def _pid_prepare_data_time(dataLst, nDropPCA=None, nBin=4):
    # Concatenate all sessions
    dataRSP = np.concatenate(dataLst, axis=0)  # Concatenate trials and sessions
    dataSP = numpy_merge_dimensions(dataRSP, 0, 2)

    if nDropPCA is not None:
        dataSP = drop_PCA(dataSP, nDropPCA)

    print(dataSP.shape)

    dataBin2D = bin_data(dataSP, nBin, axis=1)  # Bin data separately for each channel
    dataBin3D = dataBin2D.reshape(dataRSP.shape)

    return dataBin3D, dataBin3D.shape[2]


# Calculate 3D PID with two sources and 1 target. If more than one target is provided,
def pid(dataLst, mc, labelsAll=None, labelsSrc=None, labelsTrg=None, dropChannels=None, nBin=4, nDropPCA=None,
        verbose=True, permuteTarget=False, metric='BivariatePID', dim=3, timeSweep=False):
    '''
    :param dataLst:     List of data over sessions, each dataset is of shape 'rsp'
    :param mc:          MetricCalculator
    :param labelsAll:   List of labels of all the channels. Needed to identify indices of source and target channels
    :param labelsSrc:   List of labels of the source channels. Must be exactly two
    :param labelsTrg:   List of labels of the target channels. Each target is analysed separately
    :param nBin:        Number of bins to use to bin the data
    :param nDropPCA:    Number of primary principal components to drop prior to analysis
    :param verbose:     Verbosity of output
    :param permuteTarget:  Whether to permute data by target, resulting in shuffled estimator
    :param metric:      Type of PID estimator
    :param dim:         Dimension of to consider (3 or 4)
    :return:            Dataframe containing PID results for each combination of sources and targets
    '''

    metricName = metric + str(dim) + 'D'

    # Prepare data
    if timeSweep:
        dataBin, nChannel = _pid_prepare_data_time(dataLst, nDropPCA=nDropPCA, nBin=nBin)

        if verbose:
            print("Time-sweep analysis with shape:", dataBin.shape)

        dimOrdSrc = 'rsp'
        dimOrdTrg = 's'
    else:
        dataBin, nChannel = _pid_prepare_data_avg(dataLst, nDropPCA=nDropPCA, nBin=nBin)

        if verbose:
            print("Single-point analysis with shape:", dataBin.shape)

        dimOrdSrc = 'rp'
        dimOrdTrg = ''

    # Set data
    mc.set_data(dataBin, dimOrdSrc)

    # Estimator settings
    settingsEstimator = _pid_estimator_settings(metric, dim=dim)

    if verbose:
        print("Computing PID...")

    havePIDAll = (labelsSrc is None) and (labelsTrg is None) and (labelsAll is None)
    havePIDSpecific = (labelsSrc is not None) and (labelsTrg is not None) and (labelsAll is not None) and (dropChannels is None)

    if havePIDAll:
        return _pid_all(mc, nChannel, dimOrdTrg, metricName, settingsEstimator,
                        dim=dim, dropChannels=dropChannels, shuffle=permuteTarget)
    elif havePIDSpecific:
        return _pid_specific(mc, labelsAll, labelsSrc, labelsTrg, dimOrdTrg, metricName, settingsEstimator,
                             dim=dim, shuffle=permuteTarget)
    else:
        raise ValueError('Must provide both source and target indices or neither')
