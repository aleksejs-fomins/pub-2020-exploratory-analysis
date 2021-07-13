import numpy as np

from mesostat.utils.signals.resample import bin_data
from mesostat.utils.signals.filter import drop_PCA
import mesostat.utils.iterators.matrix as matiter


# Calculate 3D PID with two sources and 1 target. If more than one target is provided,
def pid(dataLst, mc, labelsAll=None, labelsSrc=None, labelsTrg=None, dropChannels=None, nBin=4, nDropPCA=None,
        verbose=True, permuteTarget=False, metric='BivariatePID', dim=3):
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

    assert dim in [3, 4]   # Only 3D and 4D estimators currently supported
    metricName = metric + str(dim) + 'D'

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

    # Estimator settings
    settingsEstimator = {'pid_estimator': 'TartuPID'} if metric=='BivariatePID' else {'pid_estimator': 'SxPID'}
    settingsEstimator['lags_pid'] = [0] * (dim-1)

    mc.set_data(dataBin, 'rp')

    ############################
    # Permutation Testing
    ############################

    # if verbose:
    #     print("Permutation-Testing...")

    # # Since all channels are binned to the same quantiles,
    # # the permutation test is exactly the same for all of them, so we need any three channels as input
    # if nPerm > 0:
    #     settings_test = {'src': [0, 1], 'trg': 2, 'settings_estimator': settings_estimator}
    #     fTest = lambda x: bivariate_pid_3D(x, settings_test)
    #     dataTest = dataBin[:, :3][..., None]  # Add fake 1D sample dimension
    #     fRand = perm_test_resample(fTest, dataTest, nPerm, iterAxis=1)
    # else:
    #     fRand = None

    if verbose:
        print("Computing PID...")

    if (labelsSrc is None) and (labelsTrg is None) and (labelsAll is None):
        ###############################
        # Loop over all possible targets excluding sources
        ###############################

        # Find combinations of all source pairs
        if dim == 3:
            channelIdxCombinations = list(matiter.iter_gn_3D(nChannel))   # Triplets
        else:
            channelIdxCombinations = list(matiter.iter_ggn_4D(nChannel))  # Quadruplets

        if dropChannels is not None:
            dropSet = set(dropChannels)
            channelIdxCombinations = [idx for idx in channelIdxCombinations if len(set(idx).intersection(dropSet)) == 0]

        print('--nComb', len(channelIdxCombinations))

        rez = mc.metric3D(metricName, '',
                          metricSettings={'settings_estimator': settingsEstimator, 'shuffle': permuteTarget},
                          sweepSettings={'channels': channelIdxCombinations})

        return channelIdxCombinations, rez
        # 3D result: (triplets, pidType, statType);;  statType == {'p', 'effSize', 'muTrue', 'muRand'}
        # return np.array([percentile_twosided(rezThis, fRand, settings={"haveEffectSize": True, "haveMeans": True})[1:].T for rezThis in rez])

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

        if dim == 3:
            sourceIdxCombinations = matiter.sample_list(sourceIdxs, matiter.iter_g_2D(nSrc))  # Pairs of sources
        else:
            sourceIdxCombinations = matiter.sample_list(sourceIdxs, matiter.iter_gg_3D(nSrc))  # Triplets of sources
        nCombinations = len(sourceIdxCombinations)

        rez = mc.metric3D(metricName, '',
                          metricSettings={'settings_estimator': settingsEstimator, 'shuffle': permuteTarget},
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
    else:
        raise ValueError('Must provide both source and target indices or neither')
