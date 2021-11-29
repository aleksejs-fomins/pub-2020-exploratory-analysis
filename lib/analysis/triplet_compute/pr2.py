from lib.analysis.triplet_compute.metric_common import metric_all, metric_specific, preprocess_data


# Calculate 3D PR2 with two sources and 1 target.
def pr2_multi(dataLst, mc,
              labelsAll=None, labelsSrc=None, labelsTrg=None,
              dropChannels=None, nDropPCA=None, verbose=True, permuteTarget=False,
              timeSweep=False, labelsAsText=False):

    '''
    :param dataLst:     List of data over sessions, each dataset is of shape 'rsp'
    :param mc:          MetricCalculator
    :param labelsAll:   List of labels of all the channels. Needed to identify indices of source and target channels
    :param labelsSrc:   List of labels of the source channels. Must be exactly two
    :param labelsTrg:   List of labels of the target channels. Each target is analysed separately
    :param nDropPCA:    Number of primary principal components to drop prior to analysis
    :param verbose:     Verbosity of output
    :param permuteTarget:  Whether to permute data by target, resulting in shuffled estimator
    :return:            Dataframe containing PID results for each combination of sources and targets
    '''

    # Prepare data
    if timeSweep:
        dataBin = preprocess_data(dataLst, nDropPCA=nDropPCA, nBin=None, timeAvg=False)

        if verbose:
            print("Time-sweep analysis with shape:", dataBin.shape)

        dimOrdSrc = 'rsp'
        dimOrdTrg = 's'
    else:
        dataBin = preprocess_data(dataLst, nDropPCA=nDropPCA, nBin=None, timeAvg=True)

        if verbose:
            print("Single-point analysis with shape:", dataBin.shape)

        dimOrdSrc = 'rp'
        dimOrdTrg = ''

    if verbose:
        print("Computing PR2...")

    havePIDAll = (labelsSrc is None) and (labelsTrg is None) and (labelsAll is None)
    havePIDSpecific = (labelsSrc is not None) and (labelsTrg is not None) and (labelsAll is not None) and (dropChannels is None)

    if havePIDAll:
        return metric_all(mc, dataBin, dimOrdSrc, dimOrdTrg, 'PR2', {},
                          dim=3, dropChannels=dropChannels, shuffle=permuteTarget)
    elif havePIDSpecific:
        return metric_specific(mc, dataBin, dimOrdSrc, dimOrdTrg, labelsAll, labelsSrc, labelsTrg,
                               'PR2', {}, dim=3, shuffle=permuteTarget, labelsAsText=labelsAsText)
    else:
        raise ValueError('Must provide both source and target indices or neither')
