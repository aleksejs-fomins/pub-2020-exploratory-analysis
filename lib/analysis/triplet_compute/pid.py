from lib.analysis.triplet_compute.metric_common import metric_all, metric_specific, preprocess_data


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


# Calculate 3D PID with two sources and 1 target. If more than one target is provided,
def pid(dataLst, mc, labelsAll=None, labelsSrc=None, labelsTrg=None, dropChannels=None, nBin=4, nDropPCA=None,
        verbose=True, permuteTarget=False, metric='BivariatePID', dim=3, timeSweep=False, labelsAsText=False):
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
        dataBin = preprocess_data(dataLst, nDropPCA=nDropPCA, nBin=nBin, timeAvg=False)

        if verbose:
            print("Time-sweep analysis with shape:", dataBin.shape)

        dimOrdSrc = 'rsp'
        dimOrdTrg = 's'
    else:
        dataBin = preprocess_data(dataLst, nDropPCA=nDropPCA, nBin=nBin, timeAvg=True)

        if verbose:
            print("Single-point analysis with shape:", dataBin.shape)

        dimOrdSrc = 'rp'
        dimOrdTrg = ''

    # Estimator settings
    settingsEstimator = _pid_estimator_settings(metric, dim=dim)

    if verbose:
        print("Computing PID...")

    havePIDAll = (labelsSrc is None) and (labelsTrg is None) and (labelsAll is None)
    havePIDSpecific = (labelsSrc is not None) and (labelsTrg is not None) and (labelsAll is not None) and (dropChannels is None)

    if havePIDAll:
        return metric_all(mc, dataBin, dimOrdSrc, dimOrdTrg, metricName, settingsEstimator,
                        dim=dim, dropChannels=dropChannels, shuffle=permuteTarget)
    elif havePIDSpecific:
        return metric_specific(mc, dataBin, dimOrdSrc, dimOrdTrg, labelsAll, labelsSrc, labelsTrg, metricName,
                               settingsEstimator, dim=dim, shuffle=permuteTarget, labelsAsText=labelsAsText)
    else:
        raise ValueError('Must provide both source and target indices or neither')
