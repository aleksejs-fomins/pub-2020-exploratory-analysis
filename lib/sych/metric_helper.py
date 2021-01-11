import numpy as np
from IPython.display import display
from ipywidgets import IntProgress


def dimord_to_labels(dimOrd):
    dimOrdDict = {
        'p': 'channels',
        'r': 'trials',
        's': 'timesteps'
    }

    return tuple([dimOrdDict[d] for d in dimOrd])


def metric_by_session(dataDB, mc, ds, mousename, metricName, dimOrdTrg,
                      dataName=None, cropTime=None, datatype=None, trialType=None, performance=None,
                      zscoreDim=None, timeWindow=None, metricSettings=None, sweepSettings=None):

    # Drop all arguments that were not specified
    # In some use cases non-specified arguments are not implemented
    kwargs = {'trialType': trialType, 'zscoreDim': zscoreDim, 'cropTime': cropTime, 'datatype': datatype, 'performance': performance}
    kwargs = {k : v for k, v in kwargs.items() if v is not None}

    dataLst = dataDB.get_neuro_data({'mousename': mousename}, **kwargs)

    rez = []
    progBar = IntProgress(min=0, max=len(dataLst), description=mousename)
    display(progBar)  # display the bar
    for dataSession in dataLst:
        # Calculate stuff
        # IMPORTANT: DO NOT DO ZScore on cropped data at this point. ZScore is done on whole data during extraction
        mc.set_data(dataSession, 'rsp', timeWindow=timeWindow)
        rez += [mc.metric3D(metricName, dimOrdTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)]
        progBar.value += 1

    attrsDict = {
        **kwargs, **{
        'mousename': mousename,
        'metric': metricName,
        'target_dim': str(('sessions',) + dimord_to_labels(dimOrdTrg))
    }}

    if dataName is None:
        dataName = metricName

    ds.save_data(dataName, np.array(rez), attrsDict)


def metric_by_selector(dataDB, mc, ds, selector, metricName, dimOrdTrg,
                      dataName=None, cropTime=None, datatype=None, trialType=None, performance=None,
                      zscoreDim=None, timeWindow=None, metricSettings=None, sweepSettings=None):

    # Drop all arguments that were not specified
    # In some use cases non-specified arguments are not implemented
    kwargs = {'trialType': trialType, 'zscoreDim': zscoreDim, 'cropTime': cropTime, 'datatype': datatype, 'performance': performance}
    kwargs = {k : v for k, v in kwargs.items() if v is not None}

    dataLst = dataDB.get_neuro_data(selector, **kwargs)

    dataRSP = np.concatenate(dataLst, axis=0)

    if (len(dataLst) == 0) or len(dataRSP) == 0:
        print('for', selector, trialType, performance, 'there are no trials, skipping')
    else:
        # Calculate stuff
        # IMPORTANT: DO NOT DO ZScore on cropped data at this point. ZScore is done on whole data during extraction
        mc.set_data(dataRSP, 'rsp', timeWindow=timeWindow)
        rez = mc.metric3D(metricName, dimOrdTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)

        attrsDict = {
            **kwargs, **selector, **{
            'metric': metricName,
            'target_dim': str(dimord_to_labels(dimOrdTrg))
        }}

        if dataName is None:
            dataName = metricName

        ds.save_data(dataName, np.array(rez), attrsDict)
