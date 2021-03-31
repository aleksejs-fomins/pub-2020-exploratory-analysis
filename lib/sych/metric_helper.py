import numpy as np
from IPython.display import display
from ipywidgets import IntProgress

from mesostat.utils.arrays import numpy_nonelist_to_array

def dimord_to_labels(dimOrd):
    dimOrdDict = {
        'p': 'channels',
        'r': 'trials',
        's': 'timesteps'
    }

    return tuple([dimOrdDict[d] for d in dimOrd])


def metric_by_session(dataDB, mc, ds, mousename, metricName, dimOrdTrg,
                      dataName=None, cropTime=None, minTrials=1,
                      timeWindow=None, timeAvg=False, metricSettings=None, sweepSettings=None, **kwargs):

    # Drop all arguments that were not specified
    # In some use cases non-specified arguments are not implemented
    # kwargs = {'trialType': trialType, 'zscoreDim': zscoreDim, 'datatype': datatype, 'performance': performance}
    if cropTime is not None:
        kwargs['cropTime'] = cropTime[1]

    kwargs = {k : v for k, v in kwargs.items() if v is not None}

    dataLst = dataDB.get_neuro_data({'mousename': mousename}, **kwargs)

    rez = []
    progBar = IntProgress(min=0, max=len(dataLst), description=mousename)
    display(progBar)  # display the bar
    for dataSession in dataLst:
        if dataSession.shape[0] < minTrials:
            print('Warning: skipping session with too few trials', dataSession.shape[0])
            rez += [None]
        else:
            # Calculate stuff
            # IMPORTANT: DO NOT DO ZScore on cropped data at this point. ZScore is done on whole data during extraction
            if not timeAvg:
                mc.set_data(dataSession, 'rsp', timeWindow=timeWindow)
            else:
                if timeWindow is not None:
                    raise ValueError('Time-averaging and timeWindow are incompatible')

                mc.set_data(np.nanmean(dataSession, axis=1), 'rp')

            rez += [mc.metric3D(metricName, dimOrdTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)]
        progBar.value += 1

    if cropTime is not None:
        kwargs['cropTime'] = cropTime[0]
    attrsDict = {
        **kwargs, **{
        'mousename': mousename,
        'metric': metricName,
        'target_dim': str(('sessions',) + dimord_to_labels(dimOrdTrg))
    }}

    if dataName is None:
        dataName = metricName

    ds.save_data(dataName, numpy_nonelist_to_array(rez), attrsDict)


def metric_by_selector(dataDB, mc, ds, selector, metricName, dimOrdTrg,
                      dataName=None, cropTime=None, minTrials=1,
                       timeWindow=None, timeAvg=False, metricSettings=None, sweepSettings=None, **kwargs):

    # Drop all arguments that were not specified
    # In some use cases non-specified arguments are not implemented
    #kwargs = {'trialType': trialType, 'zscoreDim': zscoreDim, 'datatype': datatype, 'performance': performance}
    if cropTime is not None:
        kwargs['cropTime'] = cropTime[1]

    kwargs = {k : v for k, v in kwargs.items() if v is not None}

    dataLst = dataDB.get_neuro_data(selector, **kwargs)
    dataRSP = np.concatenate(dataLst, axis=0)

    print(dataRSP.shape)

    if len(dataLst) == 0:
        print('for', selector, kwargs, 'there are no sessions, skipping')
    elif len(dataRSP) < minTrials:
        print('for', selector, kwargs, 'too few trials', len(dataRSP), ', skipping')
    else:
        # Calculate stuff
        # IMPORTANT: DO NOT DO ZScore on cropped data at this point. ZScore is done on whole data during extraction
        if not timeAvg:
            mc.set_data(dataRSP, 'rsp', timeWindow=timeWindow)
        else:
            if timeWindow is not None:
                raise ValueError('Time-averaging and timeWindow are incompatible')

            mc.set_data(np.nanmean(dataRSP, axis=1), 'rp')

        rez = mc.metric3D(metricName, dimOrdTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)

        if cropTime is not None:
            kwargs['cropTime'] = cropTime[0]
        attrsDict = {
            **kwargs, **selector, **{
            'metric': metricName,
            'target_dim': str(dimord_to_labels(dimOrdTrg))
        }}

        if dataName is None:
            dataName = metricName

        ds.save_data(dataName, np.array(rez), attrsDict)
