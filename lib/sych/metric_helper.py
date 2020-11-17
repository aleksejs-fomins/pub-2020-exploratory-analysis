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
    if datatype is None:
        dataLst = dataDB.get_neuro_data({'mousename': mousename}, trialType=trialType, cropTime=cropTime, performance=performance)
    else:
        dataLst = dataDB.get_neuro_data({'mousename': mousename}, trialType=trialType, datatype=datatype, performance=performance)

    rez = []
    progBar = IntProgress(min=0, max=len(dataLst), description=mousename)
    display(progBar)  # display the bar
    for dataSession in dataLst:
        # Calculate stuff
        mc.set_data(dataSession, 'rsp', timeWindow=timeWindow, zscoreDim=zscoreDim)
        rez += [mc.metric3D(metricName, dimOrdTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)]
        progBar.value += 1

    attrsDict = {
        'mousename': mousename,
        'metric': metricName,
        'target_dim': str(('sessions',) + dimord_to_labels(dimOrdTrg))
    }

    if dataName is None:
        dataName = metricName

    ds.save_data(dataName, np.array(rez), attrsDict)


def metric_by_selector(dataDB, mc, ds, selector, metricName, dimOrdTrg,
                      dataName=None, cropTime=None, datatype=None, trialType=None, performance=None,
                      zscoreDim=None, timeWindow=None, metricSettings=None, sweepSettings=None):

    if datatype is None:
        dataLst = dataDB.get_neuro_data(selector, trialType=trialType, cropTime=cropTime, performance=performance)
    else:
        dataLst = dataDB.get_neuro_data(selector, trialType=trialType, datatype=datatype, performance=performance)

    dataRSP = np.concatenate(dataLst, axis=0)

    if (len(dataLst) == 0) or len(dataRSP) == 0:
        print('for', selector, trialType, performance, 'there are no trials, skipping')
    else:
        # Calculate stuff
        mc.set_data(dataRSP, 'rsp', timeWindow=timeWindow, zscoreDim=zscoreDim)
        rez = mc.metric3D(metricName, dimOrdTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)

        attrsDict = {
            **selector, **{
            'metric': metricName,
            'target_dim': str(dimord_to_labels(dimOrdTrg))
        }}

        if dataName is None:
            dataName = metricName

        ds.save_data(dataName, np.array(rez), attrsDict)
