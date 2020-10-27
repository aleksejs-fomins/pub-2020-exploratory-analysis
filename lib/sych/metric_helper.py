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
                      dataName=None, cropTime=None, trialType=None,
                      zscoreDim=None, timeWindow=None, metricSettings=None, sweepSettings=None):

    dataLst = dataDB.get_neuro_data({'mousename': mousename}, trialType=trialType, cropTime=cropTime)

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