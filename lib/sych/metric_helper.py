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


def metric_by_session(dataDB, mc, ds, mousename, metricName, dimOrdTrg, dataName=None, cropTime=None, zscoreDim=None, metricSettings=None, sweepSettings=None):
    rows = dataDB.get_rows('neuro', {'mousename': mousename})

    progBar = IntProgress(min=0, max=len(rows), description=mousename)
    display(progBar)  # display the bar

    rez = []
    for idx, row in rows.iterrows():
        if cropTime is None:
            data = dataDB.dataNeuronal[idx]
        else:
            data = dataDB.dataNeuronal[idx][:, :cropTime]

        mc.set_data(data, 'rsp', zscoreDim=zscoreDim)
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