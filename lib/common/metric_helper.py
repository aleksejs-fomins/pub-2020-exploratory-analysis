import numpy as np
from IPython.display import display
from ipywidgets import IntProgress

from mesostat.utils.arrays import numpy_nonelist_to_array

from lib.common.verbosity import dimord_to_labels
from lib.common.param_sweep import DataParameterSweep
from lib.common.datawrapper import get_data_list


def metric_by_session(dataDB, mc, ds, mousename, metricName, dimOrdTrg,
                      dataName=None, skipExisting=False, minTrials=1, dropChannels=None,
                      timeWindow=None, timeAvg=False, metricSettings=None, sweepSettings=None, **kwargs):

    # Drop all arguments that were not specified
    # In some use cases non-specified arguments are not implemented
    # kwargs = {'trialType': trialType, 'zscoreDim': zscoreDim, 'datatype': datatype, 'performance': performance}
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if dataName is None:
        dataName = metricName

    attrsDict = {
        **kwargs, **{
        'mousename': mousename,
        'metric': metricName,
        'target_dim': str(('sessions',) + dimord_to_labels(dimOrdTrg)).replace("'", "")
    }}

    dsDataLabels = ds.ping_data(dataName, attrsDict)
    if not skipExisting and len(dsDataLabels) > 0:
        dsuffix = dataName + '_' + '_'.join(attrsDict.values())
        print('Skipping existing', dsuffix)
    else:
        dataLst = dataDB.get_neuro_data({'mousename': mousename}, **kwargs)

        rez = []
        progBar = IntProgress(min=0, max=len(dataLst), description=mousename)
        display(progBar)  # display the bar
        for dataRSP in dataLst:
            if dataRSP.shape[0] < minTrials:
                print('Warning: skipping session with too few trials', dataRSP.shape[0])
                rez += [None]
            else:
                if dropChannels is not None:
                    nChannels = dataRSP.shape[2]
                    channelMask = np.ones(nChannels).astype(bool)
                    channelMask[dropChannels] = 0
                    dataRSP = dataRSP[:, :, channelMask]

                # Calculate stuff
                # IMPORTANT: DO NOT DO ZScore on cropped data at this point. ZScore is done on whole data during extraction
                if not timeAvg:
                    mc.set_data(dataRSP, 'rsp', timeWindow=timeWindow)
                else:
                    if timeWindow is not None:
                        raise ValueError('Time-averaging and timeWindow are incompatible')

                    mc.set_data(np.nanmean(dataRSP, axis=1), 'rp')

                rez += [mc.metric3D(metricName, dimOrdTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)]
            progBar.value += 1

        if not np.all([r is None for r in rez]):
            ds.delete_rows(dsDataLabels, verbose=False)
            ds.save_data(dataName, numpy_nonelist_to_array(rez), attrsDict)
        else:
            print('Warning, all sessions had too few trials, this mouse is skipped')


def metric_by_selector(dataDB, mc, ds, selector, metricName, dimOrdTrg,
                       dataName=None, skipExisting=False, minTrials=1, dropChannels=None, dataFunc=None,
                       timeWindow=None, timeAvg=False, metricSettings=None, sweepSettings=None, **kwargs):

    # Drop all arguments that were not specified
    # In some use cases non-specified arguments are not implemented
    #kwargs = {'trialType': trialType, 'zscoreDim': zscoreDim, 'datatype': datatype, 'performance': performance}
    kwargs = {k: str(v) for k, v in kwargs.items() if (v is not None) and (v != 'None')}
    print(kwargs)

    if dataName is None:
        dataName = metricName

    attrsDict = {
        **kwargs, **selector, **{
            'metric': metricName,
            'target_dim': str(dimord_to_labels(dimOrdTrg)).replace("'", "")
        }}

    dsDataLabels = ds.ping_data(dataName, attrsDict)
    if not skipExisting and len(dsDataLabels) > 0:
        dsuffix = dataName + '_' + '_'.join(attrsDict.values())
        print('Skipping existing', dsuffix)
    else:
        if dataFunc is None:
            dataLst = dataDB.get_neuro_data(selector, **kwargs)
        else:
            dataLst = dataFunc(dataDB, selector, **kwargs)

        dataRSP = np.concatenate(dataLst, axis=0)
        print('--', dataRSP.shape)

        if len(dataLst) == 0:
            print('for', selector, kwargs, 'there are no sessions, skipping')
        elif len(dataRSP) < minTrials:
            print('for', selector, kwargs, 'too few trials', len(dataRSP), ', skipping')
        else:
            if dropChannels is not None:
                nChannels = dataRSP.shape[2]
                channelMask = np.ones(nChannels).astype(bool)
                channelMask[dropChannels] = 0
                dataRSP = dataRSP[:, :, channelMask]

            # Calculate stuff
            # IMPORTANT: DO NOT DO ZScore on cropped data at this point. ZScore is done on whole data during extraction
            if not timeAvg:
                mc.set_data(dataRSP, 'rsp', timeWindow=timeWindow)
            else:
                if timeWindow is not None:
                    raise ValueError('Time-averaging and timeWindow are incompatible')

                mc.set_data(np.nanmean(dataRSP, axis=1), 'rp')

            rez = mc.metric3D(metricName, dimOrdTrg, metricSettings=metricSettings, sweepSettings=sweepSettings)

            ds.delete_rows(dsDataLabels, verbose=False)
            ds.save_data(dataName, np.array(rez), attrsDict)


def calc_metric_mouse(dataDB, mc, ds, metricName, dimOrdTrg, nameSuffix, skipExisting=False, verbose=True,
                      metricSettings=None, sweepSettings=None, minTrials=1, dropChannels=None,
                      timeAvg=False, exclQueryLst=None, dataFunc=None, **kwargs):  # dataTypes='auto', trialTypeNames=None, perfNames=None, intervNames=None

    autoAppendDict = {'trialType' : ['None'], 'performance': ['None']}
    dps = DataParameterSweep(dataDB, exclQueryLst, autoAppendDict=autoAppendDict, mousename='auto', **kwargs)

    progBar = IntProgress(min=0, max=len(dps.sweepDF), description=nameSuffix)
    display(progBar)  # display the bar

    for idx, row in dps.sweepDF.iterrows():
        kwargs = dict(row)
        del kwargs['mousename']

        zscoreDim = 'rs' if kwargs['datatype'] == 'raw' else None

        if verbose:
            print(metricName, nameSuffix, kwargs)

        metric_by_selector(dataDB, mc, ds, {'mousename': row['mousename']}, metricName, dimOrdTrg,
                           dataName=nameSuffix,
                           skipExisting=skipExisting,
                           dropChannels=dropChannels,
                           minTrials=minTrials,
                           timeAvg=timeAvg,
                           zscoreDim=zscoreDim,
                           metricSettings=metricSettings,
                           sweepSettings=sweepSettings,
                           dataFunc=dataFunc,
                           **kwargs)

        progBar.value += 1


def calc_metric_session(dataDB, mc, ds, metricName, dimOrdTrg, nameSuffix, minTrials=1, skipExisting=False,
                        metricSettings=None, sweepSettings=None, dropChannels=None,
                        verbose=True, exclQueryLst=None, **kwargs):  # dataTypes='auto', trialTypeNames=None, perfNames=None, intervNames=None,

    autoAppendDict = {'trialType' : ['None'], 'performance': ['None']}
    dps = DataParameterSweep(dataDB, exclQueryLst, autoAppendDict=autoAppendDict, mousename='auto', **kwargs)

    progBar = IntProgress(min=0, max=len(dps.sweepDF), description=nameSuffix)
    display(progBar)  # display the bar

    for idx, row in dps.sweepDF.iterrows():
        kwargs = dict(row)
        del kwargs['mousename']

        zscoreDim = 'rs' if kwargs['datatype'] == 'raw' else None

        if verbose:
            print(metricName, nameSuffix, kwargs)

        metric_by_session(dataDB, mc, ds, row['mousename'], metricName, dimOrdTrg,
                          skipExisting=skipExisting,
                          dropChannels=dropChannels,
                          dataName=nameSuffix,
                          minTrials=minTrials,
                          zscoreDim=zscoreDim,
                          metricSettings=metricSettings,
                          sweepSettings=sweepSettings, timeAvg=True,
                          **kwargs)

        progBar.value += 1


def calc_metric_mouse_delay(dataDB, mc, ds, metricName, dimOrdTrg, nameSuffix, haveDelay=False,
                            tTrgDelay=2.0, tTrgRew=1.0, **kwargs):
    # Proxy that allows delay averaging upon data aquisition
    dataFunc = lambda dataDB, selector, **kwargs: get_data_list(dataDB, haveDelay, selector['mousename'],
                                                                tTrgDelay=tTrgDelay, tTrgRew=tTrgRew,**kwargs)

    calc_metric_mouse(dataDB, mc, ds, metricName, dimOrdTrg, nameSuffix, dataFunc=dataFunc, **kwargs)