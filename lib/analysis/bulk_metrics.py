import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import IntProgress

from lib.sych.metric_helper import metric_by_session, metric_by_selector



none2all = lambda x: x if x is not None else 'All'


def metric_mouse_bulk(dataDB, mc, ds, metricName, dimOrdTrg, nameSuffix,
                      metricSettings=None, sweepSettings=None,
                      trialTypeNames=None, perfNames=None, cropTime=None, verbose=True):

    mouseNameDummy = 'mvg_4'
    perfNames = ['naive', 'expert'] if perfNames is None else perfNames
    trialTypeNamesDummy = dataDB.get_trial_type_names(mouseNameDummy) if trialTypeNames is None else trialTypeNames

    nMice = len(dataDB.mice)
    nDataType = len(dataDB.get_data_types(mouseNameDummy))
    nPerf = 1 + len(perfNames)
    nTrialTypes = 1 + len(trialTypeNamesDummy)
    nTot = nMice * nDataType * nPerf * nTrialTypes
    progBar = IntProgress(min=0, max=nTot, description=nameSuffix)
    display(progBar)  # display the bar

    for mousename in dataDB.mice:
        for datatype in dataDB.get_data_types(mousename):
            for performance in [None] + perfNames:
                trialTypeNames = dataDB.get_trial_type_names(mousename) if trialTypeNames is None else trialTypeNames

                for trialType in [None] + trialTypeNames:
                    dataName = '_'.join([metricName, nameSuffix, datatype, none2all(performance), none2all(trialType)])
                    zscoreDim = 'rs' if datatype == 'raw' else None

                    if verbose:
                        print(mousename, dataName)

                    metric_by_selector(dataDB, mc, ds, {'mousename': mousename}, metricName, dimOrdTrg,
                                       dataName=dataName, datatype=datatype, trialType=trialType, cropTime=cropTime,
                                       performance=performance,
                                       zscoreDim=zscoreDim,
                                       metricSettings=metricSettings,
                                       sweepSettings=sweepSettings)

                    progBar.value += 1


def metric_mouse_bulk_vs_session(dataDB, mc, ds, metricName, dimOrdTrg,
                                 metricSettings=None, sweepSettings=None, trialTypeNames=None, verbose=True):

    mouseNameDummy = 'mvg_4'
    trialTypeNamesDummy = dataDB.get_trial_type_names(mouseNameDummy) if trialTypeNames is None else trialTypeNames

    nMice = len(dataDB.mice)
    nDataType = len(dataDB.get_data_types(mouseNameDummy))
    nPerf = 3
    nTrialTypes = 1 + len(trialTypeNamesDummy)
    nTot = nMice * nDataType * nPerf * nTrialTypes
    progBar = IntProgress(min=0, max=nTot, description=metricName)
    display(progBar)  # display the bar


    for mousename in dataDB.mice:
        for datatype in dataDB.get_data_types(mousename):
            for performance in [None, 'naive', 'expert']:
                trialTypeNames = dataDB.get_trial_type_names(mousename) if trialTypeNames is None else trialTypeNames

                for trialType in [None] + trialTypeNames:
                    dataName = '_'.join([metricName, 'session', datatype, none2all(performance), none2all(trialType)])
                    zscoreDim = 'rs' if datatype == 'raw' else None

                    if verbose:
                        print(mousename, dataName)

                    metric_by_session(dataDB, mc, ds, mousename, metricName, dimOrdTrg,
                                      dataName=dataName, datatype=datatype, trialType=trialType,
                                      performance=performance,
                                      zscoreDim=zscoreDim,
                                      metricSettings=metricSettings,
                                      sweepSettings=sweepSettings)


def plot_metric_bulk(dataDB, ds, metricName, nameSuffix, prepFunc=None, ylim=None, yscale=None, verbose=True):
    dummyMouseName = 'mvg_4'

    dfAll = ds.list_dsets_pd()
    for datatype in dataDB.get_data_types(dummyMouseName):
        for performance in [None, 'naive', 'expert']:
            for trialType in [None] + dataDB.get_trial_type_names(dummyMouseName):
                dataName = '_'.join([metricName, nameSuffix, datatype, none2all(performance), none2all(trialType)])
                dfThis = dfAll[dfAll['name'] == dataName]
                dfThis = dfThis.sort_values(by=['mousename'])

                if verbose:
                    print(dataName)

                if len(dfThis) == 0:
                    print('--Nothing found, skipping')
                else:
                    plt.figure()
                    for idx, row in dfThis.iterrows():
                        dataThis = ds.get_data(row['dset'])
                        if prepFunc is not None:
                            dataThis = prepFunc(dataThis)

                        #                     if datatype == 'raw':
                        #                         nTrialThis = dataDB.get_ntrial_bytype({'mousename' : row['mousename']}, trialType=trialType, performance=performance)
                        #                         dataThis *= np.sqrt(48*nTrialThis)
                        #                         print('--', row['mousename'], nTrialThis)

                        if nameSuffix == 'time':
                            plt.plot(np.arange(0, 8, 1 / 20), dataThis, label=row['mousename'])
                        else:
                            plt.plot(dataThis, label=row['mousename'])

                    if yscale is not None:
                        plt.yscale(yscale)

                    plt.legend()
                    plt.ylim(ylim)
                    plt.savefig('pics/' + dataName + '.pdf')
                    plt.close()


def plot_metric_bulk_vs_session(dataDB, ds, metricName, trialTypeNames=None, ylim=None, verbose=True):
    dummyMouseName = 'mvg_4'

    dfAll = ds.list_dsets_pd()
    for datatype in dataDB.get_data_types(dummyMouseName):
        for performance in [None, 'naive', 'expert']:
            if trialTypeNames is None:
                trialTypeNamesThis = dataDB.get_trial_type_names(dummyMouseName)
            else:
                trialTypeNamesThis = trialTypeNames

            for trialType in [None] + trialTypeNamesThis:
                dataName = '_'.join([metricName, 'session', datatype, none2all(performance), none2all(trialType)])
                dfThis = dfAll[dfAll['name'] == dataName]
                dfThis = dfThis.sort_values(by=['mousename'])

                if verbose:
                    print(dataName, len(dfThis))

                plt.figure()
                for idx, row in dfThis.iterrows():
                    dataThis = ds.get_data(row['dset'])

                    plt.plot(dataThis, label=row['mousename'])

                plt.legend()
                plt.ylim(ylim)
                plt.savefig('pics/' + dataName + '.pdf')
                plt.close()