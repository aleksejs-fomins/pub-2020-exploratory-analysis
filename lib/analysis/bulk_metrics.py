import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import IntProgress

from mesostat.utils.pandas_helper import pd_query, pd_move_cols_front

from lib.sych.metric_helper import metric_by_session, metric_by_selector


none2all = lambda x: x if x is not None else 'All'


def auto2val(x, xNone, xAutoFunc):
    if x is None:
        return xNone
    elif x == 'auto':
        return xAutoFunc()  # Auto may be expensive or not defined for some cases, only call when needed
    else:
        return x


def metric_mouse_bulk(dataDB, mc, ds, metricName, dimOrdTrg, nameSuffix,
                      metricSettings=None, sweepSettings=None,
                      trialTypeNames=None, perfNames=None, cropTime=None, verbose=True):

    dataTypes      = dataDB.get_data_types()
    perfNames      = auto2val(perfNames, [None], lambda : [None, 'naive', 'expert'])
    trialTypeNames = auto2val(trialTypeNames, [None], lambda : [None] + list(dataDB.get_trial_type_names()))

    nTot = len(dataDB.mice) * len(dataTypes) * len(perfNames) * len(trialTypeNames)
    progBar = IntProgress(min=0, max=nTot, description=nameSuffix)
    display(progBar)  # display the bar

    for mousename in dataDB.mice:
        for datatype in dataTypes:
            for performance in perfNames:
                for trialType in trialTypeNames:
                    zscoreDim = 'rs' if datatype == 'raw' else None

                    if verbose:
                        print(mousename, [metricName, nameSuffix, datatype, none2all(performance), none2all(trialType)])

                    metric_by_selector(dataDB, mc, ds, {'mousename': mousename}, metricName, dimOrdTrg,
                                       dataName=nameSuffix, datatype=datatype, trialType=trialType, cropTime=cropTime,
                                       performance=performance,
                                       zscoreDim=zscoreDim,
                                       metricSettings=metricSettings,
                                       sweepSettings=sweepSettings)

                    progBar.value += 1


def metric_mouse_bulk_vs_session(dataDB, mc, ds, metricName, dimOrdTrg, nameSuffix,
                                 metricSettings=None, sweepSettings=None,
                                 trialTypeNames=None, perfNames=None, verbose=True):

    dataTypes      = dataDB.get_data_types()
    perfNames      = auto2val(perfNames, [None], lambda : [None, 'naive', 'expert'])
    trialTypeNames = auto2val(trialTypeNames, [None], lambda : [None] + list(dataDB.get_trial_type_names()))

    nTot = len(dataDB.mice) * len(dataTypes) * len(perfNames) * len(trialTypeNames)
    progBar = IntProgress(min=0, max=nTot, description=metricName)
    display(progBar)  # display the bar

    for mousename in dataDB.mice:
        for datatype in dataTypes:
            for performance in perfNames:
                for trialType in trialTypeNames:
                    zscoreDim = 'rs' if datatype == 'raw' else None

                    if verbose:
                        print(mousename, [metricName, nameSuffix, datatype, none2all(performance), none2all(trialType)])

                    metric_by_session(dataDB, mc, ds, mousename, metricName, dimOrdTrg,
                                      dataName=nameSuffix, datatype=datatype, trialType=trialType,
                                      performance=performance,
                                      zscoreDim=zscoreDim,
                                      metricSettings=metricSettings,
                                      sweepSettings=sweepSettings)


def plot_metric_bulk(ds, metricName, nameSuffix, prepFunc=None, ylim=None, yscale=None, verbose=True, xFunc=None, dropCols=None):
    # 1. Extract all results for this test
    dfAll = ds.list_dsets_pd().fillna('None')
    if dropCols is not None:
        dfAll = dfAll.drop(dropCols, axis=1)

    dfAnalysis = pd_query(dfAll, {'metric' : metricName, "name" : nameSuffix})
    dfAnalysis = pd_move_cols_front(dfAnalysis, ['metric', 'name', 'mousename'])  # Move leading columns forwards for more informative printing/saving

    # Loop over all other columns except mousename
    colsExcl = list(set(dfAnalysis.columns) - {'mousename'})
    for colVals, dfSub in dfAnalysis.groupby(colsExcl):
        plt.figure()

        if verbose:
            print(list(colVals))

        for idxMouse, rowMouse in dfSub.iterrows():
            dataThis = ds.get_data(rowMouse['dset'])

            if prepFunc is not None:
                dataThis = prepFunc(dataThis)

            #                     if datatype == 'raw':
            #                         nTrialThis = dataDB.get_ntrial_bytype({'mousename' : row['mousename']}, trialType=trialType, performance=performance)
            #                         dataThis *= np.sqrt(48*nTrialThis)
            #                         print('--', row['mousename'], nTrialThis)

            if xFunc is None:
                plt.plot(dataThis, label=rowMouse['mousename'])
            else:
                plt.plot(xFunc(len(dataThis)), dataThis, label=rowMouse['mousename'])

        if yscale is not None:
            plt.yscale(yscale)

        dataName = rowMouse.drop(['datetime', 'dset', 'shape', 'target_dim'])
        dataName = '_'.join([str(el) for el in dataName])

        plt.legend()
        plt.ylim(ylim)
        plt.savefig('pics/' + dataName + '.pdf')
        plt.close()


def plot_TC(dataDB, ds, ylim=None, yscale=None, verbose=True):
    dfAll = ds.list_dsets_pd()
    for datatype in dataDB.get_data_types():
        for performance in [None, 'naive', 'expert']:
            for trialType in [None] + dataDB.get_trial_type_names():
                dataNameChannel = '_'.join(['avg_entropy', 'time-channel', datatype, none2all(performance), none2all(trialType)])
                dataNameBulk = '_'.join(['avg_entropy', 'time', datatype, none2all(performance), none2all(trialType)])
                dataNameTC = '_'.join(['total_corr', 'time', datatype, none2all(performance), none2all(trialType)])

                dfChannel = dfAll[dfAll['name'] == dataNameChannel]
                dfChannel = dfChannel.sort_values(by=['mousename'])

                dfBulk = dfAll[dfAll['name'] == dataNameBulk]
                dfBulk = dfBulk.sort_values(by=['mousename'])

                if verbose:
                    print(dataNameChannel, dataNameBulk)

                if len(dfChannel) != len(dfBulk):
                    raise ValueError('Non-matching bulk and channel entropy storage', len(dfChannel), len(dfBulk))
                if len(dfChannel) == 0:
                    print('--Nothing found, skipping')
                else:
                    plt.figure()
                    for (idxCh, rowCh), (idxB, rowB) in zip(dfChannel.iterrows(), dfBulk.iterrows()):
                        dataCh = ds.get_data(rowCh['dset'])   # (nTime, nChannel)
                        dataB = ds.get_data(rowB['dset'])     # (nTime, )

                        avgTC = np.mean(dataCh, axis=1) - dataB

                        plt.plot(np.arange(0, 8, 1 / 20), avgTC, label=rowCh['mousename'])

                    if yscale is not None:
                        plt.yscale(yscale)

                    plt.legend()
                    plt.ylim(ylim)
                    plt.savefig('pics/' + dataNameTC + '.pdf')
                    plt.close()
