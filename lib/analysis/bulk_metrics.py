import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import IntProgress
import seaborn as sns
import statannot

from mesostat.utils.pandas_helper import pd_query, pd_move_cols_front, outer_product_df, pd_append_row
from mesostat.stat.machinelearning import drop_nan_rows
from mesostat.visualization.mpl_barplot import sns_barplot

from lib.sych.metric_helper import metric_by_session, metric_by_selector


none2all = lambda x: x if x is not None else 'All'


# Switch between 3 cases:
#    If x is None, do nothing
#    If x='auto', use provided function result for the dictionary element
#    Otherwise, use provided x value for the dictionary element
def _dict_append_auto(d, key, x, xAutoFunc):
    if x == 'auto':
        d[key] = xAutoFunc()  # Auto may be expensive or not defined for some cases, only call when needed
    elif x is not None:
        d[key] = x


def metric_mouse_bulk(dataDB, mc, ds, metricName, dimOrdTrg, nameSuffix, skipExisting=False, verbose=True,
                      metricSettings=None, sweepSettings=None, minTrials=1, dropChannels=None,
                      dataTypes='auto', trialTypeNames=None, perfNames=None, cropTime=None, timeAvg=False):

    argSweepDict = {'mousename' : dataDB.mice}
    _dict_append_auto(argSweepDict, 'datatype', dataTypes, dataDB.get_data_types)
    _dict_append_auto(argSweepDict, 'performance', perfNames, lambda : [None, 'naive', 'expert'])
    _dict_append_auto(argSweepDict, 'trialType', trialTypeNames, lambda : [None] + list(dataDB.get_trial_type_names()))
    sweepDF = outer_product_df(argSweepDict)

    progBar = IntProgress(min=0, max=len(sweepDF), description=nameSuffix)
    display(progBar)  # display the bar

    for idx, row in sweepDF.iterrows():
        kwargs = dict(row)
        del kwargs['mousename']

        zscoreDim = 'rs' if kwargs['datatype'] == 'raw' else None

        if verbose:
            print(metricName, nameSuffix, kwargs)

        metric_by_selector(dataDB, mc, ds, {'mousename': row['mousename']}, metricName, dimOrdTrg,
                           dataName=nameSuffix,
                           skipExisting=skipExisting,
                           dropChannels=dropChannels,
                           cropTime=cropTime,
                           minTrials=minTrials,
                           timeAvg=timeAvg,
                           zscoreDim=zscoreDim,
                           metricSettings=metricSettings,
                           sweepSettings=sweepSettings,
                           **kwargs)

        progBar.value += 1


def metric_mouse_bulk_vs_session(dataDB, mc, ds, metricName, nameSuffix, minTrials=1, skipExisting=False,
                                 metricSettings=None, sweepSettings=None, dropChannels=None, dataTypes='auto',
                                 trialTypeNames=None, perfNames=None, cropTime=None, verbose=True):


    argSweepDict = {'mousename' : dataDB.mice}
    _dict_append_auto(argSweepDict, 'datatype', dataTypes, dataDB.get_data_types)
    _dict_append_auto(argSweepDict, 'performance', perfNames, lambda : [None, 'naive', 'expert'])
    _dict_append_auto(argSweepDict, 'trialType', trialTypeNames, lambda : [None] + list(dataDB.get_trial_type_names()))
    sweepDF = outer_product_df(argSweepDict)

    progBar = IntProgress(min=0, max=len(sweepDF), description=nameSuffix)
    display(progBar)  # display the bar

    for idx, row in sweepDF.iterrows():
        kwargs = dict(row)
        del kwargs['mousename']

        zscoreDim = 'rs' if kwargs['datatype'] == 'raw' else None

        if verbose:
            print(metricName, nameSuffix, kwargs)

        metric_by_session(dataDB, mc, ds, row['mousename'], metricName, '',
                          skipExisting=skipExisting,
                          dropChannels=dropChannels,
                          dataName=nameSuffix,
                          minTrials=minTrials,
                          zscoreDim=zscoreDim,
                          metricSettings=metricSettings,
                          sweepSettings=sweepSettings, timeAvg=True, cropTime=cropTime,
                          **kwargs)

        progBar.value += 1


def plot_metric_bulk(ds, metricName, nameSuffix, prepFunc=None, xlim=None, ylim=None, yscale=None,
                     verbose=True, xFunc=None):#, dropCols=None):
    # 1. Extract all results for this test
    dfAll = ds.list_dsets_pd().fillna('None')
    # if dropCols is not None:
    #     dfAll = dfAll.drop(dropCols, axis=1)

    dfAnalysis = pd_query(dfAll, {'metric' : metricName, "name" : nameSuffix})
    dfAnalysis = pd_move_cols_front(dfAnalysis, ['metric', 'name', 'mousename'])  # Move leading columns forwards for more informative printing/saving
    dfAnalysis = dfAnalysis.drop(['target_dim', 'datetime', 'shape'], axis=1)

    # Loop over all other columns except mousename
    colsExcl = list(set(dfAnalysis.columns) - {'mousename', 'dset'})

    for colVals, dfSub in dfAnalysis.groupby(colsExcl):
        plt.figure()

        if verbose:
            print(list(colVals))

        for idxMouse, rowMouse in dfSub.sort_values(by='mousename').iterrows():
            print(list(rowMouse.values))

            dataThis = ds.get_data(rowMouse['dset'])

            if prepFunc is not None:
                dataThis = prepFunc(dataThis)

            #                     if datatype == 'raw':
            #                         nTrialThis = dataDB.get_ntrial_bytype({'mousename' : row['mousename']}, trialType=trialType, performance=performance)
            #                         dataThis *= np.sqrt(48*nTrialThis)
            #                         print('--', row['mousename'], nTrialThis)

            x = np.arange(len(dataThis)) if xFunc is None else np.array(xFunc(rowMouse['mousename'], len(dataThis)))
            x, dataThis = drop_nan_rows([x, dataThis])

            plt.plot(x, dataThis, label=rowMouse['mousename'])

        if yscale is not None:
            plt.yscale(yscale)

        dataName = rowMouse.drop(['dset', 'mousename'])
        dataName = '_'.join([str(el) for el in dataName])

        plt.legend()
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.savefig('pics/' + dataName + '.png')
        plt.close()


def scatter_metric_bulk(ds, metricName, nameSuffix, prepFunc=None, xlim=None, ylim=None, yscale=None,
                        verbose=True, xFunc=None, haveRegression=False):#, dropCols=None):
    # 1. Extract all results for this test
    dfAll = ds.list_dsets_pd().fillna('None')
    # if dropCols is not None:
    #     dfAll = dfAll.drop(dropCols, axis=1)

    dfAnalysis = pd_query(dfAll, {'metric' : metricName, "name" : nameSuffix})
    dfAnalysis = pd_move_cols_front(dfAnalysis, ['metric', 'name', 'mousename'])  # Move leading columns forwards for more informative printing/saving
    dfAnalysis = dfAnalysis.drop(['target_dim', 'datetime', 'shape'], axis=1)

    if 'performance' in dfAnalysis.columns:
        dfAnalysis = dfAnalysis[dfAnalysis['performance'] == 'None'].drop(['performance'], axis=1)

    # Loop over all other columns except mousename
    colsExcl = list(set(dfAnalysis.columns) - {'mousename', 'dset'})

    for colVals, dfSub in dfAnalysis.groupby(colsExcl):
        fig, ax = plt.subplots()

        if verbose:
            print(list(colVals))

        xLst = []
        yLst = []
        for idxMouse, rowMouse in dfSub.sort_values(by='mousename').iterrows():
            print(list(rowMouse.values))

            dataThis = ds.get_data(rowMouse['dset'])

            if prepFunc is not None:
                dataThis = prepFunc(dataThis)

            #                     if datatype == 'raw':
            #                         nTrialThis = dataDB.get_ntrial_bytype({'mousename' : row['mousename']}, trialType=trialType, performance=performance)
            #                         dataThis *= np.sqrt(48*nTrialThis)
            #                         print('--', row['mousename'], nTrialThis)

            x = np.arange(len(dataThis)) if xFunc is None else np.array(xFunc(rowMouse['mousename'], len(dataThis)))
            print(dataThis.shape)

            x, dataThis = drop_nan_rows([x, dataThis])
            print(dataThis.shape)

            ax.plot(x, dataThis, '.', label=rowMouse['mousename'])
            xLst += [x]
            yLst += [dataThis]

        if yscale is not None:
            plt.yscale(yscale)

        dataName = rowMouse.drop(['dset', 'mousename'])
        dataName = '_'.join([str(el) for el in dataName])

        ax.legend()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if haveRegression:
            sns.regplot(ax=ax, x=np.hstack(xLst), y=np.hstack(yLst), scatter=False)

        fig.savefig('pics/' + dataName + '.png')
        plt.close()


def barplot_conditions(ds, metricName, nameSuffix, verbose=True, trialTypes=None):
    '''
    Sweep over datatypes
    1. (Mouse * [iGO, iNOGO]) @ {interv='AVG'}
    2. (Mouse * interv / AVG) @ {trialType=None}
    '''

    # 1. Extract all results for this test
    dfAll = ds.list_dsets_pd().fillna('None')

    dfAnalysis = pd_query(dfAll, {'metric' : metricName, "name" : nameSuffix})
    dfAnalysis = pd_move_cols_front(dfAnalysis, ['metric', 'name', 'mousename'])  # Move leading columns forwards for more informative printing/saving
    dfAnalysis = dfAnalysis.drop(['target_dim', 'datetime', 'shape'], axis=1)

    if 'performance' in dfAnalysis.columns:
        sweepLst = ['datatype', 'performance']
    else:
        sweepLst = ['datatype']

    for key, dfDataType in dfAnalysis.groupby(sweepLst):
        plotSuffix = '_'.join(key)

        #################################
        # Plot 1 ::: Mouse * TrialType
        #################################

        df1 = pd_query(dfDataType, {'cropTime' : 'AVG'})
        if trialTypes is not None:
            df1 = df1[df1['trialType'].isin(trialTypes)]

        dfData1 = pd.DataFrame(columns=['mousename', 'trialType', metricName])

        for idx, row in df1.iterrows():
            data = ds.get_data(row['dset'])
            for d in data:
                dfData1 = pd_append_row(dfData1, [row['mousename'], row['trialType'], d])

        dfData1 = dfData1.sort_values('mousename')
        fig, ax = plt.subplots()
        sns_barplot(ax, dfData1, "mousename", metricName, 'trialType', annotHue=True)
        # sns.barplot(ax=ax, x="mousename", y=metricName, hue='trialType', data=dfData1)
        fig.savefig(metricName + '_barplot_trialtype_' + plotSuffix + '.png')
        plt.close()

        #################################
        # Plot 2 ::: Mouse * Phase
        #################################

        df2 = pd_query(dfDataType, {'trialType' : 'None'})

        display(df2.head())

        df2 = df2[df2['cropTime'] != 'AVG']
        if key[0] == 'bn_trial':
            df2 = df2[df2['cropTime'] != 'PRE']

        dfData2 = pd.DataFrame(columns=['mousename', 'phase', metricName])

        for idx, row in df2.iterrows():
            data = ds.get_data(row['dset'])
            for d in data:
                dfData2 = pd_append_row(dfData2, [row['mousename'], row['cropTime'], d])

        dfData2 = dfData2.sort_values('mousename')

        fig, ax = plt.subplots()
        sns_barplot(ax, dfData2.sort_values(by=['phase', 'mousename']), "mousename", metricName, 'phase', annotHue=False)
        # sns.barplot(ax=ax, x="mousename", y=metricName, hue='phase', data=dfData2)
        fig.savefig(metricName + '_barplot_phase_' + plotSuffix + '.png')
        plt.close()


# TODO: Replace nested for with pandas iterator. Make sure None arguments are not iterated over
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

