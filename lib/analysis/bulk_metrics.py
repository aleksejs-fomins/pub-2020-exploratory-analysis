import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import IntProgress
import seaborn as sns

from mesostat.utils.system import make_path
from mesostat.utils.pandas_helper import pd_query, pd_move_cols_front, pd_append_row
from mesostat.stat.machinelearning import drop_nan_rows
from mesostat.visualization.mpl_barplot import sns_barplot
from mesostat.visualization.mpl_matrix import plot_df_2D_outer_product

none2all = lambda x: x if x is not None else 'All'


def plot_metric_bulk_1D(dataDB, ds, metricName, nameSuffix, prepFunc=None, xlim=None, ylim=None, yscale=None,
                     verbose=True, xFunc=None, haveTimeLabels=False):#, dropCols=None):
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
        fig, ax = plt.subplots(figsize=(4, 4))

        if verbose:
            print(list(colVals))

        for idxMouse, rowMouse in dfSub.sort_values(by='mousename').iterrows():
            print(list(rowMouse.values))

            dataThis = ds.get_data(rowMouse['dset'])
            assert dataThis.ndim == 1, 'Only using 1D data for this plot function'

            if prepFunc is not None:
                dataThis = prepFunc(dataThis)

            #                     if datatype == 'raw':
            #                         nTrialThis = dataDB.get_ntrial_bytype({'mousename' : row['mousename']}, trialType=trialType, performance=performance)
            #                         dataThis *= np.sqrt(48*nTrialThis)
            #                         print('--', row['mousename'], nTrialThis)

            x = np.arange(len(dataThis)) if xFunc is None else np.array(xFunc(rowMouse['mousename'], len(dataThis)))
            x, dataThis = drop_nan_rows([x, dataThis])

            ax.plot(x, dataThis, label=rowMouse['mousename'])

        if yscale is not None:
            ax.set_yscale(yscale)

        if haveTimeLabels:
            dataDB.label_plot_timestamps(ax, linecolor='y', textcolor='k', shX=-0.5, shY=0.05)

        dataName = rowMouse.drop(['dset', 'mousename'])
        dataName = '_'.join([str(el) for el in dataName])

        prefixPath = 'pics/bulk/' + metricName + '/'
        make_path(prefixPath)

        ax.legend()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(nameSuffix)
        ax.set_ylabel(metricName)
        plt.savefig(prefixPath + dataName + '.png', dpi=200)
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

        prefixPath = 'pics/bulk/' + metricName + '/'
        make_path(prefixPath)

        fig.savefig(prefixPath + dataName + '.png')
        plt.close()


def barplot_conditions(ds, metricName, nameSuffix, verbose=True, trialTypes=None, intervNames=None):
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
        plotSuffix = '_'.join(key) if isinstance(key, list) else '_'.join([key])

        if verbose:
            print(plotSuffix)

        intervNamesData = list(set(dfDataType['intervName']))
        trialTypesData =  list(set(dfDataType['trialType']))

        intervNames = intervNamesData if intervNames is None else [i for i in intervNames if i in intervNamesData]
        trialTypes = trialTypesData if trialTypes is None else [i for i in trialTypes if i in trialTypesData]

        #################################
        # Plot 1 ::: Mouse * TrialType
        #################################

        for intervName in intervNames:
            df1 = pd_query(dfDataType, {'intervName' : intervName})
            if trialTypes is not None:
                df1 = df1[df1['trialType'].isin(trialTypes)]

            dfData1 = pd.DataFrame(columns=['mousename', 'trialType', metricName])

            for idx, row in df1.iterrows():
                data = ds.get_data(row['dset'])
                for d in data:
                    dfData1 = pd_append_row(dfData1, [row['mousename'], row['trialType'], d])

            mice = sorted(set(dfData1['mousename']))
            fig, ax = plt.subplots()
            sns_barplot(ax, dfData1, "mousename", metricName, 'trialType', annotHue=True, xOrd=mice, hOrd=trialTypes)
            # sns.barplot(ax=ax, x="mousename", y=metricName, hue='trialType', data=dfData1)

            prefixPath = 'pics/bulk/' + metricName + '/barplot_conditions/'
            make_path(prefixPath)

            fig.savefig(prefixPath + 'barplot_trialtype_' + plotSuffix + '_' + intervName + '.png', dpi=300)
            plt.close()

        #################################
        # Plot 2 ::: Mouse * Phase
        #################################

        for trialType in ['None'] + trialTypes:
            df2 = pd_query(dfDataType, {'trialType' : trialType})

            # display(df2.head())

            df2 = df2[df2['intervName'] != 'AVG']
            if key[0] == 'bn_trial':
                df2 = df2[df2['intervName'] != 'PRE']

            dfData2 = pd.DataFrame(columns=['mousename', 'phase', metricName])

            for idx, row in df2.iterrows():
                data = ds.get_data(row['dset'])
                for d in data:
                    dfData2 = pd_append_row(dfData2, [row['mousename'], row['intervName'], d])

            dfData2 = dfData2.sort_values('mousename')

            mice = sorted(set(dfData2['mousename']))
            fig, ax = plt.subplots()
            sns_barplot(ax, dfData2, "mousename", metricName, 'phase', annotHue=False, xOrd=mice, hOrd=intervNames)
            # sns.barplot(ax=ax, x="mousename", y=metricName, hue='phase', data=dfData2)

            prefixPath = 'pics/bulk/' + metricName + '/barplot_conditions/'
            make_path(prefixPath)

            fig.savefig(prefixPath + 'barplot_phase_' + plotSuffix + '_' + trialType + '.png', dpi=300)
            plt.close()


def plot_2D_outer_bymouse(dataDB, ds, metricName, nameSuffix='bymouse'):
    # Read dataframe, filter out desired metric
    df = ds.list_dsets_pd()
    dfEff = pd_query(df, {'metric': metricName, 'name': nameSuffix}).copy()

    # Read and append scalar to the dataframe
    rezLst = []
    for idx, row in dfEff.iterrows():
        rezLst += [ds.get_data(row['dset'])]
    dfEff[metricName] = np.array(rezLst)

    # Drop useless columns
    dfEff.drop(['dset', 'shape', 'datetime', 'target_dim', 'zscoreDim'], axis=1)

    # Drop averages for this plot
    dfEff = dfEff.loc[dfEff['intervName'] != 'AVG']

    fig, ax = plt.subplots(figsize=(4, 4))
    plot_df_2D_outer_product(ax, dfEff, ['datatype', 'mousename'], ['intervName', 'trialType'],
                             'rank_effective', orderDict={'intervName': dataDB.get_interval_names()}, vmin=1)
    plt.savefig(metricName + '_' + nameSuffix + '.svg')
    plt.show()


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

