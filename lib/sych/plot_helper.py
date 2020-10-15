import matplotlib.pyplot as plt
from ast import literal_eval as str2tuple

from mesostat.utils.pandas_helper import get_rows_colvals
from mesostat.utils.plotting import imshow


def imshow_dataset_by_mouse(dataDB, ds, dsetName, plotNameSuffix='', limits=None, cmap='jet', aspect=None, fig1size=(5,5)):
    resultDF = ds.list_dsets_pd()

    fig, ax = plt.subplots(ncols=len(dataDB.mice), figsize=(fig1size[0] * len(dataDB.mice), fig1size[1]))
    for iMouse, mousename in enumerate(sorted(dataDB.mice)):
        rows = get_rows_colvals(resultDF, {"mousename" : mousename, "name" : dsetName})

        # Find index of the latest result
        maxRowIdx = rows['datetime'].idxmax()
        dsetKey = rows.loc[maxRowIdx]['dset']
        shapeLabels = str2tuple(rows.loc[maxRowIdx]['target_dim'])

        # Get data
        data = ds.get_data(dsetKey)

        # Plot data
        imshow(fig, ax[iMouse], data, xlabel=shapeLabels[0], ylabel=shapeLabels[1], title=mousename, haveColorBar=True, limits=limits, cmap=cmap, aspect=aspect)

        thrIdx = dataDB.expertThrIdx[mousename]
        if thrIdx is not None:
            ax[iMouse].axhline(y=thrIdx, color='w', linestyle='--')

    plt.savefig(dsetName + plotNameSuffix + '.pdf')
    plt.show()


def imshow_dataset_by_session(dataDB, ds, dsetName, plotNameSuffix='', limits=None, cmap='jet', aspect=None, colBased=True, fig1size=(5,5)):
    resultDF = ds.list_dsets_pd()

    for iMouse, mousename in enumerate(sorted(dataDB.mice)):
        rows = get_rows_colvals(resultDF, {"mousename" : mousename, "name" : dsetName})

        # Find index of the latest result
        maxRowIdx = rows['datetime'].idxmax()
        dsetKey = rows.loc[maxRowIdx]['dset']
        shapeLabels = str2tuple(rows.loc[maxRowIdx]['target_dim'])

        # Get data
        data = ds.get_data(dsetKey)
        assert data.ndim == 3, "Dataset must include sessions and 2 other dimensions"
        # assert len(shapeLabels), "Dataset must include sessions and 2 other dimensions"
        # assert shapeLabels[0] == 'sessions', "First dataset dimension must be sessions"

        rowsNeuro = dataDB.get_rows('neuro', {'mousename': mousename})
        nSession = len(rowsNeuro)

        if colBased:
            fig, ax = plt.subplots(ncols=nSession,  figsize=(fig1size[0] * nSession, fig1size[1]))
        else:
            fig, ax = plt.subplots(nrows=nSession, figsize=(fig1size[0], fig1size[1] * nSession))
        # fig.suptitle(mousename)

        for iSession, (idx, row) in enumerate(rowsNeuro.iterrows()):
            imshow(fig, ax[iSession], data[iSession], xlabel=None, ylabel=None, title=row['mousekey'], haveColorBar=True, limits=limits, cmap=cmap, aspect=aspect)

        plt.savefig(dsetName + "_" + mousename + plotNameSuffix + '.pdf')
        plt.show()
