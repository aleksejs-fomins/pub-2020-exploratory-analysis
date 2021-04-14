import numpy as np
import matplotlib.pyplot as plt

from mesostat.visualization.mpl_matrix import imshow


def imshow_dataset_by_mouse(dataDB, ds, dsetName, plotNameSuffix='', limits=None, cmap='jet',
                            aspect=None, fig1size=(5,5), havePerf=True, dropX=None, dropY=None):
    resultDF = ds.list_dsets_pd()

    fig, ax = plt.subplots(ncols=len(dataDB.mice), figsize=(fig1size[0] * len(dataDB.mice), fig1size[1]), squeeze=False)
    fig.suptitle(dsetName)
    for iMouse, mousename in enumerate(sorted(dataDB.mice)):
        queryDict = {"mousename" : mousename, "name" : dsetName}
        data, attrs = ds.get_data_recent_by_query(queryDict, listDF=resultDF)
        shapeLabels = attrs['target_dim']

        mat = data.T
        print(mat.shape)
        if dropX:
            mat[dropX] = np.nan
        if dropY:
            mat[:, dropY] = np.nan

        # Plot data
        imshow(fig, ax[0][iMouse], mat, xlabel=shapeLabels[0], ylabel=shapeLabels[1], title=mousename,
               haveColorBar=True, limits=limits, cmap=cmap, aspect=aspect)

        if havePerf:
            thrIdx = dataDB.get_first_expert_session_idx(mousename)
            if thrIdx is not None:
                ax[0][iMouse].axvline(x=thrIdx, color='w', linestyle='--')

    plt.savefig(dsetName + plotNameSuffix + '.pdf')
    plt.show()


def imshow_dataset_by_session(dataDB, ds, dsetName, plotNameSuffix='', limits=None,
                              cmap='jet', aspect=None, colBased=True, fig1size=(5,5)):
    resultDF = ds.list_dsets_pd()

    for iMouse, mousename in enumerate(sorted(dataDB.mice)):
        # Get data
        queryDict = {"mousename" : mousename, "name" : dsetName}
        data, attrs = ds.get_data_recent_by_query(queryDict, listDF=resultDF)
        shapeLabels = attrs['target_dim']

        assert data.ndim == 3, "Dataset must include sessions and 2 other dimensions"
        # assert len(shapeLabels), "Dataset must include sessions and 2 other dimensions"
        # assert shapeLabels[0] == 'sessions', "First dataset dimension must be sessions"

        sessions = dataDB.get_sessions(mousename)
        nSession = len(sessions)

        if colBased:
            fig, ax = plt.subplots(ncols=nSession,  figsize=(fig1size[0] * nSession, fig1size[1]))
        else:
            fig, ax = plt.subplots(nrows=nSession, figsize=(fig1size[0], fig1size[1] * nSession))
        # fig.suptitle(mousename)

        for iSession, session in enumerate(sessions):
            imshow(fig, ax[iSession], data[iSession], xlabel=None, ylabel=None, title=session, haveColorBar=True, limits=limits, cmap=cmap, aspect=aspect)

        plt.savefig(dsetName + "_" + mousename + plotNameSuffix + '.pdf')
        plt.show()
