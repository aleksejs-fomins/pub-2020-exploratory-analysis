import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import IntProgress

from mesostat.utils.system import make_path
from mesostat.visualization.mpl_timescale_bar import add_timescale_bar
from mesostat.visualization.opencv_video import merge_images_cv2

from lib.common.param_sweep import DataParameterSweep, param_vals_to_suffix, pd_row_to_kwargs


def merge_image_sequence_movie(pathprefix, suffix, idxMin, idxMax, trgPathName=None, deleteSrc=False, fps=30):
    srcPwdLst = []
    for idx in range(idxMin, idxMax):
        pwd = pathprefix + str(idx) + suffix
        if not os.path.isfile(pwd):
            raise FileNotFoundError('File not found', pwd)
        srcPwdLst += [pwd]

    if trgPathName is None:
        trgPathName = os.path.splitext(pathprefix + suffix)[0]

    merge_images_cv2(srcPwdLst, trgPathName, fps=fps, FOURCC='MJPG', isColor=True)

    if deleteSrc:
        for pwd in srcPwdLst:
            os.remove(pwd)


def cluster_brain_plot(fig, ax, dataDB, clusters, dropChannels=None):
    clusterDict = {c: np.where(clusters == c)[0] for c in sorted(set(clusters))}
    if dropChannels is not None:
        # Correct channel indices given that some channels were dropped
        dropChannels = np.array(dropChannels)
        clusterDict = {c: [el + np.sum(dropChannels < el) for el in v] for c, v in clusterDict.items()}

    dataDB.plot_area_clusters(fig, ax, clusterDict, haveLegend=True)


def movie_mouse_trialtype(dataDB, dataKWArgs, calcKWArgs, plotKWArgs, calc_func, plot_func,
                          prefixPath='', exclQueryLst=None, haveDelay=False, fontsize=20, tTrgDelay=2.0, tTrgRew=2.0):
    assert 'trialType' in dataKWArgs.keys(), 'Requires trial types'
    assert 'intervName' not in dataKWArgs.keys(), 'Movie intended for full range'
    dps = DataParameterSweep(dataDB, exclQueryLst, mousename='auto', **dataKWArgs)
    nMice = dps.param_size('mousename')
    nTrialType = dps.param_size('trialType')

    for paramVals, dfTmp in dps.sweepDF.groupby(dps.invert_param(['mousename', 'trialType'])):
        plotSuffix = param_vals_to_suffix(paramVals)

        # Store all preprocessed data first
        dataDict = {}
        for mousename, dfMouse in dfTmp.groupby(['mousename']):
            for idx, row in dfMouse.iterrows():
                trialType = row['trialType']
                print('Reading data, ', plotSuffix, mousename, trialType)

                kwargsThis = pd_row_to_kwargs(row, parseNone=True, dropKeys=['mousename'])

                dataDict[(mousename, trialType)] = calc_func(dataDB, mousename, calcKWArgs, haveDelay=haveDelay,
                                                             tTrgDelay=tTrgDelay, tTrgRew=tTrgRew, **kwargsThis)

        # Test that all datasets have the same duration
        shapeSet = set([v.shape for v in dataDict.values()])
        assert len(shapeSet) == 1
        nTimes = shapeSet.pop()[0]

        progBar = IntProgress(min=0, max=nTimes, description=plotSuffix)
        display(progBar)  # display the bar
        for iTime in range(nTimes):
            make_path(prefixPath)
            outfname = prefixPath + plotSuffix + '_' + str(iTime) + '.png'

            if os.path.isfile(outfname):
                print('Already calculated', iTime, 'skipping')
                progBar.value += 1
                continue

            fig, ax = plt.subplots(nrows=nMice, ncols=nTrialType, figsize=(4 * nTrialType, 4 * nMice), tight_layout=True)

            for iMouse, mousename in enumerate(dps.param('mousename')):
                ax[iMouse][0].set_ylabel(mousename, fontsize=fontsize)
                for iTT, trialType in enumerate(dps.param('trialType')):
                    ax[0][iTT].set_title(trialType, fontsize=fontsize)
                    # print(datatype, mousename)

                    dataP = dataDict[(mousename, trialType)][iTime]

                    rightMost = iTT == nTrialType - 1
                    plot_func(dataDB, fig, ax[iMouse][iTT], dataP, haveColorBar=rightMost, **plotKWArgs)

            # Add a timescale bar to the figure
            timestamps = dataDB.get_timestamps(mousename, session=None)
            if 'delay' not in timestamps.keys():
                tsKeys = ['PRE'] + list(timestamps.keys())
                tsVals = list(timestamps.values()) + [nTimes / dataDB.targetFPS]
            else:
                tsKeys = ['PRE'] + list(timestamps.keys()) + ['reward']
                tsVals = list(timestamps.values()) + [timestamps['delay'] + tTrgDelay, nTimes / dataDB.targetFPS]

            print(tsVals, iTime / dataDB.targetFPS)
            add_timescale_bar(fig, tsKeys, tsVals, iTime / dataDB.targetFPS)

            fig.savefig(outfname, bbox_inches='tight')
            # plt.close()
            plt.cla()
            plt.clf()
            plt.close('all')
            progBar.value += 1
    return prefixPath
