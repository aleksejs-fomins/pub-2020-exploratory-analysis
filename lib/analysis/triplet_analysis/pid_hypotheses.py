import h5py
import numpy as np
import matplotlib.pyplot as plt

from mesostat.utils.signals.resample import bin_data
from mesostat.utils.signals.filter import drop_PCA
from mesostat.visualization.mpl_colorbar import imshow_add_color_bar
import mesostat.utils.iterators.matrix as matiter

from lib.analysis.triplet_compute.pid_common import pid


'''
Hypothesis-plots:
Plots for specific subsets of sources and target constituting a hypothesis
'''

def hypotheses_calc_pid(dataDB, mc, hDict, h5outname, datatypes=None, nDropPCA=None, **kwargs):
    if datatypes is None:
        datatypes = dataDB.get_data_types()

    for hLabel, (intervName, sources, targets) in hDict.items():
        for datatype in datatypes:
            for mousename in dataDB.mice:
                print(hLabel, datatype, mousename)

                # Calculate PID
                channelNames = dataDB.get_channel_labels(mousename)
                dataLst = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype,
                                                zscoreDim=None, intervName=intervName, **kwargs)

                rezLabels, rezValues = pid(dataLst, mc, channelNames, sources, targets, nBin=4, nDropPCA=nDropPCA,
                                           labelsAsText=True)

                with h5py.File(h5outname, 'a') as h5f:
                    fieldNameLabels = '_'.join(['PID_labels', hLabel, mousename, datatype])
                    fieldNameValues = '_'.join(['PID_values', hLabel, mousename, datatype])
                    h5f[fieldNameLabels] = np.array(rezLabels)
                    h5f[fieldNameValues] = np.array(rezValues)

                #     # Display resulting dataframe
                #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                #         display(rezDF.sort_values(by=['S1', 'S2', 'T', 'PID', 'mousename']))

                # # Save to file
                # dataLabel = '_'.join(['PID', datatype, hLabel])
                # rezDF.to_hdf(h5outname, dataLabel, mode='a', format='table', data_columns=True)


def hypotheses_plot_pid_3D(dataDB, hDict, h5outname, datatypes=None, yscale='log', clip=(1.0E-3, 1), ylim=(1.0E-3, 1)):
    if datatypes is None:
        datatypes = dataDB.get_data_types()

    for hLabel in hDict.keys():
        for datatype in datatypes:

            nMice = len(dataDB.mice)
            fig, ax = plt.subplots(ncols=nMice, figsize=(6 * nMice, 6))

            for iMouse, mousename in enumerate(dataDB.mice):
                groupKey = '_'.join([hLabel, mousename, datatype])
                print(groupKey)

                with h5py.File(h5outname, 'r') as h5f:
                    labels = h5f['PID_labels_' + groupKey]
                    values = h5f['PID_values_' + groupKey]

                for label, value in zip(labels, values):
                    if clip is not None:
                        value = np.clip(value, *clip)

                    ax[iMouse].plot(value, label=label)

                ax[iMouse].legend()
                ax[iMouse].set_title(mousename)
                ax[iMouse].set_ylabel('Info(Bits)')
                ax[iMouse].set_xticks([0, 1, 2, 3])
                ax[iMouse].set_xticklabels(['Unique1', 'Unique2', 'Redundancy', 'Synergy'])
                if yscale is not None:
                    ax[iMouse].set_yscale(yscale)
                if ylim is not None:
                    ax[iMouse].set_ylim(ylim)

            plt.savefig('pics/PID_BY_MOUSE_'+ hLabel + '_' + datatype + '.pdf')
            plt.close(fig)


def hypotheses_calc_plot_info3D(dataDB, hDict, datatypes=None, nBin=4, nDropPCA=None, **kwargs):
    if datatypes is None:
        datatypes = dataDB.get_data_types()

    for datatype in datatypes:
        for hLabel, (intervName, sources, targets) in hDict.items():
            print(hLabel)

            dataLabel = '_'.join(['PID', datatype, hLabel, intervName])

            # Find combinations of all source pairs
            nSrc = len(sources)
            sourcePairs =  matiter.sample_list(sources, matiter.iter_g_2D(nSrc))  # Pairs of sources
            for s1Label, s2Label in sourcePairs:
                for labelTrg in targets:
                    nMice = len(dataDB.mice)
                    fig, ax = plt.subplots(nrows=nMice, ncols=nBin, figsize=(4 * nBin, 4 * nMice), tight_layout=True)

                    for iMouse, mousename in enumerate(sorted(dataDB.mice)):
                        channelNames = dataDB.get_channel_labels(mousename)
                        s1Idx = channelNames.index(s1Label)
                        s2Idx = channelNames.index(s2Label)
                        targetIdx = channelNames.index(labelTrg)

                        dataLst = dataDB.get_neuro_data({'mousename': mousename},
                                                        datatype=datatype, intervName=intervName, **kwargs)

                        dataRSP = np.concatenate(dataLst, axis=0)  # Concatenate all sessions
                        dataRP = np.mean(dataRSP, axis=1)  # Average out time

                        if nDropPCA is not None:
                            dataRP = drop_PCA(dataRP, nDropPCA)

                        dataBin = bin_data(dataRP, nBin, axis=1)  # Binarize data over channels

                        h3d = np.histogramdd(dataBin[:, [targetIdx, s1Idx, s2Idx]], bins=(nBin, nBin, nBin))[0]
                        h3d /= np.sum(h3d)  # Normalize

                        for iTrgBin in range(nBin):
                            img = ax[iMouse][iTrgBin].imshow(h3d[iTrgBin], vmin=0, vmax=10 / nBin ** 3, cmap='jet')
                            ax[iMouse][iTrgBin].set_ylabel(s1Label)  # First label is rows a.k.a Y-AXIS!!!
                            ax[iMouse][iTrgBin].set_xlabel(s2Label)
                            ax[iMouse][iTrgBin].set_title(labelTrg + '=' + str(iTrgBin))
                            imshow_add_color_bar(fig, ax[iMouse][iTrgBin], img)

                    plt.savefig('pics/info3D_' + dataLabel + '_'  + s1Label + '_' + s2Label + '_' + labelTrg + '.pdf')
                    plt.close(fig)
