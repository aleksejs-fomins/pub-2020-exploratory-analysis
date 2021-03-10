import numpy as np
import matplotlib.pyplot as plt

from mesostat.utils.signals.resample import zscore
import mesostat.stat.consistency.pca as pca


def plot_pca_consistency(dataDB, intervDict, dropFirst=None):
    mice = sorted(dataDB.mice)
    nMice = len(mice)

    for datatype in dataDB.get_data_types():
        for intervName, interv in intervDict.items():
            fnameSuffix = datatype + '_' + intervName
            print(fnameSuffix)

            dataLst = []

            for iMouse, mousename in enumerate(mice):
                dataRSPLst = dataDB.get_neuro_data({'mousename': mousename}, datatype=datatype,
                                                   cropTime=interv)
                dataRSP = np.concatenate(dataRSPLst, axis=0)
                dataRP = np.mean(dataRSP, axis=1)
                dataRP = zscore(dataRP, axis=0)

                dataLst += [dataRP]

            fig, ax = plt.subplots(nrows=nMice, ncols=nMice, figsize=(4 * nMice, 4 * nMice))

            rezMat = np.zeros((nMice, nMice))
            for iMouse in range(nMice):
                for jMouse in range(nMice):
                    rezXY, rezYX, arrXX, arrXY, arrYY, arrYX = pca.paired_comparison(
                        dataLst[iMouse], dataLst[jMouse], dropFirst=dropFirst)
                    rezMat[iMouse][jMouse] = rezXY # np.mean([rezXY, rezYX])

                    ax[iMouse][jMouse].plot(arrXX, label='XX')
                    ax[iMouse][jMouse].plot(arrXY, label='XY')
                    ax[iMouse][jMouse].legend()

            plt.savefig('pca_consistency_bymouse_evals_' + fnameSuffix + '.png')
            plt.close()

            plt.figure()
            plt.imshow(rezMat, vmin=0, vmax=1)
            plt.colorbar()
            plt.savefig('pca_consistency_bymouse_metric_' + fnameSuffix + '.png')
            plt.close()
