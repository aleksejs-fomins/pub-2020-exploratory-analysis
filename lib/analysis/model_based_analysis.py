import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

from mesostat.utils.arrays import numpy_merge_dimensions

from lib.preprocessing.polyfit import poly_fit_transform


###############################
#  Preprocessing
###############################

def optogen_trial_idxs(data, trialStartIdxs, fps, tTrial=8.0, debugplot=False):
    nChannel = data.shape[1]
    postSh = trialStartIdxs + int(tTrial * fps)

    # Account for last trial too short
    if postSh[-1] > len(data):
        dataRSP = np.array([data[l:r, :nChannel] for l, r in zip(trialStartIdxs, postSh)][:-1])
    else:
        dataRSP = np.array([data[l:r, :nChannel] for l, r in zip(trialStartIdxs, postSh)])

    dataRP = np.linalg.norm(dataRSP, axis=1)
    if debugplot:
        for i in range(nChannel):
            dataRP[:, i] /= np.mean(dataRP[:, i])

        df = pd.DataFrame(dataRP, columns=np.arange(nChannel))
        plt.figure()
        sns.violinplot(data=df, inner='point')
    dropIdxs = []
    for iCh in range(nChannel):
        dropIdxs += [dataRP[:, iCh] > 2 * np.median(dataRP[:, iCh])]

    rez = np.any(dropIdxs, axis=0)

    # Account for last trial too short
    if postSh[-1] > len(data):
        rez = np.hstack([rez, [True]])

    print("Found", np.sum(rez), 'of', len(rez), 'optogenetic/short trials')
    return rez


def set_trials_nan(data, trialIdxs, trialStartIdxs, fps, tTrial=8.0):
    dataNew = data.copy()
    nanTrialStartIdxs = trialStartIdxs[trialIdxs]
    nanTrialStopIdxs = nanTrialStartIdxs + int(tTrial * fps)
    for l, r in zip(nanTrialStartIdxs, nanTrialStopIdxs):
        dataNew[l:r] = np.nan
    return dataNew


def dff_poly(times, data, ord):
    dataFitted = np.zeros(data.shape)
    for iCh in range(data.shape[1]):
        dataFitted[:, iCh] = poly_fit_transform(times, data[:, iCh], ord)

    dataDFF = data / dataFitted - 1
    return dataFitted, dataDFF


def plot_fitted_data(times, data, dataFitted, dataDFF, iCh, labels):
    fig, ax = plt.subplots(ncols=2, figsize=(12, 3))
    ax[0].set_ylabel(str(iCh) + ' ' + labels[iCh])
    ax[0].plot(times, data[:, iCh])
    ax[0].plot(times, dataFitted[:, iCh])
    ax[1].plot(times, dataDFF[:, iCh])
    plt.show()


def get_trial_timestep_indices(trialStartIdxs, fps, tTrial=8.0):
    rez = []
    trShift = int(fps * tTrial)
    for startIdx in trialStartIdxs:
        rez += [np.arange(startIdx, startIdx + trShift)]
    return np.hstack(rez)


###############################
#  General stuff
###############################

def ridge_fit_predict(x, y, alpha=1.0):
    clf = Ridge(alpha=alpha)
    xEff = x if x.ndim == 2 else x[:, None]
    clf.fit(xEff, y)
    return clf.predict(xEff)


# Fit each channel independently of each other
def fit_predict_bychannel(xCh, yCh, alpha=1.0):
    nChannel = xCh.shape[1]
    rez = [ridge_fit_predict(xCh[:, iCh], yCh[:, iCh], alpha=alpha) for iCh in range(nChannel)]
    return np.array(rez).T


# Fit each channel using sources of all channels
def fit_predict_multivar_bychannel(xCh, yCh, alpha=1.0):
    nChannel = xCh.shape[1]
    xEff = xCh if xCh.ndim == 2 else numpy_merge_dimensions(xCh, 1, 3)
    rez = [ridge_fit_predict(xEff, yCh[:, iCh], alpha=alpha) for iCh in range(nChannel)]
    return np.array(rez).T


def rms(y, axis=None):
    return np.sqrt(np.mean(y**2, axis=axis))


def plot_rmse_bychannel(y, yHatDict, haveLog=False):
    plt.figure()
    L2y = rms(y, axis=0)
    for label, yHat in yHatDict.items():
        L2err = rms(y-yHat, axis=0)
        plt.plot(L2err / L2y, label=label)
    plt.xlabel('Channel')
    plt.ylabel('Relative RMSE')
    if haveLog:
        plt.yscale('log')
    plt.legend()
    plt.show()


###############################
#  Models
###############################

def get_src_ar1(data, idxsTrg):
    return data[idxsTrg-1]


def get_src_har(data, idxsTrg, harLagsLst):
    nTime = len(idxsTrg)
    nChannel = data.shape[1]
    nFeature = 1 + len(harLagsLst)
    dataSrc = np.zeros((nTime, nChannel, nFeature))
    dataSrc[:, :, 0] = data[idxsTrg - 1]   # Always include AR(1) term

    for iLag, lag in enumerate(harLagsLst):
        for iTime, idxTime in enumerate(idxsTrg):
            idxTimeMin = max(0, idxTime-lag)
            dataSrc[iTime, :, iLag+1] = np.nanmean(data[idxTimeMin:idxTime], axis=0)

    return dataSrc

