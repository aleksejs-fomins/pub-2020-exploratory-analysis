import numpy as np
from mesostat.utils.signals.resample import resample_kernel_same_interv


def _resample_delay(dataRSP, tStart, tStop, FPS=20.0, padReward=False):
    iStart = int(tStart * FPS)
    iStop = int(tStop * FPS)
    nDelTrg = int(2.0 * FPS)
    nRewTrg = int(1.0 * FPS)

    dataPRE = dataRSP[:, :iStart]
    dataDEL = dataRSP[:, iStart:iStop]
    dataREW = dataRSP[:, iStop:]

    nDel = dataDEL.shape[1]
    nRew = dataREW.shape[1]

    # Resample delay to 2s
    W = resample_kernel_same_interv(nDel, nDelTrg)
    dataDEL = np.einsum('lj,ijk->ilk', W, dataDEL)

    if padReward:
        if nRew > nRewTrg:
            # Crop reward to 1 second if it exceeds
            dataREW = dataREW[:, :nRewTrg]
        elif nRew < nRewTrg:
            # Pad reward with NAN if it is too short
            tmp = np.full((dataREW.shape[0], nRewTrg, dataREW.shape[2]), np.nan)
            tmp[:, :nRew] = dataREW
            dataREW = tmp

    return np.concatenate([dataPRE, dataDEL, dataREW], axis=1)


def get_data_list(dataDB, haveDelay, mousename, **kwargs):
    if not haveDelay:
        dataRSPLst = dataDB.get_neuro_data({'mousename': mousename}, **kwargs)
    else:
        dataRSPLst = []
        for session in dataDB.get_sessions(mousename):
            dataRSP = dataDB.get_neuro_data({'session': session}, **kwargs)[0]
            delayStart = dataDB.get_interval_times(session, mousename, 'DEL')[0][0]
            delayEnd = delayStart + dataDB.get_delay_length(mousename, session)
            dataRSPLst += [_resample_delay(dataRSP, delayStart, delayEnd, FPS=dataDB.targetFreq, padReward=True)]

    return dataRSPLst