import numpy as np
from mesostat.utils.signals.resample import resample_kernel_same_interv


def resample_delay(dataRSP, tStartDelay, tStopDelay, tTrgDelay=2.0, tTrgRew=1.0, FPS=20.0, padReward=False):
    '''
    :param dataRSP:          data
    :param tStartDelay:      start time of delay (seconds)
    :param tStopDelay:       stop time of delay (seconds)
    :param tTrgDelay:        target duration of delay (seconds) - will be resampled to this number
    :param tTrgRew:          target duration of reward (seconds) - will be cropped/padded to this number
    :param FPS:              FPS of original recording
    :param padReward:        If yes, reward is padded with NAN if it is too short. If no, results may be different size if reward is short
    :return:
    '''
    iStart = int(tStartDelay * FPS)
    iStop = int(tStopDelay * FPS)
    nDelTrg = int(tTrgDelay * FPS)
    nRewTrg = int(tTrgRew * FPS)

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


def get_data_list(dataDB, haveDelay, mousename, tTrgDelay=2.0, tTrgRew=1.0, **kwargs):
    if not haveDelay:
        dataRSPLst = dataDB.get_neuro_data({'mousename': mousename}, **kwargs)
    else:
        dataRSPLst = []
        for session in dataDB.get_sessions(mousename):
            dataRSP = dataDB.get_neuro_data({'session': session}, **kwargs)[0]
            delayStart = dataDB.get_interval_times(session, mousename, 'DEL')[0][0]
            delayEnd = delayStart + dataDB.get_delay_length(mousename, session)
            dataRSPLst += [resample_delay(dataRSP, delayStart, delayEnd, tTrgDelay=tTrgDelay, tTrgRew=tTrgRew,
                                          FPS=dataDB.targetFreq, padReward=True)]

    return dataRSPLst
