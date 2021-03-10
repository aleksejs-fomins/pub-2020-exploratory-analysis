import numpy as np

from mesostat.utils.arrays import slice_sorted
from mesostat.utils.signals.resample import resample, resample_kernel


def resample_lick(f_lick, neuro, behaviour, TARGET_TIMES, TARGET_FREQ):
    param_resample = {'method' : 'averaging', 'kind' : 'kernel', 'ker_sig2' : (2.0/TARGET_FREQ)**2}
    l,r = slice_sorted(f_lick['tLicks'], [0, 8])
    tLicksTrunc = f_lick['tLicks'][l:r]

    #nTrials = f_lick['reward_time'].shape[0]
    nTrialsNeuro = neuro.shape[0]
    
    TARGET_NTIMES = len(TARGET_TIMES)
    wResample = resample_kernel(tLicksTrunc, TARGET_TIMES, param_resample['ker_sig2'])

    data_lick = np.zeros((nTrialsNeuro, TARGET_NTIMES))
    keymap_lick = {
        'iGO'    : 'licks_go',
        'iNOGO'  : 'licks_nogo',
        'iFA'    : 'licks_FA',
        'iMISS'  : 'licks_miss',
        'iEARLY' : 'licks_early'}
    
    # Fix the problem that lickdata is not guaranteed to be a 2D array. It can be an empty array, or a 1D array if NTRIALS=1
    def inferNTrialsFromLick(lickdata):
        if len(lickdata) == 0:
            return 0
        elif len(lickdata.shape) == 1:
            return 1
        else:
            return lickdata.shape[1]
        
    # Fix the problem that behaviour is sometimes just a signle integer, not array
    def behavToArray(b):
        return np.array([b], dtype=int) if type(b)==int else b
    
    # Consistency TEST_1
    nTrialsReaction = len(f_lick['reaction_time'])
    nTrialsReward = len(f_lick['reward_time'])
    nTrialsPuff = len(f_lick['puff'])
    nTrialsTypeLick = [inferNTrialsFromLick(f_lick[v]) for v in keymap_lick.values()]
    nTrialsTypeLickTot = np.sum(nTrialsTypeLick)
    if not ((nTrialsReaction == nTrialsReward) and (nTrialsReaction==nTrialsPuff) and (nTrialsReaction==nTrialsTypeLickTot)):
        # raise ValueError("Trial count inside file inconsistent", nTrialsReaction, nTrialsReward, nTrialsPuff, nTrialsTypeLickTot)
        print("Trial count inside file inconsistent", nTrialsReaction, nTrialsReward, nTrialsPuff, nTrialsTypeLickTot)
        
    # Consistency TEST_2
    nTrialsTypeBeh = [len(behavToArray(behaviour[k])) for k in keymap_lick.keys() if k in behaviour.keys()]
    if nTrialsTypeLickTot != nTrialsNeuro:
        # raise ValueError("Lick trial count inconsistent with neuronal", nTrialsTypeLickTot, nTrialsNeuro)
        print("Lick trial count inconsistent with neuronal", nTrialsTypeLickTot, nTrialsNeuro)
        print("Inferred from behaviour", nTrialsTypeBeh)
        print("Inferred from lick     ", nTrialsTypeLick)
    
    for k, nTrBeh, nTrLick in zip(keymap_lick.keys(), nTrialsTypeBeh, nTrialsTypeLick):
        if nTrBeh != nTrLick:
            # raise ValueError("Lick trial count for "+k+" inconsistent with neuronal", nTrialsTypeBeh, nTrialsTypeLick)
            print("Lick trial count for "+k+" inconsistent with neuronal", nTrBeh, nTrLick)
            print("Inferred from behaviour", nTrialsTypeBeh)
            print("Inferred from lick     ", nTrialsTypeLick)
    
    for k, v in keymap_lick.items():
        if (k != "iEARLY") and (len(f_lick[v]) > 0):
            # Truncate data
            lick_trunc = f_lick[v][l:r]
            
            # Fix the problem when in case of only 1 trial it is a 1D array
            if len(lick_trunc.shape)==1:
                lick_trunc = lick_trunc.reshape(len(lick_trunc), 1)

            # Resample and stitch all trials back together into one 2D matrix
            # Note MATLAB index in behaviour is +1
            for iSubTrial, iTrialMATLAB in enumerate(behavToArray(behaviour[k])):
                if iTrialMATLAB-1 < nTrialsNeuro:
                    data_lick[iTrialMATLAB - 1] = wResample.dot(lick_trunc[:, iSubTrial])
                    #data_lick[iTrialMATLAB-1] = resample(tLicksTrunc, lick_trunc[:, iSubTrial], TARGET_TIMES, param_resample)
    return data_lick


def resample_paw(f_paw, TARGET_TIMES, TARGET_FREQ):
    # Estimate baseline
    #mle = np.nanmean(f_paw['trialsPaw'])
    hist_v, hist_t = np.histogram(f_paw['trialsPaw'].flatten(), bins='auto')
    mle = hist_t[np.argmax(hist_v)]
    #print("Baseline estimated to be at", mle)

    # 1) Subtract baseline and rescale
    data_paw = f_paw['trialsPaw'] - mle
    data_paw[np.isnan(data_paw)] = 0
    #data_paw /= np.max(data_paw)

    # Truncate
    l,r = slice_sorted(f_paw['tPaw'], [0, 8])
    paw_times_tmp = f_paw['tPaw'][l:r]
    data_paw = data_paw[:, l:r]

    # Resample
    nTrials = data_paw.shape[0]

    if f_paw['freqPaw'] < TARGET_FREQ:
        param_resample = {'method' : 'interpolative', 'kind' : 'cubic'}
        data_paw = np.array([resample(paw_times_tmp, d, TARGET_TIMES, param_resample) for d in data_paw])
    else:
        param_resample = {'method' : 'averaging', 'kind' : 'kernel', 'ker_sig2' : (0.5/TARGET_FREQ)**2}
        wResample = resample_kernel(paw_times_tmp, TARGET_TIMES, param_resample['ker_sig2'])
        data_paw = data_paw.dot(wResample.T)
        #data_paw = np.array([resample(paw_times_tmp, data_paw[iTrial], TARGET_TIMES, param_resample) for iTrial in range(nTrials)])

    # Normalize
    data_paw /= np.max(data_paw)
    
    return data_paw


def resample_whisk(f_whisk, TARGET_TIMES):
    # 1) Subtract baseline
    nTrials = f_whisk['whiskAngle'].shape[1]
    f_whisk['whiskAngle'] -= np.nanmean(f_whisk['whiskAngle'])
    f_whisk['whiskAngle'][np.isnan(f_whisk['whiskAngle'])] = 0
    f_whisk['whiskAbsVelocity'] = np.vstack((np.abs(f_whisk['whiskAngle'][1:] - f_whisk['whiskAngle'][:-1]), np.zeros(nTrials)))

    # Truncate
    l,r = slice_sorted(f_whisk['tWhisk'], [0, 8])
    f_whisk['tWhisk']     = f_whisk['tWhisk'][l:r]
    f_whisk['whiskAngle'] = f_whisk['whiskAngle'][l:r]
    f_whisk['whiskAbsVelocity'] = f_whisk['whiskAbsVelocity'][l:r]

    # Resample
    param_resample = {'method' : 'averaging', 'kind' : 'kernel', 'ker_sig2' : (4/200)**2}
    wResample = resample_kernel(f_whisk['tWhisk'], TARGET_TIMES, param_resample['ker_sig2'])
    f_whisk['whiskAbsVelocity'] = wResample.dot(f_whisk['whiskAbsVelocity']).T
    #f_whisk['whiskAbsVelocity'] = np.array([resample(f_whisk['tWhisk'], f_whisk['whiskAbsVelocity'][:, iTrial], TARGET_TIMES, param_resample) for iTrial in range(nTrials)])

    return f_whisk['whiskAbsVelocity']
