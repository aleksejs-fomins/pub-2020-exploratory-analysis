import h5py
import os
import numpy as np

from mesostat.utils.matlab_helper import loadmat, matstruct2dict
from mesostat.utils.dictionaries import merge_dicts

from lib.sych.mouse_performance import mouse_performance_single_session


def session_name_to_mousename(mousekey):
    return '_'.join(mousekey.split('_')[:2])


# Read data and behaviour matlab files given containing folder
def read_neuro_perf(folderpath, verbose=True, withPerformance=True):
    # Read MAT file from command line
    if verbose:
        print("Reading Yaro data from", folderpath)
    fname_data        = os.path.join(folderpath, "data.mat")
    fname_behaviour   = os.path.join(folderpath, "behaviorvar.mat")
    fpath_performance = os.path.join(folderpath, "Transfer Entropy")
    
    waitRetry = 3  # Seconds wait before trying to reload the file if it is not accessible
    data        = loadmat(fname_data, waitRetry=waitRetry)['data']
    nTrialsData = data.shape[0]
    behavior    = loadmat(fname_behaviour, waitRetry=waitRetry)
    behavior = {k : v for k, v in behavior.items() if k[0] != '_'}      # Get rid of useless fields in behaviour

    if withPerformance:
        if (nTrialsData < len(behavior['iGO']) + len(behavior['iNOGO'])):
            print("Warning: For", os.path.basename(folderpath), "nTrials inconsistent with behaviour", nTrialsData, len(behavior['iGO']), len(behavior['iNOGO']))

        performance = mouse_performance_single_session(nTrialsData, behavior)

        if not os.path.exists(fpath_performance):
            print("--Warning: No performance metrics found for", os.path.dirname(folderpath), "; Using calculated")
        else:
            fname_performance = os.path.join(fpath_performance, "performance.mat")
            performanceExt = loadmat(fname_performance, waitRetry=waitRetry)['performance']
            if performanceExt != performance:
                print("Calculated performance", performance, "does not match external", performanceExt)
    
    # Convert trials structure to a dictionary
    behavior['trials'] = merge_dicts([matstruct2dict(obj) for obj in behavior['trials']])
    
    # d_trials = matstruct2dict(behavior['trials'][0])
    # for i in range(1, len(behavior['trials'])):
    #     d_trials = merge_dict(d_trials, matstruct2dict(behavior['trials'][i]))
    # behavior['trials'] = d_trials
    
    # CONSISTENCY TEST 1 - If behavioural trials are more than neuronal, crop:
    behavToArray = lambda b: np.array([b], dtype=int) if type(b)==int else b   # If array has 1 index it appears as a number :(
    behKeys = ['iGO', 'iNOGO', 'iFA', 'iMISS']
    for behKey in behKeys:
        behArray = behavToArray(behavior[behKey])
        
        behNTrialsThis = len(behArray)
        if behNTrialsThis > 0:
            behMaxIdxThis  = np.max(behArray) - 1  # Note Matlab indices start from 1            
            if behMaxIdxThis >= nTrialsData:
                print("--Warning: For", behKey, "behaviour max index", behMaxIdxThis, "exceeds nTrials", nTrialsData)
                behavior[behKey] = behavior[behKey][behavior[behKey] < nTrialsData]
                print("---Cropped excessive behaviour trials from", behNTrialsThis, "to", len(behavior[behKey]))
            
    # CONSISTENCY TEST 2 - If neuronal trials are more than behavioural
    # Note that there are other keys except the above four, so data trials may be more than those for the four keys
    behIndices = [idx for key in behKeys for idx in behavToArray(behavior[key])]
    assert len(behIndices) <= nTrialsData, "After cropping behavioural trials may not exceed data trials"
    
    # CONSISTENCY TEST 3 - Test behavioural indices for duplicates
    for idx in behIndices:
        thisCount = behIndices.count(idx)
        if thisCount != 1:
            keysRep = {key : list(behavToArray(behavior[key])).count(idx) for key in behKeys if idx in behavToArray(behavior[key])}
            print("--WARNING: index", idx, "appears multiple times:", keysRep)
            
        #assert behIndices.count(idx) == 1, "Found duplicates in behaviour indices"

    if withPerformance:
        return data, behavior, performance
    else:
        return data, behavior


# # Read multiple neuro and performance files from a root folder
# def read_neuro_perf_multi(rootpath):
#
#     # Get all subfolders, mark them as mice
#     mice = get_subfolders(rootpath)
#     micedict = {}
#
#     # For each mouse, get all subfolders, mark them as days
#     for mouse in mice:
#         mousepath = os.path.join(rootpath, mouse)
#         days = get_subfolders(mousepath)
#         micedict[mouse] = {day : {} for day in days}
#
#         # For each day, read mat files
#         for day in days:
#             daypath = os.path.join(mousepath, day)
#             data, behaviour = read_neuro_perf(daypath)
#             micedict[mouse][day] = {
#                 'data' : data,
#                 'behaviour' : behaviour
#             }
#
#     return micedict


def read_lick(folderpath, verbose=True):
    if verbose:
        print("Processing lick folder", folderpath)
    
    rez = {}
    
    ################################
    # Process Reaction times file
    ################################
    rt_file = os.path.join(folderpath, "RT_264.mat")
    rt = loadmat(rt_file)
    
    rez['reaction_time'] = 3.0 + rt['reaction_time']
    
    ################################
    # Process lick_traces file
    ################################
    def lick_filter(data, bot_th, top_th):
        data[np.isnan(data)] = 0
        return np.logical_or(data <= bot_th, data >= top_th).astype(int)
    
    lick_traces_file = os.path.join(folderpath, "lick_traces.mat")
    lick_traces = loadmat(lick_traces_file)
    
    nTimesLick = len(lick_traces['licks_go'])
    freqLick = 100 # Hz
    rez['tLicks'] = np.arange(0, nTimesLick) / freqLick
    
    # Top threshold is wrong sometimes. Yaro said to use exact one
    thBot, thTop = lick_traces['bot_thresh'], 2.64
    
    for k in ['licks_go', 'licks_nogo', 'licks_miss', 'licks_FA', 'licks_early']:
        rez[k] = lick_filter(lick_traces[k], thBot, thTop)
        
    ################################
    # Process trials file
    ################################
    TIMESCALE_TRACES = 0.001 # ms
    trials_file = os.path.join(folderpath, os.path.basename(folderpath)+".mat")
    #print(trials_file)
    
    lick_trials = loadmat(trials_file)

    # NOTE: lick_trials['licks']['lick_vector'] is just a repeat from above lick_traces file
#     lick_trials['licks'] = merge_dicts([matstruct2dict(obj) for obj in lick_trials['licks']])
    lick_trials['trials'] = merge_dicts([matstruct2dict(obj) for obj in lick_trials['trials']])
    fixearly = lambda trial : np.nan if trial=='Early' else trial
    lick_trials['trials']['reward_time'] = [fixearly(trial) for trial in lick_trials['trials']['reward_time']]
    rez['reward_time'] = np.array(lick_trials['trials']['reward_time'], dtype=float) * TIMESCALE_TRACES
    rez['puff'] = [np.array(puff, dtype=float)*TIMESCALE_TRACES for puff in lick_trials['trials']['puff']]
        
    return rez


def read_paw(folderpath, verbose=True):
    if verbose:
        print("Processing paw folder", folderpath)
    
    filepath = os.path.join(folderpath, 'trials.mat')
    rezdict = {'trialsPaw' : loadmat(filepath)['trials']}
    
    nTrialsPaw, nTimesPaw = rezdict['trialsPaw'].shape
    if nTimesPaw == 64:
        freqPaw = 7
    elif nTimesPaw > 250:
        freqPaw = 30
    else:
        raise ValueError("Unexpected number of paw timesteps", nTimePaw)

    rezdict['tPaw'] = np.arange(0, nTimesPaw) / freqPaw
    rezdict['freqPaw'] = freqPaw
    return rezdict


def read_whisk(folderpath, verbose=True):
    if verbose:
        print("Processing whisk folder", folderpath)
    
    #############################
    # Read whisking angle
    #############################
    rezdict = {'whiskAngle' : loadmat(os.path.join(folderpath, 'whiskAngle.mat'))['whiskAngle']}
    nTimesWhisk, nTrialsWhisk = rezdict['whiskAngle'].shape
    if nTimesWhisk <= 900:
        freqWhisk = 40
    elif nTimesWhisk >= 1600:
        freqWhisk = 200
    else:
        freqWhisk = 40
        # raise ValueError("Unexpected number of whisk timesteps", nTimesWhisk)
        print("Unexpected number of whisk timesteps", nTimesWhisk)
    
    rezdict['tWhisk']           = np.arange(0, nTimesWhisk) / freqWhisk
    rezdict['whiskAbsVelocity'] = np.vstack((np.abs(rezdict['whiskAngle'][1:] - rezdict['whiskAngle'][:-1])*freqWhisk, np.zeros(nTrialsWhisk)))
        
    #############################
    # Read first touch
    #############################
    firstTouchFilePath = os.path.join(folderpath, os.path.basename(folderpath)+'.txt')
    if not os.path.isfile(firstTouchFilePath):
        print("Warning: first touch file does not exist", firstTouchFilePath)
        rezdict['firstTouch'] = None
    else:    
        with open(firstTouchFilePath) as fLog:
            rezdict['firstTouch'] = np.array([line.split('\t')[1] for line in fLog.readlines()[1:]], dtype=float)

    return rezdict

    
def read_lvm(filename, verbose=True):
    if verbose:
        print("Reading LVM file", filename, "... ")
    
    # Read file
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    # Figure out where the headers are
    header_endings = [i for i in range(len(lines)) if "End_of_Header" in lines[i]]
        
    # Read data after the last header ends
    idx_data_start = header_endings[-1] + 2
    data = np.array([line.strip().split('\t') for line in lines[idx_data_start:]]).astype(float)
    channel_idxs = data[:,0].astype(int)
    times = data[:,1]

    # Figure out how many channels there are
    min_channel = np.min(channel_idxs)
    max_channel = np.max(channel_idxs)
    nChannel = max_channel - min_channel + 1
    nData = len(channel_idxs)//nChannel

    # Partition data into 2D array indexed by channel
    data2D = np.zeros((nChannel, nData))
    for i in range(nChannel):
        data2D[i] = times[channel_idxs == i]
        
    print("... done! Data shape read ", data2D.shape)
    return data2D


# Extract TE from H5 file
def readTE_H5(fname, summary):
    # print("Reading file", fname)
    # filename = os.path.join(pwd_h5, os.path.join("real_data", fname))
    # h5f = h5py.File(filename, "r")
    h5f = h5py.File(fname, "r")
    data = np.array([
        np.copy(h5f['results']['TE_table']),
        np.copy(h5f['results']['delay_table']),
        np.copy(h5f['results']['p_table'])
    ])
    h5f.close()

    # Crop data based on delay and window

    # Based on max lag and averaging time window
    #  some initial and final time steps of the data
    #  are essentialy not computed. First verify that
    #  they are indeed not computed, then crop them
    N_TIMES = data.shape[-1]
    GAP_L = summary["max_lag"]
    GAP_R = summary["window"] - summary["max_lag"] - 1
    if (GAP_L <= 0) or (GAP_R < 0):
        raise ValueError("Incompatible window and maxlag values ", summary["window"], summary["max_lag"])

    conn_L = np.sum(1 - np.isnan(data[..., :GAP_L]))
    if conn_L > 0:
        raise ValueError("While maxlag is", GAP_L, "found", conn_L, "non-nan connections in the first", GAP_L,
                         "timesteps")

    if GAP_R > 0:
        conn_R = np.sum(1 - np.isnan(data[..., N_TIMES - GAP_R:]))
        if conn_R > 0:
            raise ValueError("While win-lag-1 is", GAP_R, "found", conn_R, "non-nan connections in the last", GAP_R,
                             "timesteps")

    # Compute effective sampling times
    times = summary["timestep"] * (np.arange(GAP_L, N_TIMES - GAP_R) + GAP_R / 2)

    return times, data[..., GAP_L:N_TIMES - GAP_R]


# Extract metric from folder name containing TE files
def parse_TE_folder(datapath):
    dataname = os.path.basename(datapath)
    downsampling, delayText, delay, windowText, window = dataname.split('_')
    assert downsampling in ["raw", "subsample", "subsampled"], "Can't infer downsampling from " + dataname
    assert delayText == "delay", "Unexpected data folder name " + dataname
    assert windowText == "window", "Unexpected data folder name " + dataname

    if downsampling == "raw":
        timestep = 0.05  # seconds
    else:
        timestep = 10 / 49  # seconds  (I resampled to 50, not to 51 points [whoops], so DT is not 0.2 but a bit more

    summaryTE = {
        "dataname": dataname,
        "downsampling": downsampling,
        "max_lag": int(delay),
        "window": int(window),
        "timestep": timestep
    }

    return summaryTE
