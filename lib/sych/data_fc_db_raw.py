import os
import numpy as np
import h5py

# IPython-Specific

# Mesostat includes
from mesostat.utils.system import getfiles_walk
from mesostat.utils.signals.resample import zscore_dim_ord

# Local libraries


class DataFCDatabase:
    def __init__(self, param):

        # Find and parse Data filenames
        self.mice = set()
        self.dataPathsDict = {}

        ##################################
        # Define resampling frequency
        ##################################
        self.tMin = -2           # Seconds, start of trial
        self.tMax = 8            # Seconds, end of trial
        self.targetFPS = 20      # Hz
        self.targetLength = int((self.tMax - self.tMin)*self.targetFPS)   # Trial length in timesteps
        self.times = np.arange(self.tMin, self.tMax, 1/self.targetFPS)

        ##################################
        # Find and parse data files
        ##################################
        print("Searching for data files")
        self._find_parse_neuro_files(param["root_path_data"])

        print("Extracting trial type names")
        self._extract_trial_type_names()

        print("Extracting data types")
        self._extract_data_types()

    def _find_parse_neuro_files(self, path):
        paths, names = getfiles_walk(path, ['raw', '.h5']).T
        paths = [os.path.join(path, fname) for path, fname in zip(paths, names)]
        mice = [os.path.splitext(f)[0][4:] for f in names]

        self.dataPathsDict = dict(zip(mice, paths))
        self.mice = set(mice)

    def _extract_trial_type_names(self):
        self.trialTypeNames = set()
        for mousename in self.mice:
            with h5py.File(self.dataPathsDict[mousename], 'r') as h5file:
                self.trialTypeNames.update(self._get_trial_type_names_h5(h5file))

    def _extract_data_types(self):
        self.dataTypes = set()
        for mousename in self.mice:
            with h5py.File(self.dataPathsDict[mousename], 'r') as h5file:
                self.dataTypes.update(self._get_data_types_h5(h5file))

    def _selector_to_mousename(self, selector):
        return selector['mousename'] if 'mousename' in selector else selector['session'][:5]

    # Extract sorted session names given H5 file
    def _get_sessions_h5(self, h5file, datatype='raw', performance=None, expertThr=0.7):
        dataKey = 'data_' + datatype
        sessions = sorted(list(h5file[dataKey].keys()))

        if performance is None:
            return sessions
        else:
            performances = np.array([np.array(h5file['performance'][session]) for session in sessions])
            expertIdxs = np.where(performances >= expertThr)[0]

            if performance == 'expert':
                thisIdxs = expertIdxs
            elif performance == 'naive':
                thisIdxs = set(range(len(sessions))) - set(expertIdxs)
            else:
                raise ValueError("Unexpected performance", performance)

            return [s for i,s in enumerate(sessions) if i in thisIdxs]

    # Extract trial type names given H5 file
    def _get_trial_type_names_h5(self, h5file):
        return [t.decode("utf-8") for t in h5file['trialTypeNames']]

    def _get_data_types_h5(self, h5file):
        return [key[5:] for key in h5file.keys() if 'data_' in key]

    def get_trial_type_names(self):
        return self.trialTypeNames

    def get_data_types(self):
        return self.dataTypes

    # For given mouse file and session, get trial indices corresponding to trials of indicated type
    def get_trial_type_idxs_h5(self, h5file, session, trialType):
        typeNames = self._get_trial_type_names_h5(h5file)
        types = np.array(h5file['trialTypesSelected'][session])

        # Extract necessary trials or all
        if trialType not in typeNames:
            raise ValueError("Unexpected trial type", trialType, "must have", typeNames)

        trialTypeIdx = typeNames.index(trialType)
        return np.where(types == trialTypeIdx)[0]

    def get_channel_labels(self, mousename):
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            return [l.decode('UTF8') for l in h5file['channelLabels']]

    def get_times(self):
        return self.times

    def get_nchannels(self, mousename):
        return len(self.get_channel_labels(mousename))

    def get_ntrial_bytype(self, selector, trialType=None, performance=None):
        rezLst = []

        mousename = self._selector_to_mousename(selector)
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            sessions = selector['session'] if 'session' in selector else self._get_sessions_h5(h5file, performance=performance)
            for session in sessions:
                if trialType is None:
                    rezLst += [len(h5file['trialTypesSelected'][session])]
                else:
                    rezLst += [len(self.get_trial_type_idxs_h5(h5file, session, trialType))]

        return np.sum(rezLst)

    def get_sessions(self, mousename, datatype='raw', performance=None):
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            return self._get_sessions_h5(h5file, datatype=datatype, performance=performance)

    def get_performance(self, session):
        mousename = self._selector_to_mousename({'session' : session})
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            return float(np.array(h5file['performance'][session]))

    def get_expert_session_idxs(self, mousename, expertThr=0.7):
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            sessions = self._get_sessions_h5(h5file)
            performances = np.array([np.array(h5file['performance'][session]) for session in sessions])
            return np.where(performances >= expertThr)[0]

    def get_first_expert_session_idx(self, mousename, expertThr=0.7):
            expertIdxs = self.get_expert_session_idxs(mousename, expertThr=expertThr)
            if len(expertIdxs) == 0:
                return None
            else:
                return np.min(expertIdxs)

    def cropRSP(self, dataRSP, startTime, endTime):
        assert dataRSP.shape[1] == self.targetLength

        # startIdx = int(startTime * self.targetFPS)
        # endIdx = int(endTime * self.targetFPS)
        # return dataRSP[:, startIdx:endIdx]

        idxs = np.logical_and(self.times >= startTime, self.times < endTime)
        return dataRSP[:, idxs]

    def get_neuro_data(self, selector, datatype='raw', zscoreDim=None, cropTime=None, trialType=None, performance=None):
        dataKey = 'data_' + datatype

        dataLst = []

        mousename = self._selector_to_mousename(selector)
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            sessions = [selector['session']] if 'session' in selector else self._get_sessions_h5(h5file, datatype=datatype, performance=performance)

            for sessionThis in sessions:
                dataRSP = np.array(h5file[dataKey][sessionThis])

                # Apply ZScoring if requested
                # VERY IMPORTANT: ZScoring must apply before trial type selection and cropping, it is a function of the whole dataset
                dataRSP = zscore_dim_ord(dataRSP, 'rsp', zscoreDim)

                if trialType is not None:
                    thisTrialTypeIdxs = self.get_trial_type_idxs_h5(h5file, sessionThis, trialType)
                    dataRSP = dataRSP[thisTrialTypeIdxs]
                if cropTime is not None:
                    dataRSP = self.cropRSP(dataRSP, *cropTime)

                dataLst += [dataRSP]

        return dataLst

    def get_data_raw(self, session):
        mousename = self._selector_to_mousename({"session": session})
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            data = np.copy(h5file['data'][session])
            trialIdxs = np.array(h5file['trialStartIdxs'][session])
            interTrialStartIdxs = np.array(h5file['interTrialStartIdxs'][session])
            fps = h5file['data'][session].attrs['FPS']
            trialTypes = np.array(h5file['trialTypes'][session])
            trialTypeNames = np.array([bb.decode('UTF8') for bb in h5file['trialTypeNames']])
            trialTypes = trialTypeNames[trialTypes]
            return data, trialIdxs, interTrialStartIdxs, fps, trialTypes
