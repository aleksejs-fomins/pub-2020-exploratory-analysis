import os
import numpy as np
import h5py
import pandas as pd

# IPython-Specific
from IPython.display import display
from ipywidgets import IntProgress

# Mesostat includes
from mesostat.utils.pandas_helper import get_rows_colval, get_rows_colvals
from mesostat.utils.system import getfiles_walk

# Local libraries
from lib.sych.mouse_performance import mouse_performance_allsessions


class DataFCDatabase:
    def __init__(self, param):

        # Find and parse Data filenames
        self.mice = set()
        self.dataPathsDict = {}

        ##################################
        # Define resampling frequency
        ##################################
        self.targetFPS = 20  # Hz
        self.targetLength = 160  # Timesteps, at selected frequency

        ##################################
        # Find and parse data files
        ##################################
        print("Searching for data files")
        self._find_parse_neuro_files(param["root_path_data"])

    def _find_parse_neuro_files(self, path):
        paths, names = getfiles_walk(path, ['raw', '.h5']).T
        paths = [os.path.join(path, fname) for path, fname in zip(paths, names)]
        mice = [os.path.splitext(f)[0][4:] for f in names]

        self.dataPathsDict = dict(zip(mice, paths))
        self.mice = set(mice)

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

    def get_trial_type_names(self, mousename):
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            return self._get_trial_type_names_h5(h5file)

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

    def get_data_types(self, mousename):
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            return [key[5:] for key in h5file.keys() if 'data_' in key]

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

    def get_neuro_data(self, selector, datatype='raw', trialType=None, performance=None):
        dataKey = 'data_' + datatype

        dataLst = []

        mousename = self._selector_to_mousename(selector)
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            sessions = [selector['session']] if 'session' in selector else self._get_sessions_h5(h5file, datatype=datatype, performance=performance)

            for sessionThis in sessions:
                dataRSP = np.array(h5file[dataKey][sessionThis])

                if trialType is not None:
                    thisTrialTypeIdxs = self.get_trial_type_idxs_h5(h5file, sessionThis, trialType)
                    dataRSP = dataRSP[thisTrialTypeIdxs]

                dataLst += [dataRSP]

        return dataLst
