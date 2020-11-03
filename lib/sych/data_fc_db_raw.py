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

    def get_channel_labels(self, mousename):
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            return [l.decode('UTF8') for l in h5file['channelLabels']]

    def get_nchannels(self, mousename):
        return len(self.get_channel_labels(mousename))

    def get_sessions(self, mousename, datatype='raw'):
        dataKey = 'data_' + datatype
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            return list(h5file[dataKey].keys())

    def get_data_types(self, mousename):
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            return [key[5:] for key in h5file.keys() if 'data_' in key]

    def _get_trial_type_names(self, h5file):
        return [t.decode("utf-8") for t in h5file['trialTypeNames']]

    def _extract_trial_type(self, h5file, session, trialType, dataRSP):
        typeNames = self._get_trial_type_names(h5file)
        types = np.array(h5file['trialTypesSelected'][session])

        print(session, dataRSP.shape, len(types))

        # Extract necessary trials or all
        if trialType not in typeNames:
            raise ValueError("Unexpected trial type", trialType, "must have", typeNames)

        trialTypeIdx = typeNames.index(trialType)
        return dataRSP[types == trialTypeIdx]

    def get_neuro_data(self, mousename, datatype='raw', session=None, trialType=None, performance=None):
        dataKey = 'data_' + datatype

        dataLst = []

        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            sessions = [session] if session is not None else list(h5file[dataKey].keys())

            for sessionThis in sessions:
                dataRSP = np.array(h5file[dataKey][sessionThis])

                if trialType is not None:
                    dataRSP = self._extract_trial_type(h5file, sessionThis, trialType, dataRSP)

                dataLst += [dataRSP]

        return dataLst