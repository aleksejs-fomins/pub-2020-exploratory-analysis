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
from mesostat.utils.signals import downsample_int

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
        paths, names = getfiles_walk(path, ['raw', '.h5'])
        paths = [os.path.join(path, fname) for path, fname in zip(paths, names)]
        mice = [os.path.splitext(f)[0][4:] for f in names]

        self.dataPathsDict = dict(zip(mice, paths))
        self.mice = set(mice)

    # Mark days as naive or expert based on performance threshold
    def mark_days_expert_naive(self, pTHR):
        nNeuroFiles = self.metaDataFrames['neuro'].shape[0]
        self.expertThrIdx = {}  # Which session counting alphabetically is first expert for a given mouse
        isExpert = np.zeros(nNeuroFiles, dtype=bool)
        deltaDays = np.zeros(nNeuroFiles)
        deltaDaysCentered = np.zeros(nNeuroFiles)

        # For each mouse, determine which sessions are naive and which expert
        # Also determine number of days passed since start and since expert
        for mousename in self.mice:
            thisMouseMetadata = get_rows_colval(self.metaDataFrames['neuro'], 'mousename', mousename)
            thisMouseDataIdxs = np.array(thisMouseMetadata["date"].index)
            perf = self.dataPerformance[thisMouseDataIdxs]
            skillRez = mouse_performance_allsessions(list(thisMouseMetadata["date"]), perf, pTHR)
            self.expertThrIdx[mousename], isExpert[thisMouseDataIdxs], deltaDays[thisMouseDataIdxs], deltaDaysCentered[
                thisMouseDataIdxs] = skillRez

        # Add these values to metadata
        self.metaDataFrames['neuro']['isExpert'] = isExpert
        self.metaDataFrames['neuro']['deltaDays'] = deltaDays
        self.metaDataFrames['neuro']['deltaDaysCentered'] = deltaDaysCentered

    def get_channel_labels(self, mousename):
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            return h5file['channelLabels']

    def get_nchannels(self, mousename):
        return len(self.get_channel_labels(mousename))

    def get_sessions(self, mousename):
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            return list(h5file['data'].keys())

    def _raw_extract_data_trials(self, h5file, session):
        FPS = h5file['data'][session].attrs['FPS']
        data = np.copy(h5file['data'][session])
        starts = h5file['trialStartIdxs'][session]
        intervs = h5file['interTrialStartIdxs'][session]

        dataRSP = []

        for l, r in zip(starts, intervs[1:]):
            dataTrial = data[l:r]
            if FPS != self.targetFPS:
                t = np.arange(len(dataTrial)) / FPS
                t2, dataTrial = downsample_int(t, dataTrial, FPS // self.targetFPS)
            dataRSP += [dataTrial[:self.targetLength]]  # Crop to standard length here

        return np.array(dataRSP)

    def _raw_extract_data_trial_itervals(self, h5file, session):
        FPS = h5file['data'][session].attrs['FPS']
        data = np.copy(h5file['data'][session])
        starts = h5file['trialStartIdxs'][session]
        intervs = h5file['interTrialStartIdxs'][session]

        dataLst = []
        for l, r in zip(intervs[:-1], starts):
            dataTrial = data[l:r]
            if FPS != self.targetFPS:
                t = np.arange(len(dataTrial)) / FPS
                t2, dataTrial = downsample_int(t, dataTrial, FPS // self.targetFPS)
            dataLst += [dataTrial[:self.targetLength]]  # Crop to standard length here

        return dataLst

    def _extract_trial_type(self, h5file, session, trialType, dataRSP):
        typeNames = [t.decode("utf-8") for t in h5file['trialTypeNames']]
        types = h5file['trialTypes'][session]

        # Extract necessary trials or all
        if trialType not in typeNames:
            raise ValueError("Unexpected trial type", trialType, "must have", typeNames)

        trialTypeIdx = typeNames.index(trialType)
        trialsMask = np.array(types) == trialTypeIdx
        return dataRSP[trialsMask]

    def get_neuro_data(self, mousename, session=None, trialType=None, performance=None):
        dataLst = []

        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            sessions = [session] if session is not None else list(h5file['data'].keys())

            for sessionThis in sessions:
                dataRSP = self._raw_extract_data_trials(h5file, sessionThis)

                if trialType is not None:
                    dataRSP = self._extract_trial_type(h5file, session, trialType, dataRSP)

                dataLst += [dataRSP]

        return dataLst

    def get_neuro_data_intervals(self, mousename, session=None, trialType=None, performance=None):
        dataLst = []

        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            sessions = [session] if session is not None else list(h5file['data'].keys())

            for sessionThis in sessions:
                dataRSP = self._raw_extract_data_trials(h5file, sessionThis)

                if trialType is not None:
                    dataRSP = self._extract_trial_type(h5file, session, trialType, dataRSP)

                dataLst += [dataRSP]

        return dataLst