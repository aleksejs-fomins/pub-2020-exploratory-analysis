import os
from os.path import join, isfile, splitext
import h5py
import json
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt

# Mesostat
from mesostat.utils.matlab_helper import loadmat
from mesostat.utils.signals.filter import zscore_dim_ord
from mesostat.utils.strings import enum_nonunique
from mesostat.visualization.mpl_colors import base_colors_rgb
from mesostat.visualization.mpl_legend import plt_add_fake_legend

# Local


class DataFCDatabase:
    def __init__(self, param):
        # Find and parse Data filenames
        self.mice = set()
        self.sessions = {}

        self.dimOrdCanon = 'rsp'
        self.dataTypes = ['raw', 'bn_session', 'bn_trial']
        self.accExpert = 0.7
        self.localPath = os.path.dirname(os.path.abspath(__file__))
        self.dataPath = os.path.join(os.path.dirname(os.path.dirname(self.localPath)), 'data')

        ##################################
        # Define resampling frequency
        ##################################
        # self.targetTimesteps = [20, 95]      # Crop timeframe to important stuff
        # self.targetChannels = np.arange(25)  # Crop last two brain regions, because they disbehave (too many nans)
        self.targetFreq = 20  # Hz

        ##################################
        # Find and parse data files
        ##################################
        print("Reading channel label file")
        self._find_read_channel_labels(param["root_path_data"])

        print("Reading channel area file")
        self._find_read_channel_area_names(self.dataPath)

        print("Reading allen brain map")
        self._find_read_allen_map(param["root_path_data"])

        print("Reading task structure")
        self._find_read_task_structure(param["root_path_data"])

        print("Searching for data files")
        self._find_parse_neuro_files(param["root_path_data"])

    def _find_read_channel_labels(self, path):
        '''
            Channel labels are brain regions associated to each channel index
            The channel labels need not be consistent across mice, or even within one mouse
        '''
        labelFileName = join(path, "ROI_names.mat")

        if not isfile(labelFileName):
            raise ValueError("Can't find file", labelFileName)

        self.channelLabels = loadmat(labelFileName)['ROI_names']

        # Make array unique
        self.channelLabels = enum_nonunique(self.channelLabels)
        assert len(self.channelLabels) == len(set(self.channelLabels))

    def _find_read_channel_area_names(self, path):
        self.channelAreasDF = pd.read_csv(os.path.join(path, 'gallerosalas_area_names.csv'),
                                          delimiter=';', skipinitialspace=True)

    def _find_read_allen_map(self, path):
        '''
            Find the 2D mapping file of hemispheric cortical areas according to allen brain atlas
        '''
        labelFileName = join(path, "L_modified.mat")
        if not isfile(labelFileName):
            raise ValueError("Can't find file", labelFileName)

        self.allenMap = loadmat(labelFileName)['L']                                # 2D map of cortical regions
        self.allenIndices = sorted(list(set(self.allenMap.flatten())))[2:]         # Indices of regions. Drop First two
        self.allenCounts = [np.sum(self.allenMap == i) for i in self.allenIndices] # Number of pixels per region

    def _find_read_task_structure(self, path):
        taskFileName = join(path, 'task_structure.json')

        with open(taskFileName) as f:
            self.timestamps = json.load(f)['timestamps']
            self.timestamps = {float(k) : v for k,v in self.timestamps.items()}

    def _find_parse_neuro_files(self, path):
        files = os.listdir(path)
        files = [f for f in files if (splitext(f)[1] == '.h5') and (f[:3] == 'mou')]
        mice = [splitext(f)[0] for f in files]
        self.mice = set(mice)
        self.datapaths = {mouse : join(path, f) for mouse, f in zip(mice, files)}
        print("Found mice", mice)

        # Parse sessions
        for mousename in mice:
            with h5py.File(self.datapaths[mousename], 'r') as f:
                self.sessions[mousename] = list(f['raw'].keys())

    # Find the shortest distances between areas based on allen map
    def calc_shortest_distances(self):
        pointsBoundary = {}
        for i in self.allenIndices:
            if i > 1:
                points = np.array(np.where(self.allenMap == i))
                pointsBoundary[i] = points[:, ConvexHull(points.T).vertices].T

        nRegion = len(pointsBoundary)
        minDist = np.zeros((nRegion, nRegion))
        for i, iPoints in enumerate(pointsBoundary.values()):
            for j, jPoints in enumerate(pointsBoundary.values()):
                if i < j:
                    minDist[i][j] = 10000000.0
                    for p1 in iPoints:
                        for p2 in jPoints:
                            minDist[i][j] = np.min([minDist[i][j], np.linalg.norm(p1 - p2)])
                    minDist[j][i] = minDist[i][j]

        self.allenDist = minDist

    def get_data_types(self):
        return self.dataTypes

    def get_trial_types(self, session, mousename=None):
        if mousename is None:
            mousename = self.find_mouse_by_session(session)

        df = pd.read_hdf(self.datapaths[mousename], '/metadata/'+session)
        return np.array(df['trialType'])

    def get_trial_type_names(self):
        return ['Hit', 'Miss', 'CR', 'FA']

    def get_ntrial_bytype(self, selector, trialType=None):
        trialTypes = [trialType] if trialType is not None else self.get_trial_type_names()
        if 'mousename' in selector:
            mousename = selector['mousename']
            sessions = self.get_sessions(mousename)
        else:
            mousename = self.find_mouse_by_session(selector['session'])
            sessions = [selector['session']]

        rez = 0
        for session in sessions:
            trialTypesThis = self.get_trial_types(session, mousename)
            for tt in trialTypes:
                rez += np.sum(trialTypesThis == tt)
        return rez

    #FIXME: Why is there one more label than in allen map? Is our alignment correct?
    def get_channel_labels(self, mousename=None):
        return self.channelLabels[:27]

    def get_nchannels(self, mousename):
        return len(self.get_channel_labels(mousename))

    def get_sessions(self, mousename, datatype=None):
        return self.sessions[mousename]

    def get_performance(self, session, mousename=None, metric='accuracy'):
        if mousename is None:
            mousename = self.find_mouse_by_session(session)
        with h5py.File(self.datapaths[mousename], 'r') as h5file:
            return np.copy(h5file[metric][session])

    def get_performance_mouse(self, mousename, metric='accuracy'):
        return [self.get_performance(session, mousename, metric) for session in self.get_sessions(mousename)]

    # If window greater than 1 is provided, return timesteps of data sweeped with a moving window of that length
    def get_times(self, nTime=160, window=1):
        if window == 1:
            return np.arange(nTime) / self.targetFreq
        else:
            return (np.arange(nTime - window + 1) + (window - 1) / 2) / self.targetFreq

    def find_mouse_by_session(self, session):
        miceLst = list(self.mice)
        miceHave = [session in self.sessions[mousename] for mousename in miceLst]
        assert np.sum(miceHave) == 1
        return miceLst[np.where(miceHave)[0][0]]

    def get_neuro_data(self, selector, datatype='raw', zscoreDim=None, cropTime=None, trialType=None, performance=None):
        selectorType = next(iter(selector.keys()))
        selectorVal = selector[selectorType]

        if selectorType == 'mousename':
            mousename = selectorVal
            sessions = self.sessions[mousename]
        else:
            mousename = self.find_mouse_by_session(selectorVal)
            sessions = [selectorVal]


        dataLst = []
        for session in sessions:
            # If requested, only select sessions with Naive/Expert performance
            if performance is not None:
                p = self.get_performance(session, mousename, metric='accuracy')
                if (performance == 'Expert') and p < self.accExpert:
                    continue
                elif (performance == 'Naive') and p >= self.accExpert:
                    continue

            with h5py.File(self.datapaths[mousename], 'r') as h5file:
                dataRSP = np.copy(h5file[datatype][session])
            times = self.get_times(dataRSP.shape[1])

            # If requested, only select trials of particular type
            if trialType is not None:
                trialTypes = self.get_trial_types(session, mousename=mousename)
                dataRSP = dataRSP[trialTypes == trialType]

            # Apply ZScoring if requested
            # VERY IMPORTANT: ZScoring must apply before trial type selection and cropping, it is a function of the whole dataset
            dataRSP = zscore_dim_ord(dataRSP.copy(), self.dimOrdCanon, zscoreDim)

            if cropTime is not None:
                # FIXME: Time index hardcoded. Make sure it works for any dimOrdCanon
                timeIdxs = np.logical_and(times >= cropTime[0], times < cropTime[1])
                dataRSP = dataRSP[:, timeIdxs]
            else:
                # FIXME: Number of timesteps hardcoded. Perhaps there is a better way
                dataRSP = dataRSP[:, :160]  # Ensuring all trials that are too long are cropped to this time

            dataLst += [dataRSP]

        return dataLst

    def label_plot_timestamps(self, ax, linecolor='y', textcolor='k', shX=-0.5, shY=0.05):
        # the x coords of this transformation are data, and the y coord are axes
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        for t, label in self.timestamps.items():
            ax.axvline(x=t, color=linecolor, linestyle='--')
            plt.text(t+shX, shY, label, color=textcolor, verticalalignment='bottom', transform=trans, rotation=90)

    def plot_area_lists(self, ax, regDict, haveLegend=False):
        trgShape = self.allenMap.shape + (3,)
        colors = base_colors_rgb('tableau')
        rez = np.zeros(trgShape)

        imBinary = self.allenMap == 0
        imColor = np.outer(imBinary.astype(float), np.array([0.5, 0.5, 0.5])).reshape(trgShape)
        rez += imColor

        for iGroup, (label, lst) in enumerate(regDict.items()):
            for iROI in lst:
                imBinary = self.allenMap == self.allenIndices[iROI]
                imColor = np.outer(imBinary.astype(float), colors[iGroup]).reshape(trgShape)
                rez += imColor

        if haveLegend:
            plt_add_fake_legend(ax, colors[:len(regDict)], list(regDict.keys()))

        ax.imshow(rez)


