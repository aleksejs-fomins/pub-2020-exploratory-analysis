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
from mesostat.utils.pandas_helper import pd_is_one_row, pd_query
from mesostat.utils.matlab_helper import loadmat
from mesostat.utils.signals.filter import zscore_dim_ord
from mesostat.utils.strings import enum_nonunique
from mesostat.visualization.mpl_colors import base_colors_rgb
from mesostat.visualization.mpl_legend import plt_add_fake_legend
from mesostat.visualization.mpl_matrix import imshow
from mesostat.visualization.mpl_colors import sample_cmap, rgb_change_color

# Local
from lib.gallerosalas.preprocess_common import calc_allen_shortest_distances


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

        # print("Reading task structure")
        # self._find_read_task_structure(param["root_path_data"])
        self.timestamps = {'texture': 2.0, 'delay': 5.0}

        print("Reading session structure")
        self._find_read_session_structure(param["root_path_data"])

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

    # def _find_read_task_structure(self, path):
    #     taskFileName = join(path, 'task_structure.json')
    #
    #     with open(taskFileName) as f:
    #         self.timestamps = json.load(f)['timestamps']
    #         self.timestamps = {float(k) : v for k,v in self.timestamps.items()}

    def _find_read_session_structure(self, path):
        pwdSessionStruct = join(path, "sessions_tex.csv")
        if not isfile(pwdSessionStruct):
            raise ValueError("Can't find file", pwdSessionStruct)

        self.dfSessions = pd.read_csv(pwdSessionStruct, sep='\t')
        self.dfSessions['session'] = [row['dateKey'] + '_' + row['sessionKey'] for idx, row in self.dfSessions.iterrows()]

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
        self.allenDist = calc_allen_shortest_distances(self.allenMap, self.allenIndices)

    def get_data_types(self):
        return self.dataTypes

    def get_metadata(self, session, mousename=None):
        if mousename is None:
            mousename = self.find_mouse_by_session(session)

        return pd.read_hdf(self.datapaths[mousename], '/metadataTrial/'+session)

    def get_trial_types(self, session, mousename=None):
        df = self.get_metadata(session, mousename=mousename)
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

    def map_channel_labels_canon(self):
        return dict(zip(self.get_channel_labels(), list(self.channelAreasDF['LCanon'])))

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

    def get_delay_length(self, mousename, session):
        # row = pd_is_one_row(pd_query(self.dfSessions, {'mousename': mousename, 'session': session}))[1]
        # return row['delay']
        df = pd.read_hdf(self.datapaths[mousename], 'metadataSession')
        return pd_is_one_row(df[df['session'] == session])[1]['delay']

    # Get timestamps of events during single trial (seconds)
    def get_timestamps(self, mousename, session):
        timestamps = self.timestamps.copy()
        timestamps['report'] = timestamps['delay'] + self.get_delay_length(mousename, session)
        return timestamps

    def get_interval_names(self):
        return ['PRE', 'TEX', 'DEL', 'REW']

    def get_interval_times(self, session, mousename, interval):
        if interval == 'PRE':
            return [[0, 1]]
        elif interval == 'TEX':
            return [[2, 3]]
        elif interval == 'DEL':
            return [[5, 6]]
        elif interval == 'REW':
            if mousename == 'mou_6':
                raise IOError('Mouse 6 does not have reward')

            delayLen = self.get_delay_length(mousename, session)
            return [5 + np.array([delayLen, delayLen + 0.85])]
        elif interval == 'AVG':
            return [self.get_interval_times(session, mousename, i)[0] for i in ['TEX', 'DEL', 'REW']]
        else:
            raise ValueError('Unexpected interval', interval)

    def find_mouse_by_session(self, session):
        miceLst = list(self.mice)
        miceHave = [session in self.sessions[mousename] for mousename in miceLst]
        assert np.sum(miceHave) == 1
        return miceLst[np.where(miceHave)[0][0]]

    def get_absolute_times(self, mousename, session, FPS=20):
        with h5py.File(self.datapaths[mousename], 'r') as h5file:
            nSample = h5file['data'][session].shape[1]

        '''
            1. Load session metadata
            2. Convert all times to timestamps
            3. From all timestamps, subtract first, convert to seconds
            4. Extract data, get nTimes from shape
            5. Set increment, return
        '''

        # df = pd.read_hdf(self.datapaths[mousename], '/metadata/' + session)
        # timeStamps = pd.to_datetime(df['time_stamp'], format='%H:%M:%S.%f')
        # timeDeltas = timeStamps - timeStamps[0]
        #
        # timesSh = np.arange(nSample) / FPS
        # timesRS = np.array([t.total_seconds() + timesSh for t in timeDeltas])

        df = pd.read_hdf(self.datapaths[mousename], '/metadataTrial/' + session)
        timesSh = np.arange(nSample) / FPS
        timesRS = np.array([timesSh + t for t in df['Time']])

        return timesRS

    def get_neuro_data(self, selector, datatype='raw', zscoreDim=None, intervName=None, trialType=None, performance=None):
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

                print(dataRSP.shape, len(trialTypes))
                dataRSP = dataRSP[trialTypes == trialType]

            # Apply ZScoring if requested
            # VERY IMPORTANT: ZScoring must apply before trial type selection and cropping, it is a function of the whole dataset
            dataRSP = zscore_dim_ord(dataRSP.copy(), self.dimOrdCanon, zscoreDim)

            if intervName is not None:
                timeIdxs = np.zeros(len(times))

                timesLst = self.get_interval_times(session, mousename, intervName)
                for timeL, timeR in timesLst:
                    timeIdxsThis = np.logical_and(times >= timeL, times < timeR)
                    timeIdxs = np.logical_or(timeIdxs, timeIdxsThis)

                # FIXME: Time index hardcoded. Make sure it works for any dimOrdCanon
                dataRSP = dataRSP[:, timeIdxs]
            else:
                print('Warning: using non-uniform duration across sessions')

                # if selectorType == 'mousename':
                #     raise IOError('Why do we need this?')
                #
                # # FIXME: Number of timesteps hardcoded. Perhaps there is a better way
                # # dataRSP = dataRSP[:, :160]  # Ensuring all trials that are too long are cropped to this time

            dataLst += [dataRSP]

        return dataLst

    def label_plot_timestamps(self, ax, mousename, session, linecolor='y', textcolor='k', shX=-0.5, shY=0.05):
        # the x coords of this transformation are data, and the y coord are axes
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        timestamps = self.get_timestamps(mousename, session)
        for label, t in timestamps.items():
            ax.axvline(x=t, color=linecolor, linestyle='--')
            plt.text(t+shX, shY, label, color=textcolor, verticalalignment='bottom', transform=trans, rotation=90)

    def plot_area_clusters(self, fig, ax, regDict, haveLegend=False):
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

        imshow(fig, ax, rez)
        if haveLegend:
            plt_add_fake_legend(ax, colors[:len(regDict)], list(regDict.keys()))

    def plot_area_values(self, fig, ax, valLst, vmin=None, vmax=None, cmap='jet', haveColorBar=True):
        # Mapping values to colors
        vmin = vmin if vmin is not None else np.min(valLst) * 0.9
        vmax = vmax if vmax is not None else np.max(valLst) * 1.1
        colors = sample_cmap(cmap, valLst, vmin, vmax, dropAlpha=True)

        trgShape = self.allenMap.shape + (3,)
        rez = np.zeros(trgShape)

        imBinary = self.allenMap == 0
        imColor = np.outer(imBinary.astype(float), np.array([0.5, 0.5, 0.5])).reshape(trgShape)
        rez += imColor

        for iROI, color in enumerate(colors):
            if not np.any(np.isnan(color)):
                imBinary = self.allenMap == self.allenIndices[iROI]

                imColor = np.outer(imBinary.astype(float), color).reshape(trgShape)
                rez += imColor

        rez = rgb_change_color(rez, [0, 0, 0], np.array([255, 255, 255]))
        imshow(fig, ax, rez, haveColorBar=haveColorBar, limits=(vmin,vmax), cmap=cmap)
