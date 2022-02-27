import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import imageio

# IPython-Specific

# Mesostat includes
from mesostat.utils.system import getfiles_walk
from mesostat.utils.signals.filter import zscore_dim_ord
from mesostat.visualization.mpl_colors import base_colors_rgb
from mesostat.visualization.mpl_legend import plt_add_fake_legend
from mesostat.visualization.mpl_matrix import imshow
from mesostat.visualization.mpl_colors import sample_cmap, rgb_change_color

# Local libraries
from lib.sych.area_image_process import remap_area_image


class DataFCDatabase:
    def __init__(self, param):

        # Find and parse Data filenames
        self.mice = set()
        self.dataPathsDict = {}

        # Constants
        self.expertPerfThr = 0.7

        # Get local paths
        self.localPath = os.path.dirname(os.path.abspath(__file__))
        self.dataPath = os.path.join(os.path.dirname(os.path.dirname(self.localPath)), 'data')

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

        print("Reading area color map")
        self._read_parse_area_color_map(self.dataPath)

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

    def _read_parse_area_color_map(self, path):
        image = imageio.imread(os.path.join(path, 'sych_areas.png'))
        self.areaSketch = remap_area_image(image)

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
        return list(self.trialTypeNames)

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

    def get_channel_labels(self, mousename=None):
        if mousename is None:
            mousename = list(self.mice)[0]
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            return [l.decode('UTF8') for l in h5file['channelLabels']]

    # TODO: Implement me
    def map_channel_labels_canon(self):
        return dict(zip(self.get_channel_labels(), self.get_channel_labels()))

    def get_times(self):
        return self.times

    def get_nchannels(self, mousename):
        return len(self.get_channel_labels(mousename))

    def get_ntrial_bytype(self, selector, trialType=None, performance=None):
        rezLst = []

        mousename = self._selector_to_mousename(selector)
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            sessions = [selector['session']] if 'session' in selector else self._get_sessions_h5(h5file, performance=performance)
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

    def get_performance(self, session, mousename=None):
        if mousename is None:
            mousename = self._selector_to_mousename({'session' : session})
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            return float(np.array(h5file['performance'][session]))

    def get_performance_mouse(self, mousename):
        return [self.get_performance(session, mousename) for session in self.get_sessions(mousename)]

    def get_performance_names(self):
        return ['naive', 'expert']

    def get_expert_session_idxs(self, mousename):
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            sessions = self._get_sessions_h5(h5file)
            performances = np.array([np.array(h5file['performance'][session]) for session in sessions])
            return np.where(performances >= self.expertPerfThr)[0]

    def is_expert_session(self, session, mousename=None):
        return self.get_performance(session, mousename=mousename) >= self.expertPerfThr

    def is_matching_performance(self, session, perfName, mousename=None):
        assert perfName in ['naive', 'expert']
        isExpertTrg = self.is_expert_session(session, mousename=mousename)
        isExpertSession = perfName == 'expert'
        return (isExpertTrg and isExpertSession) or ((not isExpertTrg) and (not isExpertSession))

    def get_first_expert_session_idx(self, mousename):
            expertIdxs = self.get_expert_session_idxs(mousename)
            if len(expertIdxs) == 0:
                return None
            else:
                return np.min(expertIdxs)

    def get_interval_names(self):
        return ['PRE', 'TEX', 'REW']

    def get_timestamps(self, mousename, session=None):
        return {'TEX': 3.0, 'REW': 6.0}

    def get_interval_times(self, session, mousename, interval):
        if interval == 'PRE':
            return -2, 0
        elif interval == 'TEX':
            return 3, 3.5
        elif interval == 'REW':
            return 6, 6.5
        elif interval == 'AVG':
            return 0, 8
        else:
            raise ValueError('Unexpected interval', interval)

    def find_mouse_by_session(self, session):
        return session[:5]

    # def cropRSP(self, dataRSP, startTime, endTime):
    #     assert dataRSP.shape[1] == self.targetLength
    #     idxs = np.logical_and(self.times >= startTime, self.times < endTime)
    #     return dataRSP[:, idxs]

    def get_neuro_data(self, selector, datatype='raw', zscoreDim=None, intervName=None, trialType=None, performance=None):
        dataKey = 'data_' + datatype

        dataLst = []

        mousename = self._selector_to_mousename(selector)
        path = self.dataPathsDict[mousename]
        with h5py.File(path, 'r') as h5file:
            sessions = [selector['session']] if 'session' in selector else self._get_sessions_h5(h5file, datatype=datatype, performance=performance)

            for sessionThis in sessions:
                dataRSP = np.array(h5file[dataKey][sessionThis])
                assert dataRSP.shape[1] == self.targetLength

                # Apply ZScoring if requested
                # VERY IMPORTANT: ZScoring must apply before trial type selection and cropping, it is a function of the whole dataset
                dataRSP = zscore_dim_ord(dataRSP, 'rsp', zscoreDim)

                if trialType is not None:
                    thisTrialTypeIdxs = self.get_trial_type_idxs_h5(h5file, sessionThis, trialType)

                    # print(sessions, trialType, dataRSP.shape, thisTrialTypeIdxs)

                    dataRSP = dataRSP[thisTrialTypeIdxs]
                if intervName is not None:
                    timeL, timeR = self.get_interval_times(sessionThis, mousename, intervName)
                    idxs = np.logical_and(self.times >= timeL, self.times < timeR)
                    dataRSP = dataRSP[:, idxs]

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

    def plot_area_clusters(self, fig, ax, regDict, haveLegend=False):
        trgShape = self.areaSketch.shape + (3,)
        colors = base_colors_rgb('tableau')
        rez = np.zeros(trgShape)

        imBinary = self.areaSketch == 1
        imColor = np.outer(imBinary.astype(float), np.array([0.1, 0.1, 0.1])).reshape(trgShape)
        rez += imColor

        # NOTE: Since number of clusters frequently exceeds number of discernable colors,
        # The compromise is to drop all clusters of size 1 from the plot
        regDictEff = {k: v for k, v in regDict.items() if len(v) > 1}

        if len(regDictEff) > len(colors):
            print('Warning: too many clusters to display, skipping', len(regDictEff))
            return

        for iGroup, (label, lst) in enumerate(regDictEff.items()):
            for iROI in lst:
                imBinary = self.areaSketch == (iROI + 2)
                imColor = np.outer(imBinary.astype(float), colors[iGroup]).reshape(trgShape)
                rez += imColor

        rez = rgb_change_color(rez, [0,0,0], np.array([255,255,255]))

        imshow(fig, ax, rez)
        if haveLegend:
            plt_add_fake_legend(ax, colors[:len(regDictEff)], list(regDictEff.keys()))

    def plot_area_values(self, fig, ax, valLst, vmin=None, vmax=None, cmap='jet', haveColorBar=True):
        # Mapping values to colors
        vmin = vmin if vmin is not None else np.nanmin(valLst) * 0.9
        vmax = vmax if vmax is not None else np.nanmax(valLst) * 1.1
        colors = sample_cmap(cmap, valLst, vmin, vmax, dropAlpha=True)

        trgShape = self.areaSketch.shape + (3,)
        rez = np.zeros(trgShape)

        imBinary = self.areaSketch == 1
        imColor = np.outer(imBinary.astype(float), np.array([0.1, 0.1, 0.1])).reshape(trgShape)
        rez += imColor

        for iROI, color in enumerate(colors):
            if not np.any(np.isnan(color)):
                imBinary = self.areaSketch == (iROI + 2)
                imColor = np.outer(imBinary.astype(float), color).reshape(trgShape)
                rez += imColor

        rez = rgb_change_color(rez, [0, 0, 0], np.array([255, 255, 255]))

        imshow(fig, ax, rez, haveColorBar=haveColorBar, limits=(vmin,vmax), cmap=cmap)
        return fig, ax
