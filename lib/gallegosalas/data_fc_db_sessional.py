import json
import numpy as np
import pandas as pd
from os.path import basename, join, isfile, splitext
from scipy.spatial import ConvexHull
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt

# IPython
from IPython.display import display
from ipywidgets import IntProgress

# Mesostat
from mesostat.utils.pandas_helper import pd_query
from mesostat.utils.matlab_helper import loadmat
from mesostat.utils.system import getfiles_walk
from mesostat.utils.arrays import numpy_transpose_byorder
from mesostat.utils.signals.resample import zscore_dim_ord

# Local
from lib.preprocessing.dff import dff


class DataFCDatabase:
    def __init__(self, param):
        # Find and parse Data filenames
        self.mice = set()
        self.sessions = set()
        self.metaDataFrames = {}

        self.dimOrdCanon = 'rsp'

        ##################################
        # Define resampling frequency
        ##################################
        # self.targetTimesteps = [20, 95]      # Crop timeframe to important stuff
        # self.targetChannels = np.arange(25)  # Crop last two brain regions, because they disbehave (too many nans)
        self.targetFreq = 20  # Hz
        # self.targetTimes = np.arange(*self.targetTimesteps) / self.targetFreq
        # print("Target range of", self.targetTimesteps, "timesteps amounts to", [self.targetTimes[0], self.targetTimes[-1], "seconds"])

        ##################################
        # Find and parse data files
        ##################################
        print("Reading channel label file")
        self._find_read_channel_labels(param["root_path_data"])

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

    def _find_read_allen_map(self, path):
        '''
            Find the 2D mapping file of hemispheric cortical areas according to allen brain atlas
        '''
        labelFileName = join(path, "Allen_areas.mat")
        if not isfile(labelFileName):
            raise ValueError("Can't find file", labelFileName)

        self.allenMap = loadmat(labelFileName)['L']                                # 2D map of cortical regions
        self.allenIndices = sorted(list(set(self.allenMap.flatten())))             # Indices of regions
        self.allenCounts = [np.sum(self.allenMap == i) for i in self.allenIndices] # Number of pixels per region

    def _find_read_task_structure(self, path):
        taskFileName = join(path, 'task_structure.json')

        with open(taskFileName) as f:
            self.timestamps = json.load(f)['timestamps']
            self.timestamps = {float(k) : v for k,v in self.timestamps.items()}

    def _find_parse_neuro_files(self, path):
        # Get all mat files in the root tree except those explicitly in the root path
        dataPaths = [p for p in getfiles_walk(path, [".mat"]) if p[0] != path]

        self.metaDataFrames['neuro'] = pd.DataFrame({
            "path" : [join(p[0], p[1]) for p in dataPaths],
            "mousename" : [basename(p[0]).split('_')[0] for p in dataPaths],
            "session": [splitext(p[1])[0] for p in dataPaths],
            "datatype": ['_'.join(basename(p[0]).split('_')[1:]) for p in dataPaths],
            "performance": [0.0 for p in dataPaths]
        })

        self.mice.update(set(self.metaDataFrames['neuro']['mousename']))
        self.sessions.update(set(self.metaDataFrames['neuro']['session']))

    # FIXME: Hardcoded nChannel last. Make sure it works for any canonical dim order
    def _allen_normalize_data(self, data):
        nTrial, nTime, nChannel = data.shape
        for iChannel in range(nChannel):
            data[:, :, iChannel] /= self.allenCounts[iChannel + 2]  # First two values are background and region separators
        return data

    # def _selector_to_mousename(self, selector):
    #     if 'mousename' in selector:
    #         return selector['mousename']
    #     else:
    #         rows = self.get_rows('neuro', selector)
    #         mice = set(rows['mousename'])
    #         assert len(mice) == 1
    #         return list(mice)[0]

    def read_neuro_files(self):
        if 'neuro' in self.metaDataFrames.keys():
            nNeuroFiles = self.metaDataFrames['neuro'].shape[0]

            self.dataNeuronal = []
            progBar = IntProgress(min=0, max=nNeuroFiles, description='Read Neuro Data:')
            display(progBar)  # display the bar
            for idx, row in self.metaDataFrames['neuro'].iterrows():
                matFile = loadmat(row['path'], waitRetry=3)
                for k, val in matFile.items():
                    if 'Hit_timecourse' in k:
                        data = numpy_transpose_byorder(val, 'psr', self.dimOrdCanon)
                        if 'yasir' not in row['datatype']:
                            data = self._allen_normalize_data(data)   # Normalize data, unless already normalized by Yasir

                        self.dataNeuronal += [data]
                    elif k == 'dp':
                        self.metaDataFrames['neuro'].at[idx, 'performance'] = val

                progBar.value += 1
        else:
            print("No Neuro files loaded, skipping reading part")

    def _get_rows(self, frameName, coldict):
        return pd_query(self.metaDataFrames[frameName], coldict)

    def baseline_normalization(self):
        '''
            Compute baseline-normalized data, add to storage
        '''
        metaDFExtra = pd.DataFrame()

        for mousename in self.mice:
            rows = self._get_rows('neuro', {'mousename' : mousename, 'datatype' : 'raw'})
            for idx, row in rows.iterrows():
                times, data = self.get_data_by_idx(idx)

                for bnType in ['dff_trial', 'dff_session']:
                    self.dataNeuronal += [dff(times, data, self.dimOrdCanon, bnType, tBaseMin=None, tBaseMax=0.5)]
                    rowThis = row.copy()
                    rowThis['datatype'] = bnType
                    rowThis['path'] = None
                    metaDFExtra = metaDFExtra.append(rowThis)

            # IMPORTANT: New rows must come at the end and carry increasing index to match index in dataNeuronal
            self.metaDataFrames['neuro'] = self.metaDataFrames['neuro'].append(metaDFExtra, ignore_index=True)

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
        return list(set(self.metaDataFrames['neuro']['datatype']))

    def get_channel_labels(self, mousename):
        return self.channelLabels

    def get_nchannels(self, mousename):
        return len(self.get_channel_labels(mousename))

    def get_sessions(self, mousename):
        return self.sessions

    # If window greater than 1 is provided, return timesteps of data sweeped with a moving window of that length
    def get_times(self, nTime, window=1):
        if window == 1:
            return np.arange(nTime) / self.targetFreq
        else:
            return (np.arange(nTime - window + 1) + (window - 1) / 2) / self.targetFreq

    # FIXME: Hardcoded nTime second. Make sure it works for any canonical dim order
    def get_data_by_idx(self, idx):
        data = self.dataNeuronal[idx]
        nTrial, nTime, nChannel = data.shape
        times = self.get_times(nTime)
        return times, data

    def get_neuro_data(self, selector, datatype='dff_raw', zscoreDim=None, cropTime=None): #, trialType=None, performance=None):
        rows = self._get_rows('neuro', {**selector, **{'datatype' : datatype}})

        dataLst = []
        for idx, row in rows.iterrows():
            times, data = self.get_data_by_idx(idx)

            # Apply ZScoring if requested
            # VERY IMPORTANT: ZScoring must apply before trial type selection and cropping, it is a function of the whole dataset
            dataRSP = zscore_dim_ord(data.copy(), self.dimOrdCanon, zscoreDim)

            if cropTime is not None:
                # FIXME: Time index hardcoded. Make sure it works for any dimOrdCanon
                timeIdxs = np.logical_and(times >= cropTime[0], times <= cropTime[1])
                dataRSP = dataRSP[:, timeIdxs]

            dataLst += [dataRSP]

        return dataLst

    def label_plot_timestamps(self, ax, linecolor='y', textcolor='k', shX=-0.5, shY=0.05):
        # the x coords of this transformation are data, and the y coord are axes
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        for t, label in self.timestamps.items():
            ax.axvline(x=t, color=linecolor, linestyle='--')
            plt.text(t+shX, shY, label, color=textcolor, verticalalignment='bottom', transform=trans, rotation=90)
