import json
import numpy as np
import pandas as pd
from os.path import basename, dirname, join, isfile, splitext
from scipy.spatial import ConvexHull
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt

# IPython
from IPython.display import display
from ipywidgets import IntProgress

# Mesostat
from mesostat.utils.pandas_helper import pd_rows_colval, pd_query
from mesostat.utils.matlab_helper import loadmat
from mesostat.utils.system import getfiles_walk
from mesostat.utils.arrays import numpy_transpose_byorder


class DataFCDatabase :
    def __init__(self, param):
        # Find and parse Data filenames
        self.mice = set()
        self.sessions = set()
        self.metaDataFrames = {}

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


    # Channel labels are brain regions associated to each channel index
    # The channel labels need not be consistent across mice, or even within one mouse
    def _find_read_channel_labels(self, path):
        labelFileName = join(path, "ROI_names.mat")

        if not isfile(labelFileName):
            raise ValueError("Can't find file", labelFileName)

        self.channelLabels = loadmat(labelFileName)['ROI_names']


    # Find the 2D mapping file of hemispheric cortical areas according to allen brain atlas
    def _find_read_allen_map(self, path):
        labelFileName = join(path, "Allen_areas.mat")
        if not isfile(labelFileName):
            raise ValueError("Can't find file", labelFileName)

        self.allenMap = loadmat(labelFileName)['L']
        self.allenIndices = sorted(list(set(self.allenMap.flatten())))
        self.allenCounts = [np.sum(self.allenMap == i) for i in self.allenIndices ]


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
            "mousename" : [basename(p[0]) for p in dataPaths],
            "session" : [splitext(p[1])[0] for p in dataPaths]
        })

        self.mice.update(set(self.metaDataFrames['neuro']['mousename']))
        self.sessions.update(set(self.metaDataFrames['neuro']['session']))


    def read_neuro_files(self):
        if 'neuro' in self.metaDataFrames.keys():
            nNeuroFiles = self.metaDataFrames['neuro'].shape[0]

            self.dataNeuronal = []
            self.performanceDPDict = {}
            progBar = IntProgress(min=0, max=nNeuroFiles, description='Read Neuro Data:')
            display(progBar)  # display the bar
            for idx, row in self.metaDataFrames['neuro'].iterrows():
                matFile = loadmat(row['path'], waitRetry=3)
                for k, val in matFile.items():
                    if 'Hit_timecourse' in k:
                        self.dataNeuronal += [numpy_transpose_byorder(val, 'psr', 'rps')]
                    elif k == 'dp':
                        self.performanceDPDict[row['session']] = val
                progBar.value += 1
        else:
            print("No Neuro files loaded, skipping reading part")


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


    # Get timesteps for data given index
    # If window greater than 1 is provided, return timesteps of data sweeped with a moving window of that length
    def get_times_by_idx(self, nTime, window=1):
        if window == 1:
            return np.arange(nTime) / self.targetFreq
        else:
            return (np.arange(nTime - window + 1) + (window - 1) / 2) / self.targetFreq


    def get_rows(self, frameName, coldict):
        return pd_query(self.metaDataFrames[frameName], coldict)


    def label_plot_timestamps(self, ax, linecolor='y', textcolor='k', shX=-0.5, shY=0.05):
        # the x coords of this transformation are data, and the y coord are axes
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        for t, label in self.timestamps.items():
            ax.axvline(x=t, color=linecolor, linestyle='--')
            plt.text(t+shX, shY, label, color=textcolor, verticalalignment='bottom', transform=trans, rotation=90)