import numpy as np
import pandas as pd
from os.path import basename, dirname, join, isfile

# IPython
from IPython.display import display
from ipywidgets import IntProgress

# Mesostat
from mesostat.utils.pandas_helper import pd_rows_colval, pd_query
from mesostat.utils.matlab_helper import loadmat
from mesostat.utils.system import getfiles_walk


class DataFCDatabase :
    def __init__(self, param):

        # Adapt paths
        param["root_path_data"] = dirname(param["experiment_path"])

        # Find and parse Data filenames
        self.mice = set()
        self.metaDataFrames = {}


        ##################################
        # Define resampling frequency
        ##################################
        self.targetTimesteps = [20, 95]      # Crop timeframe to important stuff
        self.targetChannels = np.arange(25)  # Crop last two brain regions, because they disbehave (too many nans)
        self.targetFreq = 20  # Hz
        self.targetTimes = np.arange(*self.targetTimesteps) / self.targetFreq
        print("Target range of", self.targetTimesteps, "timesteps amounts to", [self.targetTimes[0], self.targetTimes[-1], "seconds"])

        ##################################
        # Find and parse data files
        ##################################
        if "root_path_data" in param.keys():
            print("Reading channel label file")
            self._find_read_channel_labels(param["root_path_data"])
            print("Searching for data files")
            self._find_parse_neuro_files(param["experiment_path"])
        else:
            print("No data path provided, skipping")
    #
    #     ##################################
    #     # Compute summary
    #     ##################################
    #     sumByMouse = lambda dataset: [dataset[dataset['mousename'] == mousename].shape[0] for mousename in self.mice]
    #
    #     self.summary = pd.DataFrame({
    #         key : sumByMouse(dataFrame) for key, dataFrame in self.metaDataFrames.items()
    #     }, index=self.mice)


    # Channel labels are brain regions associated to each channel index
    # The channel labels need not be consistent across mice, or even within one mouse
    def _find_read_channel_labels(self, path):
        labelFileName = join(path, "ROIs_names.mat")

        if not isfile(labelFileName):
            raise ValueError("Can't find file", labelFileName)

        self.channelLabels = loadmat(labelFileName)['ROIs_names']


    def _find_parse_neuro_files(self, path):
        dataPaths = [p[0] for p in getfiles_walk(path, ["data.mat"])]
        dataPathsRel = np.array([p[len(path) + 1:].split('/') for p in dataPaths])

        if dataPathsRel.shape[1] == 4:
            columns = ['mousename', 'activity', 'task_type', 'lolo']
        else:
            columns = ['mousename', 'lolo']

        self.metaDataFrames['neuro'] = pd.DataFrame(dataPathsRel, columns=columns)

        self.metaDataFrames['neuro'].insert(1, "path", dataPaths)
        self.mice.update(set(self.metaDataFrames['neuro']['mousename']))


    def read_neuro_files(self):
        if 'neuro' in self.metaDataFrames.keys():
            nNeuroFiles = self.metaDataFrames['neuro'].shape[0]

            self.dataNeuronal = []
            progBar = IntProgress(min=0, max=nNeuroFiles, description='Read Neuro Data:')
            display(progBar)  # display the bar
            for idx, datapath in enumerate(self.metaDataFrames['neuro']['path']):
                filepath = join(datapath, 'data.mat')
                data3D = loadmat(filepath, waitRetry=3)['data']

                # Crop bad channels
                data3D = data3D[:, :, self.targetChannels]

                # Crop to important timeframe
                data3D = data3D[:, self.targetTimesteps[0] : self.targetTimesteps[1]]

                # Crop bad trials
                goodTrialIdxs = ~np.any(np.isnan(data3D), axis=(1, 2))
                data3D = data3D[goodTrialIdxs]

                self.dataNeuronal += [data3D]
                progBar.value += 1

        else:
            print("No Neuro files loaded, skipping reading part")

    #
    #
    # # Mark days as naive or expert based on performance threshold
    # def mark_days_expert_naive(self, pTHR):
    #     nNeuroFiles = self.metaDataFrames['neuro'].shape[0]
    #     isExpert = np.zeros(nNeuroFiles, dtype=bool)
    #     deltaDays = np.zeros(nNeuroFiles)
    #     deltaDaysCentered = np.zeros(nNeuroFiles)
    #
    #     # For each mouse, determine which sessions are naive and which expert
    #     # Also determine number of days passed since start and since expert
    #     for mousename in self.mice:
    #         thisMouseMetadata = filter_rows_colval(self.metaDataFrames['neuro'], 'mousename', mousename)
    #         thisMouseDataIdxs = np.array(thisMouseMetadata["date"].index)
    #         perf = self.dataPerformance[thisMouseDataIdxs]
    #         skillRez = mouse_performance_allsessions(list(thisMouseMetadata["date"]), perf, pTHR)
    #         isExpert[thisMouseDataIdxs], deltaDays[thisMouseDataIdxs], deltaDaysCentered[thisMouseDataIdxs] = skillRez
    #
    #     # Add these values to metadata
    #     self.metaDataFrames['neuro']['isExpert'] = isExpert
    #     self.metaDataFrames['neuro']['deltaDays'] = deltaDays
    #     self.metaDataFrames['neuro']['deltaDaysCentered'] = deltaDaysCentered
    #
    #
    # def get_channel_labels(self, mousename):
    #     return self.channelLabelsDict[mousename]
    #
    #
    # def get_nchannels(self, mousename):
    #     return len(self.channelLabelsDict[mousename])
    #
    #
    def get_rows(self, frameName, coldict):
        return pd_query(self.metaDataFrames[frameName], coldict)
    #
    #
    # # Find FC data for specified rows, then crop to selected time range
    # def get_fc_data(self, idx, rangeSec=None):
    #     timesThis = self.dataTEtimes[idx]
    #     fcThis = self.dataTEFC[idx]
    #     if rangeSec is None:
    #         return timesThis, fcThis
    #     else:
    #         rng = slice_sorted(timesThis, rangeSec)
    #         return timesThis[rng[0]:rng[1]], fcThis[..., rng[0]:rng[1]]
    #
    #
    # # Provide rows for all sessions of the same mouse, iterating over combinations of other anaylsis parameters
    # def mouse_iterator(self):
    #     sweepCols = ["mousename",  "analysis", "trial", "range", "method"]
    #     sweepValues = [self.summaryTE[colname].keys() for colname in sweepCols]
    #     sweepProduct = list(itertools.product(*sweepValues))
    #
    #     for sweepComb in sweepProduct:
    #         sweepCombDict = dict(zip(sweepCols, sweepComb))
    #         rows = self.get_rows('TE', sweepCombDict)
    #         if rows.shape[0] > 0:
    #             yield sweepCombDict, rows
