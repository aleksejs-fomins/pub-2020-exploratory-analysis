# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Append base directory
import os, sys

rootname = "pub-2020-exploratory-analysis"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
sys.path.append(rootpath)
print("Appended root directory", rootpath)

from mesostat.metric.metric import MetricCalculator
from mesostat.utils.hdf5_helper import type_of_path

from lib.sych.data_fc_db_raw import DataFCDatabase
import lib.analysis.pid as pid

# tmp_path = root_path_data if 'root_path_data' in locals() else "./"
params = {}
params['root_path_data'] = '/home/alfomi/data/sych_preprocessed'
# params['root_path_data'] = '/media/alyosha/Data/TE_data/yarodata/sych_preprocessed'
# params['root_path_data'] = gui_fpath('h5path', './')

dataDB = DataFCDatabase(params)
h5outname = 'sych_result_multiregional_pid_df.h5'
mc = MetricCalculator(serial=True, verbose=False)  # , nCore=4)

cropTimes = {'TEX': (3.0, 3.5), 'REW': (6.0, 6.5)}

for mousename in ['mvg_9']:  # dataDB.mice:
    channelNames = dataDB.get_channel_labels(mousename)
    nChannels = len(channelNames)

    for datatype in ['bn_session']:
        for session in dataDB.get_sessions(mousename, datatype=datatype):
            for intervKey, interv in cropTimes.items():
                dataLabel = '_'.join(['PID', mousename, datatype, session, intervKey])
                if type_of_path(h5outname, dataLabel) is not None:
                    print(dataLabel, 'already calculated, skipping')
                else:
                    dataLst = dataDB.get_neuro_data({'session': session}, datatype=datatype,
                                                    zscoreDim=None, cropTime=interv,
                                                    trialType='iGO')

                    rezLst = []
                    for iSrc1 in range(nChannels):
                        for iSrc2 in range(iSrc1 + 1, nChannels):
                            src1 = channelNames[iSrc1]
                            src2 = channelNames[iSrc2]
                            sources = [src1, src2]
                            print(datetime.now().time(), datatype, session, intervKey, sources)

                            targets = list(set(channelNames) - set(sources))
                            rezLst += [pid.pid(dataLst, mc, channelNames, sources, targets, nPerm=2000, nBin=4)]

                    rezDF = pd.concat(rezLst, sort=False).reset_index(drop=True)

                    # Save to file
                    rezDF.to_hdf(h5outname, dataLabel, mode='a', format='table', data_columns=True)
