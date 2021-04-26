# Standard libraries
import h5py
import numpy as np
import pandas as pd
from datetime import datetime

# Append base directory
import os,sys
rootname = "pub-2020-exploratory-analysis"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
sys.path.append(rootpath)
print("Appended root directory", rootpath)

from mesostat.metric.metric import MetricCalculator
from mesostat.utils.hdf5_helper import type_of_path

from lib.gallerosalas.data_fc_db_raw import DataFCDatabase
import lib.analysis.pid as pid


# tmp_path = root_path_data if 'root_path_data' in locals() else "./"
params = {}
params['root_path_data'] = '/home/alfomi/data/yasirdata_raw'
# params['root_path_data'] = '/media/alyosha/Data/TE_data/yasirdata_raw'
# params['root_path_data'] = gui_fpath('h5path', './')

dataDB = DataFCDatabase(params)
h5outname = 'gallerosalas_result_multiregional_pid_all_df_rand.h5'
mc = MetricCalculator(serial=True, verbose=False) #, nCore=4)

cropTimes = {
    "PRE" : [0.0, 1.0],
    "TEX" : [2.5, 3.5],
    "DEL" : [5.0, 6.0],
    "REW" : [7.0, 8.0]
}

print(dataDB.get_channel_labels('mou_5'))


if not os.path.isfile(h5outname):
    with h5py.File(h5outname, 'w') as f:
        pass

for mousename in ['mou_5']: #dataDB.mice:
    channelNames = list(dataDB.get_channel_labels(mousename))
    nChannels = len(channelNames)

    for trialType in [None, 'Hit', 'CR']:
        dataLabel = '_'.join(['PID', mousename, str(trialType)])
        print(dataLabel)

        if type_of_path(h5outname, dataLabel) is not None:
            print(dataLabel, 'already calculated, skipping')
        else:
            dataLst = dataDB.get_neuro_data({'mousename': mousename}, trialType=trialType)

            dataRSP = np.concatenate(dataLst, axis=0)  # Concatenate trials and sessions
            dataRSPRand = np.random.uniform(0, 1, dataRSP.shape)

            rezDF = pid.pid([dataRSPRand], mc, channelNames,
                            labelsSrc=None, labelsTrg=None, nPerm=0, nBin=4)

            # Save to file
            rezDF.to_hdf(h5outname, dataLabel, mode='a', format='table', data_columns=True)
