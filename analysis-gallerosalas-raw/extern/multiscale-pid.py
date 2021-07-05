# Standard libraries
import h5py
import numpy as np
import pandas as pd

# Append base directory
import os,sys
rootname = "pub-2020-exploratory-analysis"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
sys.path.append(rootpath)
print("Appended root directory", rootpath)

from mesostat.metric.metric import MetricCalculator
from mesostat.utils.hdf5_helper import type_of_path
from mesostat.utils.pandas_helper import outer_product_df, drop_rows_byquery

from lib.gallerosalas.data_fc_db_raw import DataFCDatabase
import lib.analysis.pid as pid


# tmp_path = root_path_data if 'root_path_data' in locals() else "./"
params = {}
params['root_path_data'] = '/home/alfomi/data/yasirdata_raw'
# params['root_path_data'] = '/media/alyosha/Data/TE_data/yasirdata_raw'
# params['root_path_data'] = gui_fpath('h5path', './')

dataDB = DataFCDatabase(params)
h5outname = 'gallerosalas_result_multiregional_pid_df.h5'
mc = MetricCalculator(serial=True, verbose=False) #, nCore=4)

# If output file does not exist, create it
if not os.path.isfile(h5outname):
    with h5py.File(h5outname, 'w') as f:
        pass

# Sweep over following parameters
argSweepDict = {
    'mousename': ['mou_5'],  #dataDB.mice
    'intervName': dataDB.get_interval_names(),
    'datatype': dataDB.get_data_types(),
    'trialType': [None, 'Hit', 'CR']
}

# Exclude following parameter combinations
exclQueryLst = [
    {'datatype': 'bn_trial', 'intervName': 'PRE'}   # Pre-trial interval not meaningful for bn_trial
]
sweepDF = outer_product_df(argSweepDict)
sweepDF = drop_rows_byquery(sweepDF, exclQueryLst)

for idx, row in sweepDF.iterrows():
    channelNames = dataDB.get_channel_labels(row['mousename'])
    nChannels = len(channelNames)

    mouseDataLabel = 'PID_' + '_'.join([str(key) for key in row.keys()])
    for session in dataDB.get_sessions(row['mousename'], datatype=row['datatype']):
        sessionDataLabel = mouseDataLabel + '_' + session
        print(sessionDataLabel)

        if type_of_path(h5outname, sessionDataLabel) is not None:
            print(sessionDataLabel, 'already calculated, skipping')
        else:
            kwargs = dict(row)
            del kwargs['mousename']

            # Get data
            dataLst = dataDB.get_neuro_data({'session': session}, zscoreDim=None, **kwargs)

            # Calculate PID
            rezDF = pid.pid(dataLst, mc, channelNames, labelsSrc=None, labelsTrg=None, nPerm=2000, nBin=4)

            # Save to file
            rezDF.to_hdf(h5outname, sessionDataLabel, mode='a', format='table', data_columns=True)
