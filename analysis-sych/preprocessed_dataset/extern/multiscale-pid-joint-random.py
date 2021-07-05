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

from lib.sych.data_fc_db_raw import DataFCDatabase
import lib.analysis.pid as pid


# tmp_path = root_path_data if 'root_path_data' in locals() else "./"
params = {}
params['root_path_data'] = '/home/alfomi/data/sych_preprocessed'
# params['root_path_data'] = '/media/alyosha/Data/TE_data/yarodata/sych_preprocessed'
# params['root_path_data'] = gui_fpath('h5path', './')

dataDB = DataFCDatabase(params)
h5outname = 'sych_result_multiregional_pid_all_df_rand.h5'
mc = MetricCalculator(serial=True, verbose=False) #, nCore=4)

# If output file does not exist, create it
if not os.path.isfile(h5outname):
    with h5py.File(h5outname, 'w') as f:
        pass

# Sweep over following parameters
argSweepDict = {
    'mousename': ['mvg_4'],  #dataDB.mice
    'intervName': dataDB.get_interval_names(),
    'datatype': dataDB.get_data_types(),
    'trialType': [None, 'iGO', 'iNOGO'],
    'performance': ['naive', 'expert']
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
    if type_of_path(h5outname, mouseDataLabel) is not None:
        print(mouseDataLabel, 'already calculated, skipping')
    else:
        kwargs = dict(row)
        del kwargs['mousename']

        # Get data
        dataLst = dataDB.get_neuro_data({'mousename': row['mousename']}, zscoreDim=None, **kwargs)

        # Calculate PID
        rezDF = pid.pid(dataLst, mc, channelNames, labelsSrc=None, labelsTrg=None, nPerm=0, nBin=4, permuteTarget=True)

        # Save to file
        rezDF.to_hdf(h5outname, mouseDataLabel, mode='a', format='table', data_columns=True)
