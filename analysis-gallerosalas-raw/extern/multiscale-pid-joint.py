# Append base directory
import os,sys
rootname = "pub-2020-exploratory-analysis"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
sys.path.append(rootpath)
print("Appended root directory", rootpath)

from mesostat.metric.metric import MetricCalculator
from lib.gallerosalas.data_fc_db_raw import DataFCDatabase
from lib.analysis.pid_multiprocess import pid_multiprocess_mouse


# tmp_path = root_path_data if 'root_path_data' in locals() else "./"
params = {}
# params['root_path_data'] = '/home/alfomi/data/yasirdata_raw'
params['root_path_data'] = '/media/alyosha/Data/TE_data/yasirdata_raw'
# params['root_path_data'] = gui_fpath('h5path', './')

dataDB = DataFCDatabase(params)
h5outname = 'gallerosalas_result_multiregional_pid_all_df.h5'
mc = MetricCalculator(serial=True, verbose=False) #, nCore=4)

# Sweep over following parameters
argSweepDict = {
    'mousename': dataDB.mice,  # ['mvg_4']
    'intervName': dataDB.get_interval_names(),
    'datatype': ['bn_trial', 'bn_session'],
    'trialType': [None] + dataDB.get_trial_type_names(),
}

# Exclude following parameter combinations
exclQueryLst = [
    {'datatype': 'bn_trial', 'intervName': 'PRE'},   # Pre-trial interval not meaningful for bn_trial
    {'mousename' : 'mou_6', 'intervName': 'REW'}     # No reward recorded for mouse 6
]

pid_multiprocess_mouse(dataDB, mc, h5outname, argSweepDict, exclQueryLst,
                       dim=3, nBin=4, permuteTarget=False, dropChannels=[16, 26])
