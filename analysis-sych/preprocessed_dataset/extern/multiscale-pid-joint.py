# Append base directory
import os,sys
rootname = "pub-2020-exploratory-analysis"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
sys.path.append(rootpath)
print("Appended root directory", rootpath)

from mesostat.metric.metric import MetricCalculator
from lib.sych.data_fc_db_raw import DataFCDatabase
from lib.analysis.triplet_compute.datasweep import multiprocess_mouse


# tmp_path = root_path_data if 'root_path_data' in locals() else "./"
params = {}
#params['root_path_data'] = '/home/alfomi/data/sych_preprocessed'
# params['root_path_data'] = '/media/alyosha/Data/TE_data/yarodata/sych_preprocessed'
# params['root_path_data'] = gui_fpath('h5path', './')
params['root_path_data'] = '/mnt/aleksejs/project-pub1-hierarchical/sych_preprocessed/'

dataDB = DataFCDatabase(params)
mc = MetricCalculator(serial=True, verbose=False) #, nCore=4)

# Sweep over following parameters
argSweepDict = {
    'mousename': dataDB.mice,  # ['mvg_4']
    'intervName': dataDB.get_interval_names(),
    'datatype': ['bn_trial', 'bn_session'],
    'trialType': ['None', 'iGO', 'iNOGO'],
    'performance': ['naive', 'expert']
}

# Exclude following parameter combinations
exclQueryLst = [
    {'datatype': 'bn_trial', 'intervName': 'PRE'}   # Pre-trial interval not meaningful for bn_trial
]


for nBin in [2]:
    for permuteTarget in [False, True]:
        randKey = 'rand' if permuteTarget else 'data'
        h5outname = 'pid_sych_multimouse_nbin_' + str(nBin) + '_' + randKey + '.h5'

        multiprocess_mouse(dataDB, mc, h5outname, argSweepDict, exclQueryLst, 'PID', metric='BivariatePID',
                           dim=3, nBin=nBin, permuteTarget=permuteTarget, dropChannels=[21])
