# Standard libraries

# Append base directory
import os,sys
rootname = "pub-2020-exploratory-analysis"
thispath = os.path.dirname(os.path.abspath(__file__))
rootpath = os.path.join(thispath[:thispath.index(rootname)], rootname)
sys.path.append(rootpath)
print("Appended root directory", rootpath)


from mesostat.metric.metric import MetricCalculator
from mesostat.utils.hdf5_io import DataStorage

from lib.sych.data_fc_db_raw import DataFCDatabase
from lib.analysis.metric_helper import metric_by_session

# tmp_path = root_path_data if 'root_path_data' in locals() else "./"
params = {}
params['root_path_data'] = '/home/alfomi/data/sych_preprocessed'
# params['root_path_data'] = '/media/alyosha/Data/TE_data/yarodata/sych_preprocessed'
# params['root_path_data'] = gui_fpath('h5path', './')

dataDB = DataFCDatabase(params)
ds = DataStorage('sych_result_multiregional_df.h5')
mc = MetricCalculator(serial=False, verbose=False) #, nCore=4)

cropTimes = {'TEX' : (3.0, 3.5), 'REW' : (6.0, 6.5)}
metricName = 'BivariateMI'
metricSettings = {
    'min_lag_sources': 0,
    'max_lag_sources': 0,
    "cmi_estimator": "JidtGaussianCMI",
    'parallelTrg': True
}

for iMouse, mousename in enumerate(sorted(dataDB.mice)):
    for datatype in dataDB.get_data_types():
        zscoreDim = 'rs' if datatype == 'raw' else None
        for cropTimeName, cropTime in cropTimes.items():            
            dataName = metricName + '_' + datatype
            print(dataName)
            metric_by_session(dataDB, mc, ds, mousename, metricName, '', dataName='All_'+cropTimeName,
                              datatype=datatype, trialType='iGO', cropTime=cropTime,
                              zscoreDim=zscoreDim, metricSettings=metricSettings)
