import h5py
import numpy as np

fname = 'gallerosalas_result_timesweep_pid_all_df.h5'

with h5py.File(fname, 'r') as f:
    for mousename in ['mou_5', 'mou_6', 'mou_7', 'mou_9']:
        key = 'PID_'+mousename+'_bn_trial_None_20'
        print(key)
        data = np.array(f[key])
        for i in range(4):
            print(np.max(data[:, :, i]))
