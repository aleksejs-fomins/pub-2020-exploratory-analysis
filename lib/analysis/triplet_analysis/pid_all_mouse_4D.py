import h5py
import numpy as np
import seaborn as sns
import pandas as pd

from mesostat.metric.idtxl_pid import multivariate_pid_key



# import matplotlib.pyplot as plt
# from scipy.stats import mannwhitneyu, fisher_exact
# from sklearn.metrics import cohen_kappa_score
# from pathlib import Path
#
# from IPython.display import display
#
# from mesostat.utils.matrix import drop_channels, offdiag_1D, matrix_copy_triangle_symmetric
# from mesostat.stat.stat import continuous_empirical_CDF
# from mesostat.stat.classification import confusion_matrix
# from mesostat.stat.clustering import cluster_dist_matrix_min, cluster_plot
#
# from mesostat.utils.pandas_helper import pd_query, pd_merge_multiple, pd_is_one_row, pd_append_row, pd_pivot
# from mesostat.visualization.mpl_barplot import barplot_stacked_indexed, barplot_labeled, sns_barplot
# from mesostat.visualization.mpl_matrix import imshow


def read_rez(h5fname, keyPID):
    keyLabel = 'Label_' + keyPID[4:]

    with h5py.File(h5fname, 'r') as f:
        labels = np.copy(f[keyLabel])
        vals = np.copy(f[keyPID])

    print(labels.shape, vals.shape)

    # Currently expect shape (nTriplets, 4 pid types)
    assert vals.ndim == 2
    assert labels.ndim == 2
    assert vals.shape[0] == labels.shape[0]
    assert vals.shape[1] == 18  # I thought there were 21?
    assert labels.shape[1] == 4

    return labels, vals


def get_pid_labels(ndim):
    return [str(k) for k in multivariate_pid_key(ndim)]


def violin_pid_type(ax, data, xticks=True, ylim=None, zeroLine=False):
    labels = get_pid_labels(4)
    df = pd.DataFrame(data, columns=labels)
    sns.violinplot(ax=ax, data=df, scale='width')
    if xticks:
        ax.tick_params(axis='x', rotation=90)
    else:
        ax.set_xticks([])

    if ylim is not None:
        ax.set_ylim(ylim)

    if zeroLine:
        ax.axhline(y=0, linestyle='--', color='pink')
