import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from mesostat.visualization.mpl_font import update_fonts_axis


def plot_snr_violins_bymouse(dataDB, ds, dataName='autocorr_d1'):
    resultDF = ds.list_dsets_pd()

    df = pd.DataFrame()
    for iMouse, mousename in enumerate(sorted(dataDB.mice)):
        # Get data
        queryDict = {"mousename" : mousename, "name" : dataName}
        dataRP, attrs = ds.get_data_recent_by_query(queryDict, listDF=resultDF)
        assert dataRP.ndim == 2

        # Pile data into dataframe
        dfThis = pd.DataFrame()
        dfThis['AC1'] = dataRP.flatten()
        dfThis['mouse'] = mousename
        df = df.append(dfThis)

    # Plot violins by mouse
    fig, ax = plt.subplots(figsize=(4,4))
    sns.violinplot(ax=ax, data=df, x='mouse', y='AC1', cut=0)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('1-step autocorrelation')
    update_fonts_axis(ax, 12)