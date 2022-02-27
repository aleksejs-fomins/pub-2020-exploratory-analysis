import os, time
import numpy as np
from os.path import join, isfile
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io as spio


#############################
#  Matlab
#############################

def loadmat(filename, waitRetry=None):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''

    # Test if file is accessible, and retry indefinitely if required
    fileAccessible = os.path.isfile(filename)
    if not fileAccessible:
        if waitRetry is None:
            raise ValueError("Matlab file can not be accessed", filename)
        else:
            while not fileAccessible:
                print("... can't reach file", filename, ", waiting", waitRetry, "seconds")
                time.sleep(waitRetry)
                fileAccessible = os.path.isfile(filename)

    # Load data
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)

    # Get rid of useless keys
    data = {k: v for k, v in data.items() if k[0] != '_'}

    return _check_keys(data)


def _check_keys(d):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in d:
        if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key])
    return d


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            d[strg] = _todict(elem)
        else:
            d[strg] = elem
    return dict


#############################
#  Auxiliary matplotlib
#############################

def rgb_change_color(img, c1, c2):
    rez = img.copy()
    r,g,b = img.T
    white_areas = (r == c1[0]) & (b == c1[1]) & (g == c1[2])
    rez[white_areas.T] = c2
    return rez


def base_colors_rgb(key='base'):
    if key == 'base':
        colorDict = colors.BASE_COLORS
    elif key == 'tableau':
        colorDict = colors.TABLEAU_COLORS
    elif key == 'css4':
        colorDict = colors.CSS4_COLORS
    elif key == 'xkcd':
        colorDict = colors.CSS4_COLORS
    else:
        raise ValueError('Unknown color scheme')

    return [colors.to_rgb(v) for c, v in colorDict.items()]


def sample_cmap(cmap, arr, vmin=None, vmax=None, dropAlpha=False):
    arrTmp = np.array(arr)

    # Test if samples in correct range
    if vmin is None:
        vmin = np.min(arrTmp)

    if vmax is None:
        vmax = np.max(arrTmp)

    arrNorm = (arrTmp - vmin) / (vmax - vmin)
    arrNorm = np.clip(arrNorm, 0, 1)

    cmapFunc = plt.get_cmap(cmap)
    rez = [cmapFunc(elNorm) for elNorm in arrNorm]
    if not dropAlpha:
        return rez
    else:
        return [r[:3] for r in rez]


# Add colorbar to existing imshow
def imshow_add_color_bar(fig, ax, img):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(img, cax=cax, orientation='vertical')


def imshow(fig, ax, data, xlabel=None, ylabel=None, title=None, haveColorBar=False, limits=None, extent=None,
           xTicks=None, yTicks=None, haveTicks=False, cmap=None, aspect='auto', fontsize=20):
    img = ax.imshow(data, cmap=cmap, extent=extent, aspect=aspect)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    if haveColorBar:
        imshow_add_color_bar(fig, ax, img)
    if not haveTicks:
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
    if limits is not None:
        norm = colors.Normalize(vmin=limits[0], vmax=limits[1])
        img.set_norm(norm)
    if xTicks is not None:
        ax.set_xticks(np.arange(len(xTicks)))
        ax.set_xticklabels(xTicks)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    if yTicks is not None:
        ax.set_yticks(np.arange(len(yTicks)))
        ax.set_yticklabels(yTicks)
    return img


def plt_add_fake_legend(ax, colors, labels, loc=None, bbox_to_anchor=None):
    handles = [mpatches.Patch(color=c, label=l) for c,l in zip(colors, labels)]
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=bbox_to_anchor)

#############################
#  Actual functions
#############################

def load_matrix(path=''):
    # Read the matrix file
    labelFileName = join(path, "L_modified.mat")
    if not isfile(labelFileName):
        raise ValueError("Can't find file", labelFileName)

    allenMap = loadmat(labelFileName)['L']                           # 2D map of cortical regions
    allenIndices = sorted(list(set(allenMap.flatten())))[2:]         # Indices of regions. Drop First two
    return allenMap, allenIndices


def plot_area_values(fig, ax, allenMap, allenIndices, valLst, vmin=None, vmax=None, cmap='jet', haveColorBar=True):
    # Mapping values to colors
    vmin = vmin if vmin is not None else np.min(valLst) * 0.9
    vmax = vmax if vmax is not None else np.max(valLst) * 1.1
    colors = sample_cmap(cmap, valLst, vmin, vmax, dropAlpha=True)

    trgShape = allenMap.shape + (3,)
    rez = np.zeros(trgShape)

    imBinary = allenMap == 0
    imColor = np.outer(imBinary.astype(float), np.array([0.5, 0.5, 0.5])).reshape(trgShape)
    rez += imColor

    for iROI, color in enumerate(colors):
        if not np.any(np.isnan(color)):
            imBinary = allenMap == allenIndices[iROI]

            imColor = np.outer(imBinary.astype(float), color).reshape(trgShape)
            rez += imColor

    rez = rgb_change_color(rez, [0, 0, 0], np.array([255, 255, 255]))
    imshow(fig, ax, rez, haveColorBar=haveColorBar, limits=(vmin, vmax), cmap=cmap)


def plot_area_clusters(fig, ax, allenMap, allenIndices, regDict, haveLegend=False, haveColorBar=True):
    trgShape = allenMap.shape + (3,)
    colors = base_colors_rgb('tableau')
    rez = np.zeros(trgShape)

    imBinary = allenMap == 0
    imColor = np.outer(imBinary.astype(float), np.array([0.5, 0.5, 0.5])).reshape(trgShape)
    rez += imColor

    for iGroup, (label, lst) in enumerate(regDict.items()):
        for iROI in lst:
            imBinary = allenMap == allenIndices[iROI]
            imColor = np.outer(imBinary.astype(float), colors[iGroup]).reshape(trgShape)
            rez += imColor

    imshow(fig, ax, rez, haveColorBar=haveColorBar)
    if haveLegend:
        plt_add_fake_legend(ax, colors[:len(regDict)], list(regDict.keys()))
