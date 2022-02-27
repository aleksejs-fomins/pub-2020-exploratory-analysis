import h5py
import numpy as np
import pandas as pd

from mesostat.utils.arrays import unique_subtract
from mesostat.utils.pandas_helper import outer_product_df, pd_query, pd_is_one_row


def _parse_key(key):
    lst = key.split('_')
    rez = {
        'metric': lst[0],
        'mousename': '_'.join(lst[1:3]),
        'intervName': lst[3],
        'datatype': '_'.join(lst[4:6]),
        'trialType': lst[6]
    }

    if len(lst) == 7:
        return rez
    elif len(lst) == 8:
        rez['performance'] = lst[7]
        return rez
    else:
        raise ValueError('Unexpected key', key)


def summary_df(h5fname):
    with h5py.File(h5fname, 'r') as f:
        if 'lock' in f.keys() and len(f['lock'].keys()) > 0:
            print("Warning: Non-zero lock", f['lock'].keys())

        keys = set(f.keys()) - {'lock'}

    # Only keep value keys, ignore labels for now
    keys = [key for key in keys if key[:5] != 'Label']

    summaryDF = pd.DataFrame()
    for key in keys:
        summaryDF = summaryDF.append(pd.DataFrame({**{'key': key}, **_parse_key(key)}, index=[0]))

    sortValues = unique_subtract(list(summaryDF.columns), ['key'])
    return summaryDF.reset_index(drop=True).sort_values(sortValues)


def summary_update_data_sizes(dfSummary, dataDB):
    rezLst = []
    for idx, row in dfSummary.drop(['key', 'metric'], axis=1).iterrows():
        mousename = row['mousename']
        queryDict = dict(row)
        del queryDict['mousename']
        if 'trialType' in queryDict and queryDict['trialType'] == 'None':
            del queryDict['trialType']

        dataRSPLst = dataDB.get_neuro_data({'mousename': mousename}, **queryDict)
        dataRSP = np.concatenate(dataRSPLst, axis=0)
        rezLst += [dataRSP.shape[0]]

    rezDF = dfSummary.copy()
    rezDF['nData'] = rezLst
    return rezDF

    # rezDict = {}
    # for vals, dfSub in dfSummary.groupby(['mousename', 'trialType']):
    #     mousename, trialType = vals
    #     trialType = trialType if trialType != 'None' else None
    #
    #     dataLst = dataDB.get_neuro_data({'mousename': mousename}, trialType=trialType)
    #     nTr = np.sum([data.shape[0] for data in dataLst])
    #     rezDict[vals] = nTr


def read_adversarial_distr_file(pwd):
    rez = {}
    with h5py.File(pwd, 'r') as h5f:
        for k in h5f.keys():
            ptModel, ptTrg, nData = k.split('_')
            nData = int(nData)
            ptTrg = ptTrg if ptTrg != 'unq' else 'unique'

            if ptTrg not in rez.keys():
                rez[ptTrg] = {}

            rez[ptTrg][nData] = np.array(h5f[k])

    return rez


# def test_summary(dfSummary):
#     colSet = set(dfSummary.columns) - {'key'}
#     colDict = {col: set(dfSummary[col]) for col in colSet}
#     colDF = outer_product_df(colDict)
#
#     print(len(colDF), len(dfSummary))
#     print(colDict)
#
#     assert len(colDF) == len(dfSummary)
#
#     for idx, row in colDF.iterrows():
#         assert pd_is_one_row(pd_query(dfSummary, dict(row)))[0] is not None
#
#     print("Consistency passed, parameters were")
#     print(colDict)


def read_computed_3D(h5fname, keyVals, pidType):
    keyLabels = 'Label_' + keyVals[4:]

    with h5py.File(h5fname, 'r') as f:
        labels = np.copy(f[keyLabels])
        vals = np.copy(f[keyVals])

    # Currently expect shape (nTriplets, 4 pid types)
    assert vals.ndim == 2
    assert labels.ndim == 2
    assert vals.shape[0] == labels.shape[0]
    assert vals.shape[1] == 4
    assert labels.shape[1] == 3

    # Drop negatives
    vals = np.clip(vals, 0, None)

    if pidType == 'unique':
        valsU1 = vals[:, 0]
        valsU2 = vals[:, 1]
        labelsU1 = labels
        labelsU2 = labels[:, [1,0,2]]  # Swap sources, keep target
        return np.concatenate([labelsU1, labelsU2], axis=0), np.concatenate([valsU1, valsU2], axis=0)
    elif pidType == 'red':
        return labels, vals[:, 2]
    elif pidType == 'syn':
        return labels, vals[:, 3]
    else:
        raise ValueError(pidType)


def list_to_3Dmat(idxs, vals, nChannel):
    rezMat = np.full((nChannel, nChannel, nChannel), np.nan)
    rezMat[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = vals
    return rezMat