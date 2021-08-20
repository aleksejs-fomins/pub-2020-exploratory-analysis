from mesostat.utils.pandas_helper import outer_product_df, drop_rows_byquery


def _auto_param(dataDB, paramName, autoAppendDict=None):
    if paramName == 'mousename':
        rez = sorted(dataDB.mice)
    elif paramName == 'datatype':
        rez =  dataDB.get_data_types()
    elif paramName == 'trialType':
        rez =  dataDB.get_trial_type_names()
    elif paramName == 'performance':
        rez =  dataDB.get_performance_names()
    elif paramName == 'intervName':
        rez =  dataDB.get_interval_names()
    else:
        raise ValueError('Unexpected auto keyword argument', paramName)

    # If the desired automatic response is broader than standart, it can be extended with this dictionary
    if (autoAppendDict is not None) and (paramName in autoAppendDict.keys()):
        rez += autoAppendDict[paramName]
    return rez


def param_vals_to_suffix(paramVals):
    if isinstance(paramVals, list) or isinstance(paramVals, tuple):
        return '_'.join([str(s) for s in paramVals])
    else:
        return str(paramVals)   # Note pandas.groupby does not return list if has only one parameter


def pd_row_to_kwargs(row, parseNone=False, dropKeys=None):
    rez = dict(row)
    if dropKeys is not None:
        for key in dropKeys:
            del rez[key]

    if parseNone:
        rez = {k: v if v != 'None' else None for k, v in rez.items()}

    return rez


class DataParameterSweep():
    def __init__(self, dataDB, exclQueryLst=None, autoAppendDict=None, **kwargs):
        # Assemble dictionary of possible parameter values for each parameter
        # If parameter list is keyword 'auto', its possible values get filled in automatically
        self.argSweepDict = {}
        for k, v in kwargs.items():
            if v is not None:
                self.argSweepDict[k] = v if v != 'auto' else _auto_param(dataDB, k, autoAppendDict=autoAppendDict)

        self.sweepDF = outer_product_df(self.argSweepDict)
        if exclQueryLst is not None:
            self.sweepDF = drop_rows_byquery(self.sweepDF, exclQueryLst)

    def param(self, paramName):
        return self.argSweepDict[paramName]

    def param_size(self, paramName):
        return len(self.argSweepDict[paramName])

    def param_index(self, paramName, paramVal):
        return self.argSweepDict[paramName].index(paramVal)

    def invert_param(self, paramNameExclLst):
        paramNameSet = set(list(self.argSweepDict.keys()))
        paramNameSetExcl = set(paramNameExclLst)
        return list(paramNameSet - paramNameSetExcl)
