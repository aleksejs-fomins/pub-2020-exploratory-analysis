import numpy as np

from mesostat.stat.machinelearning import drop_nan_rows


def poly_fit_transform(x, y, ord):
    xEff, yEff = drop_nan_rows([x, y])
    coeff = np.polyfit(xEff, yEff, ord)   # Fit to data without nans
    p = np.poly1d(coeff)
    return p(x)                           # Evaluate for original data


def poly_fit_discrete_residual(y, ord):
    xFake = np.arange(len(y))
    return y - poly_fit_transform(xFake, y, ord)


def poly_fit_discrete_parameter_selection(y, ordMax=5):
    relResidualNorms = []
    normOld = np.linalg.norm(y)
    for ord in range(0, ordMax+1):
        # yfit = poly_fit_transform(np.arange(len(y)), y, ord)
        # relResidualNorms += [r2_score(y, yfit)]

        resThis = poly_fit_discrete_residual(y, ord)
        normThis = np.linalg.norm(resThis)
        relResidualNorms += [1 - normThis / normOld]
        normOld = normThis

    return relResidualNorms
