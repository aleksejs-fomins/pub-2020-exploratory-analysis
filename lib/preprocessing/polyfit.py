import numpy as np


def poly_fit_transform(x, y, ord):
    coeff = np.polyfit(x, y, ord)
    p = np.poly1d(coeff)
    return p(x)


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
