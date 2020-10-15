import numpy as np

from mesostat.utils.system import date_diff

# Performance is true positive + true negative frequency
def mouse_performance_single_session(nTrial, behaviour):
    nTrialGO = len(behaviour['iGO'])
    nTrialNOGO = len(behaviour['iNOGO'])
    perf = (nTrialGO + nTrialNOGO) / nTrial
    return perf


# Mark days as naive or expert based on performance threshold
def mouse_performance_allsessions(datesLst, perfLst, pTHR):
    nDays = len(datesLst)

    # Find the index of the first expert session
    # First expert session is the first session that surpasses performance of pTHR
    # In case no expert sessions exist, point at next element outside array
    candidates = np.where(perfLst > pTHR)[0]
    expertThrIdx = candidates[0] if len(candidates) > 0 else nDays

    # Decide which trials are expert/naive
    isExpert = np.full(nDays, True)
    isExpert[:expertThrIdx] = False

    # Calculate by how many days should performance centered days be shifted to match naive-expert threshold
    # In case no exper sessions exist, shift by last naive day + 1
    deltaDays = date_diff(datesLst, datesLst[0])
    deltaDaysThr = deltaDays[expertThrIdx] if expertThrIdx < nDays else deltaDays[-1] + 1
    deltaDaysCentered = deltaDays - deltaDaysThr

    # There is only a threshold if mouse reached expert level at all
    expertThrIdxEff = expertThrIdx if expertThrIdx < nDays else None

    return expertThrIdxEff, isExpert, deltaDays, deltaDaysCentered