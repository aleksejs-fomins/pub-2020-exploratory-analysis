import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


def test_multichannel(dataRP1, dataRP2, chIdxLst):
    rez = []
    for chIdxs in chIdxLst:
        dataRPgroup1 = dataRP1[:, chIdxs]
        dataRPgroup2 = dataRP2[:, chIdxs]
        rez += [svc_test_accuracy(dataRPgroup1, dataRPgroup2)]
    return rez


def svc_test_accuracy(dataRP1, dataRP2):
    X = np.concatenate([dataRP1, dataRP2], axis=0)
    y = np.array([0]*len(dataRP1) + [1]*len(dataRP2))

    clf = LinearSVC()
    scores = cross_val_score(clf, X, y, cv=10)
    # print(scores)
    return np.mean(scores)

data1 = np.random.normal(0, 1, (100, 3))
data2 = np.random.normal(1, 1, (100, 3))

for i in range(10):
    print(svc_test_accuracy(data1, data2))