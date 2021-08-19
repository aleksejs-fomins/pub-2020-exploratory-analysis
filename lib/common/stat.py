import numpy

from mesostat.stat.testing.htests import classification_accuracy_weighted, rstest_twosided


def test_metric_by_name(metricName):
    if metricName == 'accuracy':
        return classification_accuracy_weighted
    elif metricName == 'nlog_pval':
        return lambda x,y: -np.log10(rstest_twosided(x, y))[1]
    else:
        raise ValueError('Unexpected metric name', metricName)