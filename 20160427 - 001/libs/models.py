from sknn.mlp import Classifier, Layer
from libs.features import *
import numpy as np

def getModelType():
    return 'Neural-Network'

def buildModel(modelConfig):
    return Classifier(
        layers=[Layer(modelConfig['layers'][i][1], units=modelConfig['layers'][i][0])
                for i in range(len(modelConfig['layers']))],
        learning_rule=modelConfig['learning_rule'],
        learning_rate=modelConfig['learning_rate'],
        n_iter=modelConfig['n_iter']
    )

def getModelInfoTxt(modelConfig):
    if (modelConfig['type'] == 'Neural-Network'):
        return str(len(modelConfig['layers'])) + ' layers: [' + \
                ', '.join([str(n_unit) + '-' + act_func for (n_unit, act_func) in modelConfig['layers']]) + \
                '], learning_rate: ' + str(modelConfig['learning_rate']) + ', learning_rule: ' + modelConfig['learning_rule'] + \
                ', n_iterator: ' + str(modelConfig['n_iter'])
    return ''

def getModelInfoJson(modelConfig):
    if (modelConfig['type'] == 'Neural-Network'):
        return {}
    return {}

def clasify(clfs, text):
    def _f(clf, text, ttype):
        return clf.predict_proba(getFeature([text], ttype))

    probName = _f(clfs['name'], text, 'name')[0]
    probAddress = _f(clfs['address'], text, 'address')[0]
    probPhone = _f(clfs['phone'], text, 'phone')[0]

    proba = [
        probName[0] * probAddress[1] * probPhone[1],
        probName[1] * probAddress[0] * probPhone[1],
        probName[1] * probAddress[1] * probPhone[0],
        probName[1] * probAddress[1] * probPhone[1]
    ]

    # _ = np.exp(proba) / np.sum(np.exp(proba))
    _ = np.asarray(proba) / np.sum(proba)
    return max(range(len(_)), key=_.__getitem__), _

def checkModelConvergence(models, X, threshold=0.001):

    ngroups = len(models['models'])
    probs = {key: np.asarray(clf.predict_proba(X)) for key, clf in models['models'].items()}

    distance, sumvar = 0, 0
    countLessAlpha, countNameLessAlpha, countAddressLessAlpha, countPhoneLessAlpha = 0, 0, 0, 0
    df = ((ngroups - 1) * ngroups / 2 * X.shape[0])

    keys = list(probs.keys())
    for i in range(len(probs)-1):
        for j in range(i+1, len(probs)):
            tmp = (probs[keys[i]][:, 0] - probs[keys[j]][:, 0])
            distance += tmp.sum()
            sumvar += (tmp ** 2).sum()
            # tmp = np.zeros(tmp.shape)
            countLessAlpha += abs(tmp)[abs(tmp) < threshold].shape[0]
    data = {
        'mean-distance': distance / df,
        'var-distance': sumvar / df,
        'mean == 0': '',
        'p(distance<alpha)': countLessAlpha / df,
        'mean_1 == mean_2': ''
    }

    return data, probs

def checkAccuracy(prob, label):
    return sum([x[i] for i, x in zip(label, prob)]) / prob.shape[0]