import libs.store_manage as sm
import libs.models as md
from libs.features import getFeature, preprocess
import numpy as np
from libs.store_manage import saveTestStandardResults

groupModels = sm.loadAllModel()
terms = sm.loadTermTest()

delta_threshold = 0.01
dataProb = [['Type', '#', '#Model', 'Learning Rate', 'Learning Rule', 'N_Iter', 'Avg_Mean_Distance', 'Avg_Var_Distance',
         'alpha', 'H0: Avg_Mean_Distance = 0', 'P(distance < ' + str(delta_threshold) + ')',
         'p(Accept H0: mean_1 = mean_2)']]


data = {}

# ===================================================
for ttype, modelList in groupModels.items():
    X = np.asarray([getFeature([text[0]], ttype)[0] for text in terms['X']])

    for igroup, models in modelList.items():
        if (len(models['models']) > 1):
            results, probXs = md.checkModelConvergence(models, X, delta_threshold)

            # ==================================
            tmp00, tmp01 = [], []
            for jkey in probXs.keys():
                tmp00.append(jkey)
                tmp01.append('N_' + jkey)

            tmp0 = [tmp00 + [''] + tmp01]

            _f = lambda tmp, prob: np.append(tmp, prob, axis=1) if (tmp.shape[0] > 0) else prob

            tmp10, tmp11 = np.asarray([]), np.asarray([])
            for jkey, jprob in probXs.items():
                tmp10 = _f(tmp10, jprob[:, 0].reshape(jprob.shape[0], 1))
                tmp11 = _f(tmp11, jprob[:, 1].reshape(jprob.shape[0], 1))

            tmp1 = np.append(tmp10, np.append(tmp10.shape[0] * [['']], tmp11, axis=1),axis=1).tolist()

            data[ttype + '_' + str(igroup)] = tmp0 + tmp1



            # ==================================

            # tmp0, tmp = [], np.asarray([])
            # for jkey in probXs.keys():
            #     tmp0 += [jkey + '_' + ttype, jkey + '_N_' + ttype, '']
            # for jkey, jprob in probXs.items():
            #     if tmp.shape[0] > 0:
            #         tmp = np.append(tmp, np.append(jprob, [['']]*jprob.shape[0],axis=1), axis=1)
            #     else:
            #         tmp = np.append(jprob, [['']]*jprob.shape[0],axis=1)
            # data[ttype + '_' + str(igroup)] = [tmp0] + tmp.tolist()

            dataProb.append([
                ttype,
                igroup,
                len(models['models']),
                models['config']['learning_rate'],
                models['config']['learning_rule'],
                models['config']['n_iter'],
                results['mean-distance'],
                results['var-distance'],
                delta_threshold,
                results['mean == 0'],
                results['p(distance<alpha)'],
                results['mean_1 == mean_2']
            ])

saveTestStandardResults(data, dataProb)