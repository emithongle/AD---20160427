import execute.repair_data as pd
import execute.train_data as td
import execute.classify_data as cd
import execute.segment_data as sd
# from libs.segment import segmentText
import datetime
#
from config import timeManager

t = datetime.datetime.now()
timeManager.setTime(str(t.date().strftime('%Y%m%d')) + '_' + str(t.time().strftime('%H%M%S')))
timeManager.setTime('20160501_190053')

print('Repairing Data...')
# pd._exec()

print('Training Data...')
# td._exec()

print('Classify Term Data...')
cd._exec()

print('Segmenting Address Data...')
sd._exec()

#
# from libs.models import getModelInfo
# from config import model_configs
#
# a = getModelInfo(model_configs['name-model'])
None
