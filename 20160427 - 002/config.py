import json

def readFile(file):
    strList = []
    infile = open(file, encoding="utf-8")
    for line in infile:
        strList.append(line)
    return strList

def loadJson(file):
    try:
        return json.loads(''.join(readFile(file)))
    except:
        return None

# ========================================

nRun = 5

folders = {
    'data-source': '0. Data-source',
    'data': '1. Data',          # name, address, phone, nothing (text)
    'data-test': '2. Test',     #
    'feature': '3. Features',   # training, testing data (csv)
    'model': '4. Models',       #
    'result': '5. Results',
    'log': '6. Logs'
}

files = {
    'data-source': { 'dictionary': 'dictionary.json' },
    'data': {'name': 'name.txt', 'address': 'address.txt', 'phone': 'phone.txt', 'nothing': 'nothing.txt',
             'train-test-name': ['train_name.txt', 'test_name.txt'],
             'train-test-address': ['train_address.txt', 'test_address.txt'],
             'train-test-phone': ['train_phone.txt', 'test_phone.txt'],
             'train-test-nothing': ['train_nothing.txt', 'test_nothing.txt']},
    'feature': {
        'name': {'train': 'train-name.csv', 'test': 'test-name.csv'},
        'address': {'train': 'train-address.csv', 'test': 'test-address.csv'},
        'phone': {'train': 'train-phone.csv', 'test': 'test-phone.csv'}
    },
    'data-test': {'term': 'termdata.csv', 'full-address': 'full-address.csv', 'other': ''},
    'model': {'name': 'model_name.pkl', 'address': 'model_address.pkl', 'phone': 'model_phone.pkl',
              'model-config': 'model-config.json'},
    'result': {'term': 'term_result.xlsx', 'segment': 'segment.xlsx', 'histogram': 'segment.png',
               'model-test-result': 'model-test.xlsx'},
    'log': 'logs.xlsx'
}


tmp = loadJson(folders['data-source'] + '/' + files['data-source']['dictionary'])
nameTermSet = tmp['name-term-set']
addressTermSet = tmp['address-term-set']
phoneTermSet = tmp['phone-term-set']

streetTermSet = tmp['street-term-set']
wardDistrictTermSet = tmp['ward-district-term-set']
cityTermSet = tmp['city-term-set']

asi = tmp['ascii']
unic = tmp['unicode']
upchars = tmp['upper-characters']


testSize = 0.3

# classLabels = {'name': [1, 0, 0, 0], 'address': [0, 1, 0, 0], 'phone': [0, 0, 1, 0], 'nothing': [0, 0, 0, 1]}
classLabels = {'name': 0, 'address': 1, 'phone': 2, 'nothing': 3}

model_test = None

model_configs = {
    'name-model': {
        'type': 'Neural-Network',
        'layers': [(10, 'Sigmoid'), (2, 'Softmax')],
        'learning_rate': 0.01,
        'learning_rule': 'adagrad',
        'n_iter': 1000
    },
    'address-model': {
        'type': 'Neural-Network',
        'layers': [(10, 'Sigmoid'), (2, 'Softmax')],
        'learning_rate': 0.01,
        'learning_rule': 'adagrad',
        'n_iter': 1000
    },
    'phone-model': {
        'type': 'Neural-Network',
        'layers': [(10, 'Sigmoid'), (2, 'Softmax')],
        'learning_rate': 0.01,
        'learning_rule': 'adagrad',
        'n_iter': 1000
    }
}
featureConfig = {
    'name': [
        # ('length', True),
        # (['', '', ''], False)

        ('length', False),

        ('#ascii', False),
        ('#digit', False),
        ('#punctuation', False),

        ('%ascii-adp', True),
        ('%digit-adp', True),
        ('%punctuation-adp', False),

        ('digit-adp/ascii-adp', True),

        ('%ascii', False),
        ('%digit', False),
        ('%punctuation', False),

        ('%keyword-name', True),
        ('%keyword-address', True),
        ('%keyword-phone', True),

        ('#max-digit-skip-all-punctuation', False),
        ('b#max-digit-skip-all-punctuation >= 7', True),
        ('%max-digit-skip-all-punctuation', False),
        ('#max-digit-skip-space-&-dot', False),
        ('%max-digit-skip-space-&-dot', False),

        ('bfirst-character-digit', True),
        ('bfirst-character-ascii', True),
        ('blast-character-digit', True),
        ('blast-character-ascii', True),

        ('b#ascii >= 6', True),
        ('b#digit >= 7', False)
    ],
    'address': [
        # ('%ascii-adp', True),
        # (['', '', ''], False)

        ('length', False),

        ('#ascii', False),
        ('#digit', False),
        ('#punctuation', False),

        ('%ascii-adp', True),
        ('%digit-adp', True),
        ('%punctuation-adp', False),

        ('digit-adp/ascii-adp', True),

        ('%ascii', False),
        ('%digit', False),
        ('%punctuation', False),

        ('%keyword-name', True),
        ('%keyword-address', True),
        ('%keyword-phone', True),

        ('%street-term', False),
        ('%ward-district-term', False),
        ('%city-term', False),

        ('#max-digit-skip-all-punctuation', False),
        ('b#max-digit-skip-all-punctuation >= 7', True),
        ('%max-digit-skip-all-punctuation', False),
        ('#max-digit-skip-space-&-dot', False),
        ('%max-digit-skip-space-&-dot', False),

        ('bfirst-character-digit', True),
        ('bfirst-character-ascii', True),
        ('blast-character-digit', False),
        ('blast-character-ascii', False),

        ('b#ascii >= 6', False),
        ('b#digit >= 7', False),

        ('b/', True)

    ],
    'phone': [
        # ('%digit-adp', True),
        # (['', '', ''], False)

        ('length', False),

        ('#ascii', False),
        ('#digit', False),
        ('#punctuation', False),

        ('%ascii-adp', True),
        ('%digit-adp', True),
        ('%punctuation-adp', False),

        ('digit-adp/ascii-adp', True),

        ('%ascii', False),
        ('%digit', False),
        ('%punctuation', False),

        ('%keyword-name', True),
        ('%keyword-address', True),
        ('%keyword-phone', True),

        ('#max-digit-skip-all-punctuation', False),
        ('b#max-digit-skip-all-punctuation >= 7', True),
        ('%max-digit-skip-all-punctuation', False),
        ('#max-digit-skip-space-&-dot', False),
        ('%max-digit-skip-space-&-dot', False),

        ('bfirst-character-digit', True),
        ('bfirst-character-ascii', True),
        ('blast-character-digit', True),
        ('blast-character-ascii', True),

        ('b#ascii >= 6', False),
        ('b#digit >= 7', True)
    ]
}

class TimeManage(object):
    def __init__(self):
        self.time = ''

    def setTime(self, _time):
        self.time = _time

    def getTime(self):
        return self.time

timeManager = TimeManage()


skip_punctuation = ' .'
rm_preprocessed_punctuation = """ ,"""
