import string

def findMaxStringP(text, type, skip=0, skip_chars=''):
    i, nskip = 0, 0
    while (i < len(text) and nskip <= skip):
        if (type == 'ascii'):
            if (text[i] not in skip_chars):
                if (text[i] in string.digits + string.punctuation):
                    nskip += 1
                elif (text[i] in string.ascii_letters):
                    nskip = 0
        elif (type == 'digit'):
            if (text[i] not in skip_chars):
                if (text[i] in string.ascii_letters + string.punctuation):
                    nskip += 1
                elif (text[i] in string.digits):
                    nskip = 0
        elif (type == 'punctuation'):
            None
        i += 1
    if nskip > skip:
        return text[:i-1].strip()
    elif (i == len(text)):
        return text.strip()

    return text[:i].strip()

def findMaxString(text, skip=0, skip_chars=''):
    tc, td = '', ''
    c = 0
    try:
        for i in range(len(text)):
            if text[i] in string.digits:
                t = findMaxStringP(text[i:], 'digit', skip, skip_chars)
                if (c > 0):
                    t = text[i-c:i] + t
                    c = 0
                if (len(td) < len(t)):
                    td = t

            elif text[i] in string.ascii_letters:
                t = findMaxStringP(text[i:], 'ascii', skip, skip_chars)
                if (c > 0):
                    t = text[i-c:i] + t
                    c = 0
                if (len(tc) < len(t)):
                    tc = t

            elif (text[i] in skip_chars): # text[i] in string.punctuation and
                # t = findMaxStringP(text[i:], 'punctuation', skip, split_chars)
                c += 1
            else:
                c = 0

    except ValueError:
        None

    return tc, td

# Matching Name/Address/Phone Term
def pctMatchingKeyword(text, termSet):
    tmp = removeDuplicate([term for term in termSet if ((' ' + text + ' ').find(' ' + term + ' ') >= 0)])
    return len(tmp)/len(termSet)

# Matching Street/Ward/District/City Term
def pctMatchingTerm(text, termSet):
    tmp = [term for term in termSet if (text.find(term) >= 0)]
    return len(tmp)/len(termSet)

def removeDuplicate(termList):
    t = []
    for i in range(len(termList)):
        b = True
        for j in range(len(t)):
            if ((' ' + t[j] + ' ').find(' ' + termList[i] + ' ') >= 0):
                b = False
                break
            elif ((' ' + termList[i] + ' ').find(' ' + t[j] + ' ') >= 0):
                t[j] = termList[i]
                b = False
                break
        if (b):
            t.append(termList[i])

    return t

def compareModelConfig(m1, m2):
    return (m1.learning_rate == m2.learning_rate
            and m1.learning_rule == m2.learning_rule
            and m1.n_iter == m2.n_iter)

def getGroupModels(models, model_configs):
    dct = {} # {'group': {config: {}, modelname: {key: modelclf}} }

    for key, config in model_configs.items():
        flag = True
        for k, mi in dct.items():
            if (config==mi['config']):
                dct[k]['models'][key] = models[key]
                flag = False
                break
        if flag:
            dct[dct.__len__()] = {'config': config, 'models': {key: models[key]}}

    return dct
