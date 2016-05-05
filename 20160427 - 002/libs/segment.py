import itertools as it
import string

def segmentText(text):
    terms = text.strip(' \n,.').split()
    return [
        [' '.join(terms[:cb[0]+ 1]), ' '.join(terms[cb[0]+1:cb[1]+1]), ' '.join(terms[cb[1]+1:])]
        for cb in list(it.combinations(range(len(terms)-1), 2))
    ]