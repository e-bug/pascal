#!/usr/bin/python
"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       9/4/2019
Date last modified: 9/4/2019
"""

import re
import sys
import html
import numpy as np
import multiprocessing
from nltk.parse.corenlp import CoreNLPParser
from langdetect import DetectorFactory 
from langdetect import detect_langs
DetectorFactory.seed = 0


global ende_ixs
parsers = {'en': CoreNLPParser(url='http://localhost:6666'),
           'de': CoreNLPParser(url='http://localhost:6667')}


def tokenize(sent, lang):
    sent = html.unescape(sent.strip())
    sent = re.sub('([\.!?])([A-Z])', '\g<1> \g<2>', sent)
    sent = ' '.join(parsers[lang].tokenize(sent)) 
    return sent


def detect_languages(sentence):
    try:
        langs = detect_langs(sentence)
        return [(l.lang, l.prob) for l in langs]
    except:
        return [('UNK', 1.0)]


def initializer(ixs):
    global ende_ixs
    ende_ixs = ixs


def parallel_en(tup):
    if tup[0] in ende_ixs:
        return tokenize(tup[1], 'en')


def parallel_de(tup):
    if tup[0] in ende_ixs:
        return tokenize(tup[1], 'de')


if __name__ == "__main__":
    path = sys.argv[1]

    pool = multiprocessing.Pool()

    lang = 'en'
    with open(path + '.' + lang) as f:
        en_lines = f.readlines()
    langs = pool.map(detect_languages, en_lines)
    en_ixs = [ix for ix, ts in enumerate(langs) if ts[0][0] == lang]

    lang = 'de'
    with open(path + '.' + lang) as f:
        de_lines = f.readlines()
    langs = pool.map(detect_languages, de_lines)
    de_ixs = [ix for ix, ts in enumerate(langs) if ts[0][0] == lang]

    pool.close()
    ende_ixs = set(en_ixs).intersection(set(de_ixs))

    pool = multiprocessing.Pool(multiprocessing.cpu_count(), initializer(ende_ixs))
    en_filt = pool.map(parallel_en, list(enumerate(en_lines)))
    de_filt = pool.map(parallel_de, list(enumerate(de_lines)))
    pool.close()

    none_arr = np.array(None)
    en_arr = np.array(en_filt)
    de_arr = np.array(de_filt)
    en_filt = en_arr[en_arr != none_arr]
    de_filt = de_arr[de_arr != none_arr]

    with open(path + '.filt.en', 'w') as f:
        for line in en_filt:
            f.write(line + '\n')

    with open(path + '.filt.de', 'w') as f:
        for line in de_filt:
            f.write(line + '\n')

