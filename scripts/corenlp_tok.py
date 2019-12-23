#!/usr/bin/python
"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       9/4/2019
Date last modified: 9/4/2019
"""

import re
import sys
import html
import multiprocessing
from nltk.parse.corenlp import CoreNLPParser


parsers = {'en': CoreNLPParser(url='http://localhost:6666'),
           'de': CoreNLPParser(url='http://localhost:6667')}


def tokenize(sent, lang):
    sent = html.unescape(sent.strip())
    sent = re.sub('([\.!?])([A-Z])', '\g<1> \g<2>', sent)
    sent = ' '.join(parsers[lang].tokenize(sent)) 
    return sent


def tokenize_en(sent):
    return tokenize(sent, 'en')


def tokenize_de(sent):
    return tokenize(sent, 'de')


if __name__ == "__main__":
    path = sys.argv[1]

    pool = multiprocessing.Pool()

    lang = 'en'
    with open(path + '.' + lang) as f:
        en_lines = f.readlines()

    lang = 'de'
    with open(path + '.' + lang) as f:
        de_lines = f.readlines()

    en_tok = pool.map(tokenize_en, en_lines)
    de_tok = pool.map(tokenize_de, de_lines)

    pool.close()

    with open(path + '.tok.en', 'w') as f:
        for line in en_tok:
            f.write(line + '\n')

    with open(path + '.tok.de', 'w') as f:
        for line in de_tok:
            f.write(line + '\n')

