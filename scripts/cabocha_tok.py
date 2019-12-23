#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by:         Emanuele Bugliarello (@e-bug)
Date created:       9/4/2019
Date last modified: 9/4/2019
"""

import sys
import numpy as np
from bs4 import BeautifulSoup


def tok_annotate(sentence):
    tok_sent = []
    dep_sent = []
    cnk_sent = []
    for ic, chunk in enumerate(sentence.find_all('chunk')):
        for it, tok in enumerate(chunk.find_all('tok')):
            tok_sent.append(tok.text)
            dep_sent.append(chunk.attrs['link'] if chunk.attrs['link'] != '-1' else chunk.attrs['id'])
            cnk_sent.append(tok.text)
        cnk_sent.append(' ')
    return ' '.join(tok_sent), ' '.join(dep_sent), ''.join(cnk_sent[:-1])

if __name__ == "__main__":
    filename = sys.argv[1]
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    soup = BeautifulSoup(data, features="html.parser")
    cb_sentences = soup.find_all('sentence')

    res = list(map(tok_annotate, cb_sentences))

    basename = filename[:-2]
    with open(basename + 'cabocha.tok', 'w') as f:
        for tok_sent, dep_sent, cnk_sent in res:
            f.write(tok_sent + '\n')
    with open(basename + 'cabocha.dep', 'w') as f:
        for tok_sent, dep_sent, cnk_sent in res:
            f.write(dep_sent + '\n')
    with open(basename + 'cabocha.cnk', 'w') as f:
        for tok_sent, dep_sent, cnk_sent in res:
            f.write(cnk_sent + '\n')

