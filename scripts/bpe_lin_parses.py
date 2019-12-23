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
from collections import defaultdict
from nltk.parse.corenlp import CoreNLPParser

mapper = {'"': '``'}
parsers = {'en': CoreNLPParser(url='http://localhost:6666'),
           'de': CoreNLPParser(url='http://localhost:6667')}


def tokenize(sent):
    tokens = html.unescape(sent.strip()).split()
    tokens = list(map(lambda t: mapper.get(t, t), tokens))
    return tokens


def remove_bpe(sent, separator='@@'):
    tokens = tokenize(sent)
    word, words = [], []
    for tok in tokens:
        if tok.endswith(separator):
            tok = tok.strip(separator)
            word.append(tok)
        else:
            word.append(tok)
            words.append(''.join(word))
            word = []
    sentence = ' '.join(words)
    return sentence


def annotate(sentences):

    def fix_tok(tok):
        tok = tok.replace("n't", "not")
        tok = tok.replace("&apos;", "'")
        tok = re.sub("'[Nn]'", "and", tok)
        tok = re.sub("'([A-Za-ce-rt-z])", "\g<1>", tok)
        return tok

    def get_parse(sent):
        p = list(dep_parser.parse(sent.split()))[0]
        leaves = set(p.leaves())
        agg = ''
        for st in str(p).split('\n'):
            st = st.strip()
            st = st.replace(')', ' )')
            splits = st.split()
            ixs = np.argwhere([e in leaves for e in splits]).flatten() 
            splits = st.split()

            if len(ixs) == 1 and ixs[0] == 0:
                # Unwanted inter-line split
                try:
                    splits.pop(1)
                    splits.pop(0)
                    agg = ' '.join(agg.split()[:-1])
                except:
                    pass
            else:
                for ix in ixs[::-1]:
                    if ix+1 not in ixs:
                        splits.pop(ix+1)
                    splits.pop(ix)
                    if ix-1 not in ixs:
                        splits.pop(ix-1)
            agg += ' ' + ' '.join(splits)

        return agg.strip()

    word_tags = []

    for ix, sent in enumerate(sentences):
        sent_token_tags = []
        splits = list(map(fix_tok, sent.split()))
        parse_sent = ' '.join(splits)
        dep_sent = get_parse(parse_sent)
        sent_token_tags.append(dep_sent)
        dep_len = len(dep_sent)
        
        while dep_len < len(splits):
            # Multiple sentences detected
            parse_sent = ' '.join(splits[dep_len:])
            dep_sent = get_parse(parse_sent)
            sent_token_tags.append(dep_sent)
            dep_len += len(dep_sent)

        word_tags.append(' '.join(sent_token_tags))

    return word_tags


if __name__ == "__main__":
    lang = sys.argv[1]
    infile = sys.argv[2]
    outfile = sys.argv[3]
    size = int(sys.argv[4])
    i = int(sys.argv[5])

    dep_parser = parsers[lang]

    # Read BPE-formatted sentences
    with open(infile, 'r', encoding='utf-8') as f:
        bpe_sents = f.readlines()
    bpe_sents = bpe_sents[i*size: min(len(bpe_sents),(i+1)*size)]

    # Attempt reconstructing original data
    recovered_sents = list(map(remove_bpe, bpe_sents))

    # Annotate tokens from reconstructed words
    parses = annotate(recovered_sents)

    # Write tags to file
    with open(outfile, 'w', encoding='utf-8') as f:
        for tags in parses:
            f.write(tags + '\n')

