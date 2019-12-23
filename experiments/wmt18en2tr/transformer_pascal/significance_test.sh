#!/bin/bash

REF="$HOME/.sacrebleu/wmt17/en-de.de"
OURS="outputs/preds.test"
BASE="../transformer/outputs/preds.test"
TOK="13a"
script="$HOME/pascal/tools/significance-tests/significance_bleu.py"

source activate pascal

python $script $REF $OURS $BASE --tok $TOK > significance.test

conda deactivate
