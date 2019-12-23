#!/bin/bash

PROJDIR=$HOME/pascal/experiments/wmt16en2de
PREDFN=$PROJDIR/transformer_pascal/outputs/preds.test
BASEFN=$PROJDIR/transformer/outputs/preds.test
SRCFN=$HOME/.sacrebleu/wmt17/en-de.en
TRUTHFN=$HOME/.sacrebleu/wmt17/en-de.de
TOK=13a

script=$HOME/pascal/tools/len2bleu.py

source activate pascal

python ${script} ${PREDFN} ${BASEFN} ${SRCFN} ${TRUTHFN} ${TOK} > delta_bleu.out

conda deactivate
