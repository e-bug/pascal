#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

src=en
tgt=de
PROJ=$HOME/pascal
INPUT=$PROJ/data/wmt16deen/corpus
OUTPUT=$INPUT/wmt16${src}2${tgt}

# activate environment
source activate pascal

# Binarize the dataset:
cd $PROJ/fairseq
python preprocess.py \
	--source-lang $src \
	--target-lang $tgt \
	--trainpref $INPUT/train.tok.clean.tok.bpe.32000 \
	--validpref $INPUT/valid.tok.tok.bpe.32000 \
	--testpref $INPUT/test.tok.tok.bpe.32000 \
	--destdir $OUTPUT \
	--workers 16 \
	--joined-dictionary

# deactivate environment
conda deactivate
