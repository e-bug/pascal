#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

src=en
tgt=tr
PROJ=$HOME/pascal
INPUT=$PROJ/data/wmt18tren/corpus
OUTPUT=$INPUT/wmt18${src}2${tgt}

# activate environment
source activate pascal

# Binarize the dataset:
cd $PROJ/fairseq
python preprocess.py \
	--source-lang $src \
	--target-lang $tgt \
	--trainpref $INPUT/train.tok.tok.bpe.16000 \
	--validpref $INPUT/valid.tok.tok.bpe.16000 \
	--testpref $INPUT/test.tok.tok.bpe.16000 \
	--destdir $OUTPUT \
	--workers 16 \
	--nwordssrc 16384 \
	--nwordstgt 16384 \
	--joined-dictionary

# deactivate environment
conda deactivate
