#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

PROJDIR=$HOME/pascal
INDIR=$PROJDIR/data/wat18jaen/corpus
OUTDIR=$PROJDIR/data/wat18jaen/lin_parses
lang=en
TRAIN=train.tok.tok.bpe.32000.$lang
VALID=dev.tok.tok.bpe.32000.$lang
TEST=test.tok.tok.bpe.32000.$lang
SCRIPTSDIR=$PROJDIR/scripts

source activate pascal

mkdir -p $OUTDIR

size=300000
i=0

python $SCRIPTSDIR/bpe_lin_parses.py $lang $INDIR/$TEST $OUTDIR/test.$lang $size $i &
python $SCRIPTSDIR/bpe_lin_parses.py $lang $INDIR/$VALID $OUTDIR/valid.$lang $size $i &

for i in {0..10}; do
  python $SCRIPTSDIR/bpe_lin_parses.py $lang $INDIR/$TRAIN $OUTDIR/train.$lang.$i $size $i &
done
wait

for i in {0..10}; do
  cat $OUTDIR/train.$lang.$i >> $OUTDIR/train.$lang
done
rm $OUTDIR/train.$lang.*

conda deactivate
