#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

PROJDIR=$HOME/pascal
INDIR=$PROJDIR/data/wmt16deen/filt_corpus
OUTDIR=$PROJDIR/data/wmt16deen/filt_tags_mean
lang=en
TRAIN=train.tok.clean.filt.bpe.32000.$lang
VALID=valid.tok.tok.bpe.32000.$lang
TEST=test.tok.tok.bpe.32000.$lang
SCRIPTSDIR=$PROJDIR/scripts

source activate pascal

mkdir -p $OUTDIR

size=90000
i=0

python $SCRIPTSDIR/bpe_tags_mean.py $lang $INDIR/$TEST $OUTDIR/test.$lang $size $i &
python $SCRIPTSDIR/bpe_tags_mean.py $lang $INDIR/$VALID $OUTDIR/valid.$lang $size $i &

for i in {0..47}; do
  python $SCRIPTSDIR/bpe_tags_mean.py $lang $INDIR/$TRAIN $OUTDIR/train.$lang.$i $size $i &
done

wait

rm $OUTDIR/train.$lang
for i in {0..47}; do
  cat $OUTDIR/train.$lang.$i >> $OUTDIR/train.$lang
done
rm $OUTDIR/train.$lang.*

conda deactivate
