#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

PROJDIR=$HOME/pascal
INDIR=$PROJDIR/data/nc11deen/corpus
OUTDIR=$PROJDIR/data/nc11deen/tags_label
lang=en
TRAIN=train.tok.clean.tok.bpe.16000.$lang
VALID=valid.tok.tok.bpe.16000.$lang
TEST=test.tok.tok.bpe.16000.$lang
SCRIPTSDIR=$PROJDIR/scripts

source activate pascal

mkdir -p $OUTDIR

size=5000
i=0

python $SCRIPTSDIR/bpe_tags_label.py $lang $INDIR/$TEST $OUTDIR/test.$lang $size $i &
python $SCRIPTSDIR/bpe_tags_label.py $lang $INDIR/$VALID $OUTDIR/valid.$lang $size $i &

for i in {0..47}; do
  python $SCRIPTSDIR/bpe_tags_label.py $lang $INDIR/$TRAIN $OUTDIR/train.$lang.$i $size $i &
done

wait

rm $OUTDIR/train.$lang
for i in {0..47}; do
  cat $OUTDIR/train.$lang.$i >> $OUTDIR/train.$lang
done
rm $OUTDIR/train.$lang.*

conda deactivate
