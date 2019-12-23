#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

src=en
tgt=de
PROJ=$HOME/pascal
TEXT=$PROJ/data/nc11deen/corpus
TAGS=$PROJ/data/nc11deen/lin_parses
OUT=$PROJ/data/nc11deen/corpus_lin_parses/${src}2${tgt}
OUTPUT=$OUT/nc11${src}2${tgt}
BPE_DIR="$PROJ/tools/subword-nmt"

# Combine text and parse files
mkdir -p $OUT
for lang in $src $tgt; do
  cp $TEXT/valid.tok.tok.$lang $OUT/valid.$lang
  cp $TEXT/test.tok.tok.$lang $OUT/test.$lang
done
cat $TEXT/train.tok.clean.tok.$src > $OUT/train.tr.$src
cat $TEXT/train.tok.clean.tok.$src > $OUT/train.pa.$src
cat $TEXT/train.tok.clean.tok.$tgt > $OUT/train.$tgt
cat $TAGS/train.$src >> $OUT/train.$tgt

# Add tags for translation and parsing
sed -i -e 's/^/<TR> /' $OUT/train.tr.$src $OUT/valid.$src $OUT/test.$src
sed -i -e 's/^/<PA> /' $OUT/train.pa.$src
sed -i 's/$/ <TR>/g' $OUT/train.tr.$src $OUT/valid.$src $OUT/test.$src
sed -i 's/$/ <PA>/g' $OUT/train.pa.$src
cat $OUT/train.tr.$src > $OUT/train.$src
cat $OUT/train.pa.$src >> $OUT/train.$src
rm $OUT/train.tr.$src $OUT/train.pa.$src

# Shuffle training set
ORDER="$OUT/.rand.$$"
trap "rm -f $ORDER;exit" 1 2
count=$(grep -c '^' "$OUT/train.$src")
seq -w $count | shuf > $ORDER
for lang in $src $tgt; do
  file=$OUT/train.$lang
  outfile=${file%.*}
  paste -d' ' $ORDER $file | sort -k1n | cut -d' ' -f2-  > "$outfile.rand.$lang"
done
rm -f $ORDER

# Learn Shared BPE
for merge_ops in 16000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "${OUT}/train.rand.$src" "${OUT}/train.rand.$tgt" | \
    ${BPE_DIR}/subword_nmt/learn_bpe.py -s $merge_ops > "${OUT}/bpe.${merge_ops}"
  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in $src $tgt; do
    for f in ${OUT}/*.${lang}; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${BPE_DIR}/subword_nmt/apply_bpe.py -c "${OUT}/bpe.${merge_ops}" < $f > "${outfile}"
    done
  done
done

# activate environment
source activate pascal

# Binarize the dataset:
cd $PROJ/fairseq
python preprocess.py \
	--source-lang $src \
	--target-lang $tgt \
	--trainpref $OUT/train.rand.bpe.16000 \
	--validpref $OUT/valid.bpe.16000 \
	--testpref $OUT/test.bpe.16000 \
	--destdir $OUTPUT \
	--workers 16 \
	--nwordssrc 16384 \
        --nwordstgt 16384 \
        --joined-dictionary

# deactivate environment
conda deactivate
