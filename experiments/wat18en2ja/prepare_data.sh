#! /usr/bin/env bash

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by:        Emanuele Bugliarello (@e-bug)
# Date last modified: 9/4/2019

PROJ_DIR="$HOME/pascal"
OUTPUT_DIR="$PROJ_DIR/data/wat18jaen/corpus"
MOSES_DIR="$PROJ_DIR/tools/mosesdecoder"
BPE_DIR="$PROJ_DIR/tools/subword-nmt"
SCRIPTS_DIR="$PROJ_DIR/scripts"

# activate environment
source activate pascal

mkdir -p $OUTPUT_DIR

## Extract sentences
#for name in dev test; do
#  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[2], "\n";' < $OUTPUT_DIR/${name}.txt > $OUTPUT_DIR/${name}.ja.txt
#  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[3], "\n";' < $OUTPUT_DIR/${name}.txt > $OUTPUT_DIR/${name}.en
#done
#for name in train-1 train-2 train-3; do
#  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[3], "\n";' < $OUTPUT_DIR/${name}.txt > $OUTPUT_DIR/${name}.ja.txt
#  perl -ne 'chomp; @a=split/ \|\|\| /; print $a[4], "\n";' < $OUTPUT_DIR/${name}.txt > $OUTPUT_DIR/${name}.en
#done
#
## Removing date expressions at EOS in Japanese in the training and development data
#for file in train-1 train-2 train-3 dev; do
#  cat $OUTPUT_DIR/${file}.ja.txt | perl -pe 's/(.)［[０-９．]+］$/${1}/;' > $OUTPUT_DIR/${file}.ja
#  rm $OUTPUT_DIR/${file}.ja.txt
#done
#mv $OUTPUT_DIR/test.ja.txt $OUTPUT_DIR/test.ja
#
## Tokenize EN data
#for f in ${OUTPUT_DIR}/*.en; do
#  echo "Tokenizing $f..."
#  ${MOSES_DIR}/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < $f > ${f%.*}.tok.en
#done
#for f in ${OUTPUT_DIR}/*.tok.en; do
#  fbase=${f%.*}
#  echo "CoreNLP tokenizing ${fbase}..."
#  python $SCRIPTS_DIR/corenlp_tok_en.py $fbase
#done
#
## Tokenize JA data
#for file in ${OUTPUT_DIR}/*.ja; do
#  echo "Cabocha tokenizing $file..."
#  cat ${file} | perl -pe 's/　/ /g;' | cabocha -f3 > ${file%.*}.f3
#  python $SCRIPTS_DIR/cabocha_tok.py ${file%.*}.f3
#  cat ${file%.*}.cabocha.tok > ${file%.*}.tok.tok.ja
#done

# Combine train files
rm ${OUTPUT_DIR}/train.tok.tok.en ${OUTPUT_DIR}/train.tok.tok.ja
rm ${OUTPUT_DIR}/train.cabocha.dep ${OUTPUT_DIR}/train.cabocha.cnk
for file in train-1 train-2 train-3; do
  cat ${OUTPUT_DIR}/${file}.tok.tok.en >> ${OUTPUT_DIR}/train.tok.tok.en
  cat ${OUTPUT_DIR}/${file}.tok.tok.ja >> ${OUTPUT_DIR}/train.tok.tok.ja
  cat ${OUTPUT_DIR}/${file}.cabocha.dep >> ${OUTPUT_DIR}/train.cabocha.dep
  cat ${OUTPUT_DIR}/${file}.cabocha.cnk >> ${OUTPUT_DIR}/train.cabocha.cnk
done

# Learn Shared BPE
for merge_ops in 32000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  ${BPE_DIR}/subword_nmt/learn_joint_bpe_and_vocab.py \
    --input ${OUTPUT_DIR}/train.tok.tok.en ${OUTPUT_DIR}/train.tok.tok.ja \
    -s $merge_ops -o ${OUTPUT_DIR}/bpe.${merge_ops} \
    --write-vocabulary ${OUTPUT_DIR}/vocab.${merge_ops}.en ${OUTPUT_DIR}/vocab.${merge_ops}.ja
  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in en ja; do
    for f in ${OUTPUT_DIR}/*.tok.tok.${lang}; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${BPE_DIR}/subword_nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" \
        --vocabulary ${OUTPUT_DIR}/vocab.${merge_ops}.$lang \
        < $f > "${outfile}"
    done
  done
done

echo "All done."

conda deactivate
