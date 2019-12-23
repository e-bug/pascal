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
OUTPUT_DIR="$PROJ_DIR/data/wmt18tren/corpus"
MOSES_DIR="$PROJ_DIR/tools/mosesdecoder"
BPE_DIR="$PROJ_DIR/tools/subword-nmt"
SCRIPTS_DIR="$PROJ_DIR/scripts"

# activate environment
source activate pascal

mkdir -p $OUTPUT_DIR

echo "Downloading WMT18 Tr-En. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR}/corpus.gz \
  http://data.statmt.org/wmt18/translation-task/preprocessed/tr-en/corpus.gz

echo "Downloading dev/test sets"
wget -nc -nv -O ${OUTPUT_DIR}/dev.tgz \
  http://data.statmt.org/wmt18/translation-task/preprocessed/tr-en/dev.tgz

# Extract everything
echo "Extracting all files..."
gunzip ${OUTPUT_DIR}/corpus.gz > ${OUTPUT_DIR}/corpus
cut -f1 ${OUTPUT_DIR}/corpus > ${OUTPUT_DIR}/train.tok.tr
cut -f2 ${OUTPUT_DIR}/corpus > ${OUTPUT_DIR}/train.tok.en
mkdir -p "${OUTPUT_DIR}/dev"
tar -xvzf "${OUTPUT_DIR}/dev.tgz" -C "${OUTPUT_DIR}/dev"
cp ${OUTPUT_DIR}/dev/newstest2016.tc.tr ${OUTPUT_DIR}/valid.tok.tr
cp ${OUTPUT_DIR}/dev/newstest2016.tc.en ${OUTPUT_DIR}/valid.tok.en
cp ${OUTPUT_DIR}/dev/newstest2017.tc.tr ${OUTPUT_DIR}/test.tok.tr
cp ${OUTPUT_DIR}/dev/newstest2017.tc.en ${OUTPUT_DIR}/test.tok.en

# Remove raw data
rm -r ${OUTPUT_DIR}/corpus ${OUTPUT_DIR}/dev*

# CoreNLP tokenization
for f in ${OUTPUT_DIR}/*.tok.en; do
  fbase=${f%.*}
  echo "CoreNLP tokenizing ${fbase}..."
  python ${SCRIPTS_DIR}/corenlp_tok_en.py $fbase &
  cp $fbase.tr $fbase.tok.tr
done

wait

# Learn Shared BPE
for merge_ops in 16000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "${OUTPUT_DIR}/train.tok.tok.tr" "${OUTPUT_DIR}/train.tok.tok.en" | \
    ${BPE_DIR}/subword_nmt/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"
  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in en tr; do
    for f in ${OUTPUT_DIR}/*tok.tok.${lang}; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${BPE_DIR}/subword_nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
    done
  done
done

echo "All done."

conda deactivate
