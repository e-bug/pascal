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
OUTPUT_DIR="$PROJ_DIR/data/wmt17deen/corpus"
MOSES_DIR="$PROJ_DIR/tools/mosesdecoder"
BPE_DIR="$PROJ_DIR/tools/subword-nmt"
SCRIPTS_DIR="$PROJ_DIR/scripts"

# activate environment
source activate pascal

mkdir -p $OUTPUT_DIR

echo "Downloading WMT17 De-En. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR}/corpus.tc.de.gz \
  http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.de.gz
wget -nc -nv -O ${OUTPUT_DIR}/corpus.tc.en.gz \
  http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/corpus.tc.en.gz

echo "Downloading dev/test sets"
wget -nc -nv -O ${OUTPUT_DIR}/dev.tgz \
  http://data.statmt.org/wmt18/translation-task/preprocessed/de-en/dev.tgz

# Extract everything
echo "Extracting all files..."
gunzip ${OUTPUT_DIR}/corpus.tc.de.gz ${OUTPUT_DIR}/corpus.tc.en.gz
mv ${OUTPUT_DIR}/corpus.tc.de ${OUTPUT_DIR}/train.tok.de
mv ${OUTPUT_DIR}/corpus.tc.en ${OUTPUT_DIR}/train.tok.en
mkdir -p "${OUTPUT_DIR}/dev"
tar -xvzf "${OUTPUT_DIR}/dev.tgz" -C "${OUTPUT_DIR}/dev"
cp ${OUTPUT_DIR}/dev/newstest2016.tc.de ${OUTPUT_DIR}/valid.tok.de
cp ${OUTPUT_DIR}/dev/newstest2016.tc.en ${OUTPUT_DIR}/valid.tok.en
cp ${OUTPUT_DIR}/dev/newstest2017.tc.de ${OUTPUT_DIR}/test.tok.de
cp ${OUTPUT_DIR}/dev/newstest2017.tc.en ${OUTPUT_DIR}/test.tok.en

# Remove raw data
rm -r ${OUTPUT_DIR}/dev*

# CoreNLP tokenization
for f in ${OUTPUT_DIR}/*.tok.en; do
  fbase=${f%.*}
  echo "CoreNLP tokenizing ${fbase}..."
  python ${SCRIPTS_DIR}/corenlp_tok.py $fbase
done

# Learn Shared BPE
for merge_ops in 32000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "${OUTPUT_DIR}/train.tok.tok.de" "${OUTPUT_DIR}/train.tok.tok.en" | \
    ${BPE_DIR}/subword_nmt/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"
  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in en de; do
    for f in ${OUTPUT_DIR}/*tok.tok.${lang}; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${BPE_DIR}/subword_nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
    done
  done
done

echo "All done."

conda deactivate
