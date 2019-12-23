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
OUTPUT_DIR="$PROJ_DIR/data/wmt16deen/filt_corpus"
MOSES_DIR="$PROJ_DIR/tools/mosesdecoder"
BPE_DIR="$PROJ_DIR/tools/subword-nmt"
SCRIPTS_DIR="$PROJ_DIR/scripts"

# activate environment
source activate pascal

mkdir -p $OUTPUT_DIR

echo "Downloading Europarl v7. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR}/europarl-v7-de-en.tgz \
  http://www.statmt.org/europarl/v7/de-en.tgz

echo "Downloading Common Crawl corpus. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR}/common-crawl.tgz \
  http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz

echo "Downloading News Commentary v11. This may take a while..."
wget -nc -nv -O ${OUTPUT_DIR}/nc-v11.tgz \
  http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz

echo "Downloading dev/test sets"
wget -nc -nv -O  ${OUTPUT_DIR}/dev.tgz \
  http://data.statmt.org/wmt16/translation-task/dev.tgz
wget -nc -nv -O  ${OUTPUT_DIR}/test.tgz \
  http://data.statmt.org/wmt16/translation-task/test.tgz

# Extract everything
echo "Extracting all files..."
mkdir -p "${OUTPUT_DIR}/europarl-v7-de-en"
tar -xvzf "${OUTPUT_DIR}/europarl-v7-de-en.tgz" -C "${OUTPUT_DIR}/europarl-v7-de-en"
mkdir -p "${OUTPUT_DIR}/common-crawl"
tar -xvzf "${OUTPUT_DIR}/common-crawl.tgz" -C "${OUTPUT_DIR}/common-crawl"
mkdir -p "${OUTPUT_DIR}/nc-v11"
tar -xvzf "${OUTPUT_DIR}/nc-v11.tgz" -C "${OUTPUT_DIR}/nc-v11"
mkdir -p "${OUTPUT_DIR}/dev"
tar -xvzf "${OUTPUT_DIR}/dev.tgz" -C "${OUTPUT_DIR}/dev"
mkdir -p "${OUTPUT_DIR}/test"
tar -xvzf "${OUTPUT_DIR}/test.tgz" -C "${OUTPUT_DIR}/test"

# Concatenate train files
cat "${OUTPUT_DIR}/europarl-v7-de-en/europarl-v7.de-en.en" \
  "${OUTPUT_DIR}/common-crawl/commoncrawl.de-en.en" \
  "${OUTPUT_DIR}/nc-v11/training-parallel-nc-v11/news-commentary-v11.de-en.en" \
  > "${OUTPUT_DIR}/train.en"

cat "${OUTPUT_DIR}/europarl-v7-de-en/europarl-v7.de-en.de" \
  "${OUTPUT_DIR}/common-crawl/commoncrawl.de-en.de" \
  "${OUTPUT_DIR}/nc-v11/training-parallel-nc-v11/news-commentary-v11.de-en.de" \
  > "${OUTPUT_DIR}/train.de"

# Convert SGM files
# Convert newstest2015 data into raw text format
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR}/dev/dev/newstest2015-deen-src.de.sgm \
  > ${OUTPUT_DIR}/valid.de
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR}/dev/dev/newstest2015-deen-ref.en.sgm \
  > ${OUTPUT_DIR}/valid.en

# Convert newstest2016 data into raw text format
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR}/test/test/newstest2016-deen-src.de.sgm \
  > ${OUTPUT_DIR}/test.de
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR}/test/test/newstest2016-deen-ref.en.sgm \
  > ${OUTPUT_DIR}/test.en

# Remove raw data
rm ${OUTPUT_DIR}/europarl-v7-de-en.tgz ${OUTPUT_DIR}/common-crawl.tgz ${OUTPUT_DIR}/nc-v11.tgz ${OUTPUT_DIR}/dev.tgz ${OUTPUT_DIR}/test.tgz
rm -r ${OUTPUT_DIR}/europarl-v7-de-en ${OUTPUT_DIR}/common-crawl ${OUTPUT_DIR}/nc-v11 ${OUTPUT_DIR}/dev ${OUTPUT_DIR}/test

# Tokenize data
for f in ${OUTPUT_DIR}/*.de; do
  echo "Tokenizing $f..."
  ${MOSES_DIR}/scripts/tokenizer/tokenizer.perl -q -l de -threads 8 < $f > ${f%.*}.tok.de
done
for f in ${OUTPUT_DIR}/*.en; do
  echo "Tokenizing $f..."
  ${MOSES_DIR}/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < $f > ${f%.*}.tok.en
done

# Clean train corpus
f=${OUTPUT_DIR}/train.tok.en
fbase=${f%.*}
echo "Cleaning ${fbase}..."
${MOSES_DIR}/scripts/training/clean-corpus-n.perl $fbase de en "${fbase}.clean" 1 80

# CoreNLP tokenization and langdetect filtering
f=${OUTPUT_DIR}/train.tok.clean.en
fbase=${f%.*}
echo "CoreNLP tokenizing and langdetect filtering ${fbase}..."
python ${SCRIPTS_DIR}/langdetect_corenlp_tok.py $fbase

# CoreNLP tokenization
for f in ${OUTPUT_DIR}/valid.tok.en ${OUTPUT_DIR}/test.tok.en; do
  fbase=${f%.*}
  echo "CoreNLP tokenizing ${fbase}..."
  python ${SCRIPTS_DIR}/corenlp_tok.py $fbase
done

# Learn Shared BPE
for merge_ops in 32000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "${OUTPUT_DIR}/train.tok.clean.filt.de" "${OUTPUT_DIR}/train.tok.clean.filt.en" | \
    ${BPE_DIR}/subword_nmt/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"
  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in en de; do
    for f in ${OUTPUT_DIR}/*tok.tok.${lang} ${OUTPUT_DIR}/train.tok.clean.filt.${lang}; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${BPE_DIR}/subword_nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
    done
  done
done

echo "All done."

conda deactivate
