#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

export CUDA_VISIBLE_DEVICES=0
src=en
tgt=ja
PROJ_PATH=$HOME/pascal/experiments/wat18${src}2${tgt}
DATA_PATH=
TAGS_PATH=
CKPT_PATH=
MODEL_DIR=$PROJ_PATH/transformer_pascal
OUTPUT_FN=$MODEL_DIR/res.txt
mosesdecoder=$HOME/pascal/tools/mosesdecoder
kyteadir=$HOME/libs/kytea/bin
ribesdir="$HOME/syntaxNMT/tools/RIBES-1.03.1"

export PATH="$HOME/libs/kytea/bin${PATH:+:${PATH}}"
export PATH="$HOME/libs/anaconda3/bin${PATH:+:${PATH}}"
. $HOME/libs/anaconda3/etc/profile.d/conda.sh
export LIBRARY_PATH=$PATH
export LIBRARY_PATH="$HOME/libs/kytea/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH=$LIBRARY_PATH

conda activate pascal

mkdir -p $MODEL_DIR/outputs
cd $HOME/pascal/fairseq


for split in valid test; do
  python generate.py $DATA_PATH \
          --tags-data $TAGS_PATH \
          --gen-subset $split \
          --task tags_translation \
          --path $CKPT_PATH/checkpoint_last.pt \
          --batch-size 128 \
          --remove-bpe \
          --beam 4 \
          --lenpen 0.6 \
          > $OUTPUT_FN

  # Extract source, predictions and ground truth
  grep '^S-[0-9]*' $OUTPUT_FN | sed 's|^..||' | sort -k1 -n | cut -f2 > $MODEL_DIR/outputs/src.tok.$split
  grep '^H-[0-9]*' $OUTPUT_FN | sed 's|^..||' | sort -k1 -n | cut -f3 > $MODEL_DIR/outputs/preds.tok.$split
  grep '^T-[0-9]*' $OUTPUT_FN | sed 's|^..||' | sort -k1 -n | cut -f2 > $MODEL_DIR/outputs/truth.tok.$split

  # Detokenize data 
  sed "s| '|'|g" $MODEL_DIR/outputs/preds.tok.$split | sed "s| /|/|g" | sed "s|/ |/|g" \
         | sed 's/ //g' \
          > $MODEL_DIR/outputs/preds.$split
  sed "s| '|'|g" $MODEL_DIR/outputs/truth.tok.$split | sed "s| /|/|g" | sed "s|/ |/|g" \

  # Tokenize JA data
  cat $DATA_PATH/${split}.ja | \
      $kyteadir/kytea -notags > $MODEL_DIR/outputs/truth.tok.$split
  cat $MODEL_DIR/outputs/preds.$split | \
      $kyteadir/kytea -notags > $MODEL_DIR/outputs/preds.tok.$split

  # Compute BLEU
  cat $MODEL_DIR/outputs/preds.tok.$split \
          | sacrebleu $MODEL_DIR/outputs/truth.tok.$split -tok none \
          > $MODEL_DIR/bleu.$split

  # Compute RIBES
  python $ribesdir/RIBES.py -r \
          $MODEL_DIR/outputs/truth.tok.$split \
          $MODEL_DIR/outputs/preds.tok.$split \
          > $MODEL_DIR/ribes.$split

done


rm $OUTPUT_FN

conda deactivate
