#!/bin/bash

# Created by:         Emanuele Bugliarello (@e-bug)
# Date created:       9/4/2019
# Date last modified: 9/4/2019

export CUDA_VISIBLE_DEVICES=0
src=en
tgt=de
PROJ_PATH=$HOME/pascal/experiments/nc11${src}2${tgt}
DATA_PATH=
TAGS_PATH=
CKPT_PATH=
MODEL_DIR=$PROJ_PATH/transformer_pascal
OUTPUT_FN=$MODEL_DIR/res.txt
mosesdecoder=$HOME/pascal/tools/mosesdecoder

export PATH="$HOME/libs/anaconda3/bin:$PATH"
. $HOME/libs/anaconda3/etc/profile.d/conda.sh

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

  # Detokenize
  perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -q -l $tgt \
          < $MODEL_DIR/outputs/preds.tok.$split \
          > $MODEL_DIR/outputs/preds.detok.$split
  perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -q -l $tgt \
          < $MODEL_DIR/outputs/truth.tok.$split \
          > $MODEL_DIR/outputs/truth.detok.$split

  # Fix some moses detokenization
  sed "s| '|'|g" $MODEL_DIR/outputs/preds.detok.$split | sed "s| /|/|g" | sed "s|/ |/|g" | sed "s| @ - @ |-|g" \
  	  > $MODEL_DIR/outputs/preds.$split
  sed "s| '|'|g" $MODEL_DIR/outputs/truth.detok.$split | sed "s| /|/|g" | sed "s|/ |/|g" | sed "s| @ - @ |-|g" \
          > $MODEL_DIR/outputs/truth.$split
  rm $MODEL_DIR/outputs/preds.detok.$split $MODEL_DIR/outputs/truth.detok.$split

  # Compute BLEU
  if [ $split == valid ]; then
    cat $MODEL_DIR/outputs/preds.$split | sacrebleu -t wmt15 -l $src-$tgt > $MODEL_DIR/bleu.$split
  else
    cat $MODEL_DIR/outputs/preds.$split | sacrebleu -t wmt16 -l $src-$tgt > $MODEL_DIR/bleu.$split
  fi

done

rm $OUTPUT_FN

conda deactivate
