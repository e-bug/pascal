#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
PROJ_PATH=$HOME/syntaxNMT/dasadev_fairseq/experiments/dep-enc/cwmt17en2de
DATA_PATH=/gs/hs0/tga-nlp-titech/emanuele/data/dep-enc/cwmt17en2de #$PROJ_PATH/data-bin
CKPT_PATH=/gs/hs0/tga-nlp-titech/emanuele/checkpoints/syntaxNMT/dasa-dev/dep-enc/cwmt17en2de/transformer #$PROJ_PATH/checkpoints/transformer
MODEL_DIR=$PROJ_PATH/transformer
OUTPUT_FN=$MODEL_DIR/res.txt
mosesdecoder=$HOME/syntaxNMT/tools/mosesdecoder

export PATH=$SP_PATH:$PATH
#export PATH="/opt/anaconda3/bin${PATH:+:${PATH}}"
#. /opt/anaconda3/etc/profile.d/conda.sh
export PATH="$HOME/libs/anaconda3/bin:$PATH"
. $HOME/libs/anaconda3/etc/profile.d/conda.sh

conda activate nmt

mkdir -p $MODEL_DIR/outputs_last
cd $HOME/syntaxNMT/dasadev_fairseq

for split in valid test; do
  python generate.py $DATA_PATH \
          --gen-subset $split \
          --path $CKPT_PATH/checkpoint_last.pt \
          --batch-size 128 \
  	  --remove-bpe \
          --beam 4 \
          --lenpen 0.6 \
          > $OUTPUT_FN

  # Extract source, predictions and ground truth
  grep '^S-[0-9]*' $OUTPUT_FN | sed 's|^..||' | sort -k1 -n | cut -f2 > $MODEL_DIR/outputs_last/src.tok.$split
  grep '^H-[0-9]*' $OUTPUT_FN | sed 's|^..||' | sort -k1 -n | cut -f3 > $MODEL_DIR/outputs_last/preds.tok.$split
  grep '^T-[0-9]*' $OUTPUT_FN | sed 's|^..||' | sort -k1 -n | cut -f2 > $MODEL_DIR/outputs_last/truth.tok.$split

  # Detokenize
  perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -q -l de \
          < $MODEL_DIR/outputs_last/preds.tok.$split \
          > $MODEL_DIR/outputs_last/preds.detok.$split
  perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -q -l de \
          < $MODEL_DIR/outputs_last/truth.tok.$split \
          > $MODEL_DIR/outputs_last/truth.detok.$split

  # Fix some moses detokenization
  sed "s| '|'|g" $MODEL_DIR/outputs_last/preds.detok.$split | sed "s| /|/|g" | sed "s|/ |/|g" | sed "s| @ - @ |-|g" \
  	  > $MODEL_DIR/outputs_last/preds.$split
  sed "s| '|'|g" $MODEL_DIR/outputs_last/truth.detok.$split | sed "s| /|/|g" | sed "s|/ |/|g" | sed "s| @ - @ |-|g" \
          > $MODEL_DIR/outputs_last/truth.$split
  rm $MODEL_DIR/outputs_last/preds.detok.$split $MODEL_DIR/outputs_last/truth.detok.$split

  # Compute BLEU
  #cat $MODEL_DIR/outputs_last/preds.$split \
  #        | sacrebleu $MODEL_DIR/outputs_last/truth.$split \
  #        > $MODEL_DIR/bleu.last.$split
  if [ $split == valid ]; then
    cat $MODEL_DIR/outputs_last/preds.$split | sacrebleu -t wmt16 -l en-de > $MODEL_DIR/bleu.last.$split
  else
    cat $MODEL_DIR/outputs_last/preds.$split | sacrebleu -t wmt17 -l en-de > $MODEL_DIR/bleu.last.$split
  fi

done

rm $OUTPUT_FN

conda deactivate
