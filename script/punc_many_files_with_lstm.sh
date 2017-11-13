#!/bin/bash

# Created on 2017-11-13
# Author: Kaituo Xu (Sogou)
# Function: Punctuate listed text files with LSTM model.
# NOTE: Execute in src directory.

if [ $# != 6 ]; then
  echo "Punctuate many files with LSTM."
  echo "Usage: <in-dir> <out-dir> <model-path> <vocab> <punc-vocab> <GPU-ids>"
  exit 1;
fi

# cd ../

INPUT_DIR=$1
OUTPUT_DIR=$2
MODEL_PATH=$3
VOCAB=$4
PUNC_VOCAB=$5
GPU_IDS=$6

files=`cd $INPUT_DIR; ls *asr_out*`

[ ! -d $OUTPUT_DIR ] && mkdir -p $OUTPUT_DIR

tool=punctuator.py

for file in $files
do
    echo "Punctuating $file"
    CUDA_VISIBLE_DEVICES=$GPU_IDS \
    python $tool --cuda \
        --data $INPUT_DIR/$file \
        --vocab $VOCAB \
        --punc_vocab $PUNC_VOCAB \
        --model_path $MODEL_PATH \
        --output $OUTPUT_DIR/$file
    echo "Put the result in $OUTPUT_DIR/$file"
done
