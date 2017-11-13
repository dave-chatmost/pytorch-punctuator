#!/bin/bash

# Created on 2017-06-12
# Author: Kaituo Xu
# Function: Compute WER for hyp files that match pattern.
# NOTE: You need to download Kaldi at first, because
#       we use kaldi util: compute-wer.

if [ $# -le 2 ]; then
  echo "Compute WER for hyp files that match pattern."
  echo "Usage: $0 <ref-dir> <hyp-dir> <simple-file-pattern>"
  echo "e.g.: $0 data/TN data/hyp asr_out"
  exit 1
fi

REF_DIR=$1
HYP_DIR=$2
PATTERN=$3

# In order to use `compute-wer`
cd ~/tools/kaldi/egs/wsj/s5/; . ./path.sh; cd -

files=`ls $HYP_DIR/*$PATTERN*`
for hyp in $files; do
  echo -e "\n*****\nFILE: $hyp\n*****"
  cor_ref=$REF_DIR/`basename ${hyp/$PATTERN/ref}`
  cor_unpunc=$REF_DIR/`basename $hyp`
  # Add utt-id to per utterance
  awk '{ print NR "\t" $0 }' $hyp > temp_hyp
  awk '{ print NR "\t" $0 }' $cor_ref > temp_ref
  awk '{ print NR "\t" $0 }' $cor_unpunc > temp_unpunc
  echo "---------- BEFORE PUNCTUATION ----------"
  compute-wer --text --mode=present ark:temp_ref ark:temp_unpunc
  echo ""
  echo "---------- AFTER PUNCTUATION ----------"
  compute-wer --text --mode=present ark:temp_ref ark:temp_hyp
done
rm -f temp_hyp temp_ref temp_unpunc
