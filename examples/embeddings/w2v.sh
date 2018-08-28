#!/bin/bash
set -x

SCRIPT_DIR=`pwd`
echo "Concatenating all training text to corpus.txt"
cat ../../corpora/train_*.txt ../../corpora/valid_*.txt > corpus.txt

cd ../..
python -u embeddings.py --vocab corpora/vocab.txt --save-path $SCRIPT_DIR/w2v.npy --w2v $SCRIPT_DIR/corpus.txt
cd $SCRIPT_DIR
