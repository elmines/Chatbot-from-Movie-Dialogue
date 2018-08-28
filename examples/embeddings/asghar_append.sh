#!/bin/bash
set -x

SCRIPT_DIR=`pwd`
cd ../..
python -u embeddings.py --vocab corpora/vocab.txt --save-path $SCRIPT_DIR/asghar_append.npy --append $SCRIPT_DIR/w2v.npy
cd $SCRIPT_DIR
