#!/bin/bash

ORIG_DIR=`pwd`
cd ..
python train.py $ORIG_DIR/vad_train.yml
cd $ORIG_DIR
