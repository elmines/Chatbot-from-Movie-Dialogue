#!/bin/bash

ORIG_DIR=`pwd`
cd ..
python train.py $ORIG_DIR/asghar_vad_train.yml
cd $ORIG_DIR
