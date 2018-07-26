#!/bin/bash

ORIG_DIR=`pwd`
cd ..
python train.py $ORIG_DIR/distr_train.yml
cd $ORIG_DIR
