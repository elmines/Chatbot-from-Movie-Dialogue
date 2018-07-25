#!/bin/bash

ORIG_DIR=`pwd`
cd ..
python infer.py $ORIG_DIR/vad_infer.yml
cd $ORIG_DIR
