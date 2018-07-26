#!/bin/bash

ORIG_DIR=`pwd`
cd ..
python infer.py $ORIG_DIR/distr_infer.yml
cd $ORIG_DIR
