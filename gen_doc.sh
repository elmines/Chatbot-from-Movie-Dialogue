#!/bin/bash

mkdir -p doc/
WORKING_DIR=`pwd`

export SPHINX_APIDOC_OPTIONS="members,special-members"

sphinx-apidoc -f --full -a --separate --doc-project EmotChatbot -o doc/ $WORKING_DIR
cd doc
make html
cd ..
