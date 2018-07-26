#!/bin/bash

mkdir -p doc/
WORKING_DIR=`pwd`

export SPHINX_APIDOC_OPTIONS="members,special-members"

sphinx-apidoc -f --full -a --separate --doc-project EmotChatbot -o doc/ $WORKING_DIR

echo "Changing into doc/ directory . . ."
cd doc

make html
echo "More specifically, the go to doc/_build/html/index.html for the documentation"
cd ..
