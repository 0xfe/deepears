#!/bin/bash
#
# Sync local state to remote server.

ADDRESS=$TF1HOST
SSHUSER=mohit

echo $ADDRESS
if [ ! -e bootstrap ]; then
    echo "Can't find directory bootstrap/. Run from deepears root directory."
    exit -1
fi

echo Syncing...
# rsync -przvl --executability --stats bootstrap $SSHUSER@$ADDRESS:
rsync -przvl --executability --stats src $SSHUSER@$ADDRESS:
