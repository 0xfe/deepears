#!/bin/bash
#
# Usage:
#    $ source env.sh
#    $ bootstrap/seed.sh $TF1HOST

if [ "$1" == "" -o "$2" == "" ]; then
  echo "Usage: $0 [address] [sshuser]"
  exit -1
fi

ADDRESS=$1
SSHUSER=$2

if [ "$SSHUSER" == "" ]; then
  SSHUSER=mohit
fi

if [ ! -e bootstrap ]; then
    echo "Can't find directory bootstrap/. Run from deepears root directory."
    exit -1
fi

echo Testing ssh server at $SSHUSER@$ADDRESS...
ssh -t $SSHUSER@$ADDRESS exit

if [ "$?" != "0" ]; then
  echo "Can't login to $ADDRESS."
  exit -1
fi

echo Testing server at $ADDRESS...
ssh -t $SSHUSER@$ADDRESS exit

if [ "$?" != "0" ]; then
  echo "Can't login to $ADDRESS."
  exit -1
fi

echo Seeding server $ADDRESS with bootstrap files...
rsync -przvl --executability --stats bootstrap $SSHUSER@$ADDRESS:
rsync -przvl --executability --stats src $SSHUSER@$ADDRESS:

echo Kicking off bootstrap on server $ADDRESS...
ssh $SSHUSER@$ADDRESS "cd bootstrap; ./install-gpu.sh"
