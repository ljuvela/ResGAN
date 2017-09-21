#!/bin/bash

SRCDIR='./output'
TARDIR='/home/ljuvela/CODE/DNNAM-nick/DNNAM-glot/scripts/currennt/ac2glot/testdata/gen_htk' # subseqs
#TARDIR='/home/ljuvela/CODE/DNNAM-nick/DNNAM-glot/scripts/currennt/ac2glot/gen/pls'

EXT='.pls' # can also be .pls_nonoise
EXT='.pls_nonoise'
TAREXT='.pls'

for f in $SRCDIR/*$EXT; do

    bname=$(basename $f $EXT)
    echo cp $f $TARDIR/$bname$TAREXT
    cp $f $TARDIR/$bname$TAREXT

done
