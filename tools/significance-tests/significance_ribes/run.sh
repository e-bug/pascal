#!/bin/bash

reference=
baseline=
proposed=
nsamples=1000

ribesdir="$HOME/pascal/tools/RIBES-1.03.1"

# Bootstrap resampling
python resample.py $reference $proposed $baseline -B $nsamples

# RIBES evaluation
python $ribesdir/RIBES.py -r $reference $proposed > ribes.x
python $ribesdir/RIBES.py -r $reference $baseline > ribes.y
for i in `seq 0 $[nsamples - 1]`; do
  python $ribesdir/RIBES.py -r refs.$i xs.$i > ribes.x.$i
  python $ribesdir/RIBES.py -r refs.$i ys.$i > ribes.y.$i
done

# Significance testing
python significance_test.py

# Clean
rm refs.* xs.* ys.* ribes.*

