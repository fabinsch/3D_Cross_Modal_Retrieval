#!/bin/sh

#  process_files.sh
#

for dirname in data2/*; do
    for subdirname in $dirname/*; do
        for model in $subdirname/*; do
            #echo "$subdirname"
            #find $model -name '*.ply' echo $model \;
            python /Users/fabischramm/Documents/ADL4CV/adl4cv/our_utils.py "$subdirname"
        done
    done
done

