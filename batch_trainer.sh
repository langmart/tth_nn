#!/bin/bash

for i in `cat filelist`
do
    if grep -xq $i started
    then
        echo "$i is already running. Continue with the next file."
    else
        echo $i >> started
        python3 $i
        sleep 5
    fi
done
rm started
