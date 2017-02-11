#!/bin/bash
# Never execute too many networks at the same time. GPUs might become
# unresponsive...
for i in $(cat filelist);
do
    python3 $i
    echo "$i" >> executed
done
