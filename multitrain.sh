#!/bin/bash

for i in $(cat filelist);
do
    python3 $i &
done
