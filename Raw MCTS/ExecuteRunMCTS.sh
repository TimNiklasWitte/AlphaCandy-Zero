#!/bin/bash

set -x #echo on

start=25
stop=275
step=25


for candy_buff_height in {1..7}
do

    python3 RunMCTS.py --start $start --stop $stop --step $step --buff $candy_buff_height
 
done