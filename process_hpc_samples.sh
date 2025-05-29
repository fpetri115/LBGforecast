#!/bin/bash

RUN_START=$1
RUN_END=$2
DATA_PATH=$3
RUN_NAME=$4
BATCH_SIZE=$5

for((i=$RUN_START; i<=RUN_END; i++))
do
    python sample_photometry.py $DATA_PATH "$RUN_NAME"_$i $BATCH_SIZE
    python photo_to_nz.py $DATA_PATH "$RUN_NAME"_$i
    rm "$DATA_PATH"sps_parameter_samples/sps_"$RUN_NAME"_$i.npy
    rm "$DATA_PATH"photo_samples/photo_"$RUN_NAME"_$i.npy

done

