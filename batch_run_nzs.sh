#!/bin/bash

NPROC=$1
NGALS=$2
NREALS=$3
RUN=$4
DATA_PATH=$5
BATCH=$6
START=$7
END=$8

for (( i=$START; i < $END; ++i )) 
do
    echo RUN: $RUN"_"$i
    mpiexec -n $NPROC python sample_sps_params.py $NGALS $NREALS $RUN"_"$i $DATA_PATH 0
    python sample_photometry.py $DATA_PATH $RUN"_"$i $BATCH
    python photo_to_nz.py $DATA_PATH $RUN"_"$i 1

    rm $DATA_PATH"nz_samples/c_"$RUN"_"$i".npy"
    rm $DATA_PATH"sps_parameter_samples/sps_"$RUN"_"$i".npy"
    rm $DATA_PATH"photo_samples/photo_"$RUN"_"$i".npy"
	
    mpiexec -n $NPROC python sample_sps_params.py $NGALS $NREALS $RUN"_mean_"$i $DATA_PATH 1
    python sample_photometry.py $DATA_PATH $RUN"_mean_"$i $BATCH
    python photo_to_nz.py $DATA_PATH $RUN"_mean_"$i 1

    rm $DATA_PATH"nz_samples/c_"$RUN"_mean_"$i".npy"
    rm $DATA_PATH"/sps_parameter_samples/sps_"$RUN"_mean_"$i".npy"
    rm $DATA_PATH"/photo_samples/photo_"$RUN"_mean_"$i".npy"
	
done




