#!/bin/bash

NPROC=$1
NGALS=$2
NREALS=$3
RUN=$4
DATA_PATH=$5
EXTRA=$6
MEAN=$7
BATCH_SIZE=$8
DUST_CHOICE=$9

mpiexec -n $NPROC python sample_sps_params.py $NGALS $NREALS $RUN $DATA_PATH $MEAN $DUST_CHOICE
python sample_photometry.py $DATA_PATH $RUN $BATCH_SIZE
python photo_to_nz.py $DATA_PATH $RUN $EXTRA


