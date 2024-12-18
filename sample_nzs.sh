#!/bin/bash

NPROC=$1
NGALS=$2
NREALS=$3
RUN=$4
DATA_PATH=$5
COLOURS=$6
MEAN=$7

mpiexec -n $NPROC python sample_sps_params.py $NGALS $NREALS $RUN $DATA_PATH $MEAN
python sample_photometry.py $DATA_PATH $RUN
python photo_to_nz.py $DATA_PATH $RUN $COLOURS


