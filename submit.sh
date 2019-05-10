#!/bin/bash

spark-submit \
    --py-files file:///home/hadoop/COMP5349-Assignment2/ml_ultis.py \
    --master yarn \
    Stage2.py \
