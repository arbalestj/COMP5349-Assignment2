#!/bin/bash

spark-submit \
    --py-files file:///home/hadoop/COMP5349-Assignment2/ml_utils.py \
    --master yarn \
    Stage3.py \
