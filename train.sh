#!/bin/bash
save_path=$1

python pgcn_train.py thumos14 --snapshot_pre ${save_path} 
