#!/bin/bash

python eval_detection_results.py thumos14 results/Flow_result results/RGB_result --score_weights 1.2 1 --nms_threshold 0.32
