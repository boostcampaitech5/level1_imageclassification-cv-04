#!/bin/sh

SIMPLE_PATH_FLAG:=0
BASE_DIR:="/opt/ml"
CURR_DIR:=$PWD
SM_CHANNEL_TRAIN:=$BASE_DIR+"/input/data/train/images"
SM_MODEL_DIR:=$BASE_DIR+$CURR_DIR+"/model"
SM_CHANNEL_EVAL:=$BASE_DIR+$CURR_DIR+"/data/eval"

if [ $# -eq 0 ]; then
	EXP_NAME:="exp"
else
	EXP_NAME:=$1
fi
SM_CHANNEL_MODEL:=$BASE_DIR+$CURR_DIR+"/model/"+$EXP_NAME

if [ $SIMPLE_PATH_FLAG -eq 0 ]; then
	SM_OUTPUT_DATA_DIR:=$BASE_DIR+$CURR_DIR+"/output"
else
	SM_OUTPUT_DATA_DIR:=$BASE_DIR+"/output"
fi
SM_GROUND_TRUTH_DIR:=$BASE_DIR+"/gt"

python train.py
python inference.py
#python evaluation.py
