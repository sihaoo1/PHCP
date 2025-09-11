#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <DIR> [additional_python_args]"
    exit 1
fi

DIR=$1
shift  
FIRST_CMD_ARGS="$1"
shift  
PYTHON_ARGS="$@"


echo "Executing: python opencood/tools/generate_fake_label.py --model_dir opencood/logs/opv2v/paper/$DIR/fake_label $FIRST_CMD_ARGS $PYTHON_ARGS"
echo "Executing: python opencood/tools/few_shot_train.py --model_dir opencood/logs/opv2v/paper/$DIR/fsl_train $PYTHON_ARGS"
echo "Executing: python opencood/tools/batch_merge_and_infer.py --base_dir opencood/logs/opv2v/paper/$DIR $PYTHON_ARGS"
sleep 2


echo "Generating fake labels..."



python opencood/tools/generate_fake_label.py --model_dir opencood/logs/opv2v/paper/$DIR/fake_label $FIRST_CMD_ARGS $PYTHON_ARGS

echo "Copying fake labels..."
mkdir -p opencood/logs/opv2v/paper/$DIR/fsl_train
cp opencood/logs/opv2v/paper/$DIR/fake_label/fake_label.pkl opencood/logs/opv2v/paper/$DIR/fsl_train

echo "Starting few-shot training..."
python opencood/tools/few_shot_train.py --model_dir opencood/logs/opv2v/paper/$DIR/fsl_train $PYTHON_ARGS > opencood/logs/opv2v/paper/$DIR/fsl_train/log.txt

echo "Running batch merge and inference..."
python opencood/tools/batch_merge_and_infer.py --base_dir opencood/logs/opv2v/paper/$DIR $PYTHON_ARGS

echo "All tasks completed successfully."
