
MODEL_NAME=$1
MODEL_TYPE=$2
DTYPE=$3
DATASET=$4
OUTPUT_DIR=./results/${MODEL_NAME}_${MODEL_TYPE}_${DTYPE}_${DATASET}
CMD="python test_unsloth.py --model_name=$MODEL_NAME --model_type=$MODEL_TYPE --dtype=$DTYPE --dataset_id=$DATASET --output_dir=$OUTPUT_DIR"

echo $CMD
$CMD