SCRIPT_NAME=$1
MODEL_NAME=$2
MODEL_TYPE=$3
DTYPE=$4
DATASET=$5
OUTPUT_DIR=$MODEL_NAME_$MODEL_TYPE_$DTYPE_$DATASET
CMD="python test_load.py --model_name=$MODEL_NAME --model_type=$MODEL_TYPE --dtype=$DTYPE --dataset_id=$DATASET --output_dir=$OUTPUT_DIR"

echo $CMD
$CMD