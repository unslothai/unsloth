MODEL_NAME=$1
MODEL_TYPE=$2
DTYPE=$3
OUTPUT_DIR=./results/${MODEL_NAME}_${MODEL_TYPE}_${DTYPE}
#NOTE: If running huggingface default models, need to change the following line for backward profiling hooks to work
#peft.tuners.lora.gptq.QuantLinear forward so that result is not updated in place
#https://github.com/huggingface/peft/blob/bfc102c0c095dc9094cdd3523b729583bfad4688/src/peft/tuners/lora/gptq.py#L70
#Original
#result += output
#return result
#New
#return result + output
#Also need to patch https://github.com/huggingface/peft/blob/bfc102c0c095dc9094cdd3523b729583bfad4688/src/peft/tuners/lora/layer.py#L320
#Make output_dir if not exists
mkdir -p $OUTPUT_DIR
CMD="python test_unsloth.py --model_name=$MODEL_NAME --model_type=$MODEL_TYPE --dtype=$DTYPE --profile --output_dir=$OUTPUT_DIR 2>&1 | tee $OUTPUT_DIR/profile.log"

echo $CMD
eval $CMD