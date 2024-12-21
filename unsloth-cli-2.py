#!/usr/bin/env python3

"""
🦥 Enhanced Script for Fine-Tuning, Merging, and Managing Language Models with Unsloth

This script significantly extends the original unsloth-cli.py with a wide range of advanced features:

- Comprehensive training pipeline with validation and testing support
- Advanced error handling and fallback options for robust model loading
- Merging functionality for LoRA adapters with dequantization options
- Flexible quantization and precision control (4-bit, 16-bit, 32-bit)
- Support for custom datasets and data formats (JSON, Parquet)
- Integration with Hugging Face models and push-to-hub functionality
- GGUF conversion for optimized model deployment
- Enhanced logging and progress tracking

Key Features:

1. Flexible Data Handling:
   - Support for Parquet, JSON, and other data formats
   - Custom data parsing and processing pipelines

2. Advanced Model Management:
   - Load and save models in various formats (Hugging Face, GGUF)
   - Quantization options for memory-efficient training and inference
   - Dequantization capabilities for precision-sensitive operations

3. Comprehensive Training Pipeline:
   - Support for train, validation, and test datasets
   - Customizable training parameters (batch size, learning rate, etc.)
   - Integration with popular optimization techniques (LoRA, gradient checkpointing)

4. Merging and Adaptation:
   - Merge LoRA adapters with base models
   - Dequantization options for merging quantized models
   #TODO @9/17/2024 ADD LoRA+LoRA and Model+Model Merging Methods (lazy Merge kit): To merge fully 
   #TODO @9/17/2024 ADD DIRECT QUANTIZER FOR SAFETENSOR: To quantize safetensors and support both gguf and safetensors

5. Deployment and Sharing:
   - GGUF conversion for optimized model deployment
   - Direct integration with Hugging Face Hub for easy model sharing

6. Robust Error Handling and Logging:
   - Detailed error messages and logging for easier debugging
   - Fallback options for model loading and processing

Usage example for training:
    python unsloth-cli-2.py train --model_name "your_model_path" --train_dataset "train.parquet" \
    --validation_dataset "val.parquet" --test_dataset "test.parquet" \
    --max_seq_length 2048 --load_in_4bit \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
    --max_steps 1000 --learning_rate 2e-5 --output_dir "outputs" \
    --save_model --save_path "model" --quantization "q4_k_m" \
    --push_to_hub --hub_path "your_hub/model" --hub_token "your_token"

Usage example for merging:
    python unsloth-cli-2.py merge --base_model_path "path/to/base/model" \
    --adapter_path "path/to/adapter" --output_path "path/to/output" \
    --dequantize f16

Usage example with dequantization:
    python unsloth-cli-2.py train --model_name "your_model_path" --train_dataset "train.parquet" \
    --load_in_4bit --dequantize

Dequantization Feature:
The --dequantize option allows you to convert quantized weights back to full precision
after loading the model. This can be useful when you want to fine-tune or use a
previously quantized model in full precision. However, please note:

1. Dequantization does not recover information lost during the initial quantization.
   The quality of the model may still be lower than if it was originally trained in
   full precision.

2. Dequantizing increases memory usage significantly, as it converts weights to
   full precision (typically float32).

3. This option is most useful when you need to perform operations that require
   full precision weights but want to start from a quantized model.

To see a full list of configurable options, use:
    python unsloth-cli-2.py train --help
    python unsloth-cli-2.py merge --help

Happy fine-tuning, merging, and deploying with Unsloth! 🦥🚀
"""

import argparse
import logging
import os
import torch
import json
import struct
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from datasets import load_dataset, DatasetDict
from peft import PeftModel
from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_safetensors_header(file_path):
    try:
        with open(file_path, 'rb') as f:
            header_size_bytes = f.read(8)
            header_size = struct.unpack('<Q', header_size_bytes)[0]
            logger.info(f"SafeTensors header size: {header_size} bytes")
            
            header_json = f.read(header_size).decode('utf-8')
            header = json.loads(header_json)
            logger.info(f"SafeTensors header content (first 1000 chars): {json.dumps(header)[:1000]}...")
            
            return header_size, header
    except Exception as e:
        logger.error(f"Error analyzing SafeTensors header: {e}")
        return None, None

def load_model_and_tokenizer(args):
    logger.info(f"Attempting to load model from: {args.model_name}")
    
    # Check file permissions
    model_dir = args.model_name if os.path.isdir(args.model_name) else os.path.dirname(args.model_name)
    logger.info(f"Checking permissions for model directory: {model_dir}")
    logger.info(f"Directory permissions: {oct(os.stat(model_dir).st_mode)[-3:]}")
    
    model_files = os.listdir(model_dir)
    logger.info(f"Files in model directory: {model_files}")
    
    safetensors_file = next((f for f in model_files if f.endswith('.safetensors')), None)
    if safetensors_file:
        safetensors_path = os.path.join(model_dir, safetensors_file)
        logger.info(f"Analyzing SafeTensors file: {safetensors_path}")
        header_size, header = analyze_safetensors_header(safetensors_path)
        if header:
            logger.info(f"Total number of tensors in SafeTensors: {len(header)}")
    
    try:
        if args.load_in_16bit:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                max_seq_length=args.max_seq_length,
                dtype=torch.float16,
                load_in_4bit=False,
            )
        elif args.load_in_4bit:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                max_seq_length=args.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                max_seq_length=args.max_seq_length,
                dtype=args.dtype,
                load_in_4bit=False,
            )
        
        if args.dequantize:
            logger.info("Dequantizing model weights...")
            model = dequantize_weights(model)
            
        logger.info(f"Model loaded successfully with FastLanguageModel in {'16-bit' if args.load_in_16bit else '4-bit' if args.load_in_4bit else 'default'} precision")
    except Exception as e:
        logger.warning(f"Failed to load with FastLanguageModel: {e}")
        logger.info("Falling back to standard HuggingFace loading...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            if args.load_in_16bit:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            elif args.load_in_4bit:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    load_in_4bit=True,
                    device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name,
                    torch_dtype=args.dtype,
                    device_map="auto"
                )
                
            if args.dequantize:
                logger.info("Dequantizing model weights...")
                model = dequantize_weights(model)
                
            logger.info(f"Model loaded successfully with standard HuggingFace method in {'16-bit' if args.load_in_16bit else '4-bit' if args.load_in_4bit else 'default'} precision")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    return model, tokenizer

def merge_adapter(base_model_path, adapter_path, output_path, dequantize='no'):
    """
    Merge a LoRA adapter into a base model, with optional dequantization.
    
    Args:
        base_model_path: Path to the base model
        adapter_path: Path to the LoRA adapter
        output_path: Path to save the merged model
        dequantize: Dequantization option ('no', 'f16', or 'f32')
    
    Returns:
        A dictionary with the merge result and output path
    """
    logger.info(f"Merging adapter from {adapter_path} into base model {base_model_path}")
    
    try:
        logger.info(f"Loading base model from: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        logger.info(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        if dequantize != 'no':
            logger.info(f"Dequantizing model to {dequantize}")
            target_dtype = torch.float16 if dequantize == 'f16' else torch.float32
            model = dequantize_weights(model, target_dtype)
        
        logger.info("Merging adapter with base model")
        merged_model = model.merge_and_unload()
        
        logger.info(f"Saving merged model to: {output_path}")
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        logger.info("Merged model saved successfully")
        
        return {"message": "Adapter merged successfully", "output_path": output_path}
    except Exception as e:
        error_msg = f"Error merging adapter: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    
def quantize_safetensor(input_path, output_path, bits=8):
    """
    Quantize a safetensor model to a lower bit precision.
    
    Args:
        input_path: Path to the input safetensor model
        output_path: Path to save the quantized model
        bits: Target bit precision (default: 8)
    
    Returns:
        True if quantization was successful, False otherwise
    """
    logger.info(f"Quantizing model from {input_path} to {bits}-bit precision")
    
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        # Load the model
        with safe_open(input_path, framework="pt", device="cpu") as f:
            model_data = {key: f.get_tensor(key) for key in f.keys()}

        # Quantize the tensors
        quantized_data = {}
        for key, tensor in model_data.items():
            if tensor.dtype in [torch.float32, torch.float16]:
                if bits == 8:
                    quantized_tensor = tensor.to(torch.int8)
                elif bits == 4:
                    # For 4-bit quantization, we need to implement a custom method
                    # This is a simplified 4-bit quantization, you might want to use a more sophisticated method
                    float_tensor = tensor.float()
                    max_val = float_tensor.abs().max()
                    scale = max_val / 7.5
                    quantized_tensor = (float_tensor / scale).round().clamp(-8, 7).to(torch.int8)
                    quantized_data[f"{key}_scale"] = torch.tensor([scale])
                else:
                    raise ValueError(f"Unsupported bit precision: {bits}")
                
                quantized_data[key] = quantized_tensor
            else:
                quantized_data[key] = tensor

        # Save the quantized model
        save_file(quantized_data, output_path)
        logger.info(f"Quantized model saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error during model quantization: {e}")
        return False
    
def dequantize_4bit_to_8bit(model):
    for name, param in model.named_parameters():
        if param.dtype == torch.int8:  # Assuming 4-bit weights are stored as int8
            # Convert 4-bit to 8-bit
            param.data = (param.data.float() / 16 * 255).round().clamp(0, 255).byte()
    return model

def dequantize_model(input_path, output_path, precision):
    logger.info(f"Dequantizing model from {input_path} to {output_path}")
    
    try:
        import torch

        model = AutoModelForCausalLM.from_pretrained(input_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(input_path)

        if precision == '8bit':
            model = dequantize_4bit_to_8bit(model)
        else:
            target_dtype = torch.float16 if precision == 'f16' else torch.float32
            model = dequantize_weights(model, target_dtype)

        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        logger.info(f"Model dequantized and saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error during model dequantization: {e}")
        return False

def dequantize_weights(model, target_dtype=torch.float32):
    """
    Dequantize the weights of a quantized model to a target data type.
    This is useful when merging models or performing operations that require full precision.
    
    Args:
        model: The quantized model to dequantize
        target_dtype: The target data type (default: torch.float32)
    
    Returns:
        The dequantized model
    """
    def dequantize_layer(layer):
        for name, param in layer.named_parameters():
            if hasattr(param, 'quant_state'):
                # Dequantize quantized parameters
                param.data = param.data.dequantize().to(target_dtype)
                delattr(param, 'quant_state')
            elif param.dtype != target_dtype:
                # Convert non-quantized parameters to the target dtype
                param.data = param.data.to(target_dtype)

    for module in model.modules():
        dequantize_layer(module)

    return model

def run_merge(args):
    """
    Execute the merge operation based on command-line arguments.
    
    Args:
        args: Command-line arguments
    
    Returns:
        The result of the merge operation
    """
    result = merge_adapter(args.base_model_path, args.adapter_path, args.output_path, args.dequantize)
    if "error" in result:
        logger.error(result["error"])
    else:
        logger.info(result["message"])
    return result

def run_dequantize(args):
    """
    Execute the dequantize operation based on command-line arguments.
    
    Args:
        args: Command-line arguments
    
    Returns:
        True if dequantization was successful, False otherwise
    """
    try:
        logger.info(f"Dequantizing model from {args.input_path} to {args.output_path}")
        success = dequantize_model(args.input_path, args.output_path, args.precision)
        if success:
            logger.info("Dequantization completed successfully")
        else:
            logger.error("Dequantization failed")
        return success
    except Exception as e:
        logger.error(f"Error during dequantization: {str(e)}")
        return False
    
def run_quantize(args):
    """
    Execute the quantize operation based on command-line arguments.
    
    Args:
        args: Command-line arguments
    
    Returns:
        True if quantization was successful, False otherwise.
    """
    try:
        logger.info(f"Quantizing model from {args.input_path} to {args.output_path}")
        success = quantize_safetensor(args.input_path, args.output_path, args.bits)
        if success:
            logger.info("Quantization completed successfully")
        else:
            logger.error("Quantization failed")
        return success
    except Exception as e:
        logger.error(f"Error during quantization: {str(e)}")
        return False
    
def run_train(args):
    """
    Execute the train operation based on command-line arguments.
    
    Args:
        args: Command-line arguments
    
    Returns:
        True if quantization was successful, False otherwise
    """
    model, tokenizer = load_model_and_tokenizer(args)

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
        use_rslora=args.use_rslora,
        loftq_config=args.loftq_config,
    )

    logger.info('=== Loading and Formatting Datasets ===')
    train_dataset = load_dataset("parquet", data_files=args.train_dataset)['train']
    
    eval_dataset = None
    if args.validation_dataset:
        eval_dataset = load_dataset("parquet", data_files=args.validation_dataset)['train']
    
    test_dataset = None
    if args.test_dataset:
        test_dataset = load_dataset("parquet", data_files=args.test_dataset)['train']

    def formatting_prompts_func(examples, is_test=False):
        instructions = examples.get("instruction", [""] * len(examples["input"]))
        inputs = examples["input"]
        if not is_test:
            outputs = examples["output"]
            texts = [f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
                     for instruction, input, output in zip(instructions, inputs, outputs)]
        else:
            texts = [f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
                     for instruction, input in zip(instructions, inputs)]
        return {"text": [text + tokenizer.eos_token for text in texts]}

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    if eval_dataset:
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
    if test_dataset:
        test_dataset = test_dataset.map(lambda x: formatting_prompts_func(x, is_test=True), batched=True)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Validation dataset size: {len(eval_dataset)}")
    if test_dataset:
        logger.info(f"Test dataset size: {len(test_dataset)}")

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim=args.optim,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=args.report_to,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=args.logging_steps if eval_dataset else None,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    logger.info('=== Starting Training ===')
    trainer_stats = trainer.train()
    logger.info(trainer_stats)

    if test_dataset:
        logger.info('=== Generating Responses for Test Dataset ===')
        generated_outputs = []
        for item in test_dataset:
            input_text = item['text']
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
            generated_ids = model.generate(input_ids, max_new_tokens=512)  # Adjust max_new_tokens as needed
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            generated_outputs.append(generated_text)
        
        # Create test_answered dataset
        test_answered = test_dataset.add_column("generated_output", generated_outputs)
        
        # Save test_answered dataset
        test_answered.to_parquet(os.path.join(args.output_dir, "test_answered.parquet"))
        logger.info(f"Test set with generated responses saved to {os.path.join(args.output_dir, 'test_answered.parquet')}")

    if args.save_model:
        logger.info('=== Saving Model ===')
        if args.save_gguf:
            if isinstance(args.quantization, list):
                for quantization_method in args.quantization:
                    logger.info(f"Saving model with quantization method: {quantization_method}")
                    model.save_pretrained_gguf(
                        args.save_path,
                        tokenizer,
                        quantization_method=quantization_method,
                    )
                    if args.push_model:
                        model.push_to_hub_gguf(
                            hub_path=args.hub_path,
                            hub_token=args.hub_token,
                            quantization_method=quantization_method,
                        )
            else:
                logger.info(f"Saving model with quantization method: {args.quantization}")
                model.save_pretrained_gguf(args.save_path, tokenizer, quantization_method=args.quantization)
                if args.push_model:
                    model.push_to_hub_gguf(
                        hub_path=args.hub_path,
                        hub_token=args.hub_token,
                        quantization_method=args.quantization,
                    )
        else:
            model.save_pretrained_merged(args.save_path, tokenizer, args.save_method)
            if args.push_model:
                model.push_to_hub_merged(args.save_path, tokenizer, args.hub_token)
    else:
        logger.warning("The model is not saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🦥 Enhanced fine-tuning, merging, and dequantizing script using Unsloth")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Training parser
    train_parser = subparsers.add_parser("train", help="Train a model")
    
    model_group = train_parser.add_argument_group("🤖 Model Options")
    model_group.add_argument('--model_name', type=str, required=True, help="Model name or path to load")
    model_group.add_argument('--max_seq_length', type=int, default=2048, help="Maximum sequence length")
    model_group.add_argument('--dtype', type=str, default=None, help="Data type for model (None for auto detection)")
    model_group.add_argument('--load_in_4bit', action='store_true', help="Use 4-bit quantization")
    model_group.add_argument('--load_in_16bit', action='store_true', help="Use 16-bit precision")
    model_group.add_argument('--train_dataset', type=str, required=True, help="Path to the training parquet dataset file")
    model_group.add_argument('--validation_dataset', type=str, help="Path to the validation parquet dataset file")
    model_group.add_argument('--test_dataset', type=str, help="Path to the test parquet dataset file")
    model_group.add_argument('--dequantize', action='store_true', help="Dequantize model weights after loading")

    # LoRA group
    lora_group = train_parser.add_argument_group("🧠 LoRA Options")
    lora_group.add_argument('--r', type=int, default=16, help="Rank for LoRA model")
    lora_group.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha parameter")
    lora_group.add_argument('--lora_dropout', type=float, default=0, help="LoRA dropout rate")
    lora_group.add_argument('--bias', type=str, default="none", help="Bias setting for LoRA")
    lora_group.add_argument('--use_gradient_checkpointing', type=str, default="unsloth", help="Use gradient checkpointing")
    lora_group.add_argument('--random_state', type=int, default=3407, help="Random state for reproducibility")
    lora_group.add_argument('--use_rslora', action='store_true', help="Use rank stabilized LoRA")
    lora_group.add_argument('--loftq_config', type=str, default=None, help="Configuration for LoftQ")
    
    # Training options group
    training_group = train_parser.add_argument_group("🎓 Training Options")
    training_group.add_argument('--per_device_train_batch_size', type=int, default=2, help="Batch size per device during training")
    training_group.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Number of gradient accumulation steps")
    training_group.add_argument('--warmup_steps', type=int, default=5, help="Number of warmup steps")
    training_group.add_argument('--max_steps', type=int, default=400, help="Maximum number of training steps")
    training_group.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate")
    training_group.add_argument('--optim', type=str, default="adamw_8bit", help="Optimizer type")
    training_group.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
    training_group.add_argument('--lr_scheduler_type', type=str, default="linear", help="Learning rate scheduler type")
    training_group.add_argument('--seed', type=int, default=3407, help="Seed for reproducibility")

    # Report group
    report_group = train_parser.add_argument_group("📊 Report Options")
    report_group.add_argument('--report_to', type=str, default="tensorboard", choices=["azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "neptune", "tensorboard", "wandb", "all", "none"], help="The list of integrations to report the results and logs to")
    report_group.add_argument('--logging_steps', type=int, default=1, help="Logging steps")

    # Save group
    save_group = train_parser.add_argument_group('💾 Save Model Options')
    save_group.add_argument('--output_dir', type=str, default="outputs", help="Output directory")
    save_group.add_argument('--save_model', action='store_true', help="Save the model after training")
    save_group.add_argument('--save_method', type=str, default="merged_16bit", choices=["merged_16bit", "merged_4bit", "lora"], help="Save method for the model")
    save_group.add_argument('--save_gguf', action='store_true', help="Convert the model to GGUF after training")
    save_group.add_argument('--save_path', type=str, default="model", help="Path to save the model")
    save_group.add_argument('--quantization', type=str, default="q8_0", nargs="+", help="Quantization method for saving the model")

    # Push group
    push_group = train_parser.add_argument_group('🚀 Push Model Options')
    push_group.add_argument('--push_model', action='store_true', help="Push the model to Hugging Face hub after training")
    push_group.add_argument('--push_gguf', action='store_true', help="Push the model as GGUF to Hugging Face hub after training")
    push_group.add_argument('--hub_path', type=str, default="hf/model", help="Path on Hugging Face hub to push the model")
    push_group.add_argument('--hub_token', type=str, help="Token for pushing the model to Hugging Face hub")

    # Merging parser
    merge_parser = subparsers.add_parser("merge", help="Merge a LoRA adapter")
    merge_parser.add_argument('--base_model_path', type=str, required=True, help="Path to the base model")
    merge_parser.add_argument('--adapter_path', type=str, required=True, help="Path to the LoRA adapter")
    merge_parser.add_argument('--output_path', type=str, required=True, help="Path to save the merged model")
    merge_parser.add_argument('--dequantize', choices=['no', 'f16', 'f32'], default='no', help="Dequantize LoRA weights before merging")

    # Dequantizing parser
    dequantize_parser = subparsers.add_parser("dequantize", help="Dequantize a model")
    dequantize_parser.add_argument('--input_path', type=str, required=True, help="Path to the input model")
    dequantize_parser.add_argument('--output_path', type=str, required=True, help="Path to save the dequantized model")
    dequantize_parser.add_argument('--precision', choices=['f16', 'f32', '8bit'], default='f16', help="Precision for dequantization")
    
    # Quantizing parser
    quantize_parser = subparsers.add_parser("quantize", help="Quantize a safetensor model")
    quantize_parser.add_argument('--input_path', type=str, required=True, help="Path to the input safetensor model")
    quantize_parser.add_argument('--output_path', type=str, required=True, help="Path to save the quantized model")
    quantize_parser.add_argument('--bits', type=int, choices=[4, 8], default=8, help="Target bit precision for quantization")

    args = parser.parse_args()

    if args.command == "train":
        if args.load_in_4bit and args.load_in_16bit:
            logger.error("Cannot use both 4-bit and 16-bit options simultaneously. Please choose one.")
            exit(1)
        try:
            run_train(args)
        except Exception as e:
            logger.error(f"An error occurred during execution: {e}")
        logger.info("Attempting to provide more information about the model...")
        
        model_dir = args.model_name if os.path.isdir(args.model_name) else os.path.dirname(args.model_name)
        logger.info(f"Model directory contents: {os.listdir(model_dir)}")
        config_file = os.path.join(model_dir, "config.json")
       
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Model config: {json.dumps(config, indent=2)}")
        
        logger.info("If the issue persists, please check the following:")
        logger.info("1. Ensure you have the necessary permissions to read the model files.")
        logger.info("2. Verify that the model files are not corrupted.")
        logger.info("3. Check if the model is compatible with the current version of transformers and safetensors.")
        logger.info("4. Try updating your libraries: pip install --upgrade transformers safetensors")
        
    elif args.command == "merge":
        result = run_merge(args)
        if "error" in result:
            logger.error(result["error"])
        else:
            logger.info(result["message"])
           
    elif args.command == "dequantize":
        success = run_dequantize(args)
        if not success:
            logger.error("Dequantization failed")
            exit(1)
           
    elif args.command == "quantize":
        success = run_quantize(args)
        if not success:
            logger.error("Quantization failed")
            exit(1)
    else:
        parser.print_help()
