from unsloth import FastModel, FastVisionModel, UnslothVisionDataCollator
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, TrainingArguments
from datasets import load_dataset, Dataset
import torch
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from multiprocessing import Process, Queue
import gc

# ruff: noqa
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))
print(sys.path)


from tests.utils.cleanup_utils import safe_remove_directory
from tests.utils.perplexity_eval import ppl_model, add_to_comparison, print_model_comparison

# Define helper functions outside of main
def formatting_prompts_func(examples):
   convos = examples["messages"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

def load_and_compute_8bit_ppl(result_queue, load_in_4bit=False, load_in_8bit=False):
    """Load model and compute perplexity in subprocess"""
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    from tests.utils.perplexity_eval import ppl_model

    # Load model
    merged_model, merged_tokenizer = FastModel.from_pretrained(
        model_name="./unsloth_out/merged_gemma3_text_model",
        max_seq_length=2048,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    # Set up tokenizer
    merged_tokenizer = get_chat_template(
    merged_tokenizer,
    chat_template = "gemma-3",
)

    # Load dataset fresh in subprocess
    dataset_ppl = load_dataset("allenai/openassistant-guanaco-reformatted", split="eval")

    # Format the dataset
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [merged_tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
        return { "text" : texts, }
    dataset_ppl = dataset_ppl.map(formatting_prompts_func, batched=True)

    # Compute perplexity using the passed dataset
    ppl_value = ppl_model(merged_model, merged_tokenizer, dataset_ppl)


    # IMPORTANT: Convert to Python float if it's a tensor
    if torch.is_tensor(ppl_value):
        ppl_value = ppl_value.cpu().item()  # Move to CPU and convert to Python scalar
    elif hasattr(ppl_value, 'item'):
        ppl_value = ppl_value.item()  # Convert numpy or other array types
    else:
        ppl_value = float(ppl_value)  # Ensure it's a float

    # Return only the perplexity value
    result_queue.put(ppl_value)

    # Clean up
    del merged_model
    del merged_tokenizer
    del dataset_ppl
    torch.cuda.empty_cache()
    gc.collect()

# Main execution code should be wrapped in this guard
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'

    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-3-4b-it",
        max_seq_length=2048,
        dtype=compute_dtype,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        attn_implementation=attn_implementation
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",
    )

    from unsloth.chat_templates import standardize_sharegpt
    dataset_train = load_dataset("allenai/openassistant-guanaco-reformatted", split="train")
    dataset_ppl = load_dataset("allenai/openassistant-guanaco-reformatted", split="eval")

    dataset_train = dataset_train.map(formatting_prompts_func, batched=True)
    dataset_ppl = dataset_ppl.map(formatting_prompts_func, batched=True)

    add_to_comparison("Base model 4 bits", ppl_model(model, tokenizer, dataset_ppl))

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # Turn off for just text!
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # SHould leave on always!

        r=16,
        #target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
        target_modules = "all-linear",
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        modules_to_save=[
          "lm_head",
          "embed_tokens"
        ]
    )

    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        max_seq_length=2048,
        packing=False,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant":False},
            warmup_ratio=0.03,
            max_steps=40,
            learning_rate=3e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            #optim="adamw_8bit",
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
            max_grad_norm=0.3,
            dataset_num_proc=2,
        ),
    )

    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )

    # run training
    trainer_stats = trainer.train()

    add_to_comparison("Qlora model", ppl_model(model, tokenizer, dataset_ppl))

    # saving and merging the model to local disk
    print("merge and save to local disk")
    model.save_pretrained_merged(
        save_directory='./unsloth_out/merged_gemma3_text_model',
        tokenizer=tokenizer
    )


    # load model from local disk and test
    print("Loading merged model in 4 bit for perplexity test")
    merged_model, merged_tokenizer = FastModel.from_pretrained(
        model_name="./unsloth_out/merged_gemma3_text_model",
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
    )

    add_to_comparison("merged model load 4bit", ppl_model(merged_model, merged_tokenizer, dataset_ppl))


    print("Computing 8-bit model perplexity in subprocess...")
    result_queue = mp.Queue()
    p = mp.Process(target=load_and_compute_8bit_ppl, args=(result_queue, False, True))
    p.start()
    p.join()

    ppl_8bit = result_queue.get()
    add_to_comparison("merged model loaded 8bits", ppl_8bit)

    print("Loading merged model in 16 bit for perplexity test")
    merged_model, merged_tokenizer = FastModel.from_pretrained(
        model_name="./unsloth_out/merged_gemma3_text_model",
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_8bit=False,
    )

    add_to_comparison("merged model loaded 16bits", ppl_model(merged_model, merged_tokenizer, dataset_ppl))

    print_model_comparison()

    # final cleanup
    safe_remove_directory("./outputs")
    safe_remove_directory("./unsloth_compiled_cache")
    safe_remove_directory("./unsloth_out")
