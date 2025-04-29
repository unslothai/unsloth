# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "check_vllm_status",
    "async_load_vllm",
    "destroy_vllm",
    "configure_synthetic_data_kit",
]
import subprocess
import time
import os
import requests
import torch
import gc

def check_vllm_status():
    try:
        response = requests.get("http://localhost:8000/metrics")
        if response.status_code == 200:
            print("vllm server is running")
            return True
    except requests.exceptions.ConnectionError:
        print("vllm server is not running")
        return False
    pass
pass


def async_load_vllm(
    model_name = "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
    max_model_len = 10000,
    gpu_memory_utilization = 0.85,
):
    vllm_process = subprocess.Popen([
            'vllm', 'serve',
            str(model_name),
            '--trust-remote-code',
            '--dtype', 'half',
            '--max-model-len', str(max_model_len),
            '--enable-chunked-prefill', 'true',
            '--quantization', 'bitsandbytes',
            '--gpu-memory-utilization', str(gpu_memory_utilization),
            '--swap_space', '4',
        ],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        start_new_session = True,
    )
    ready_message_part = b"Starting vLLM API server on"
    ready = False
    while vllm_process.poll() is None:
        output = vllm_process.stdout.readline()
        if not output:
            print("Stdout stream ended before readiness message detected.")
            break
        output_str = output.decode('utf-8', errors='ignore').strip()
        print(f"vLLM STDOUT: {output_str}")
        if ready_message_part in output:
            print(f"\n--- vLLM Server Ready (Detected: '{ready_message_part.decode()}') ---")
            ready = True
            break
        pass
    pass
    if vllm_process is None:
        raise RuntimeError("Unsloth: vllm_process failed to load!")
    check_vllm_status()
    return vllm_process
pass


def destroy_vllm(vllm_process):
    print("Attempting to terminate the VLLM server gracefully...")
    try:
        vllm_process.terminate()
        vllm_process.wait(timeout=10)
        print("Server terminated gracefully.")
    except subprocess.TimeoutExpired:
        print("Server did not terminate gracefully after 10 seconds. Forcing kill...")
        vllm_process.kill()
        vllm_process.wait()
        print("Server killed forcefully.")
    except Exception as e:
         print(f"An error occurred while trying to stop the process: {e}")
         try:
             if vllm_process.poll() is None:
                 print("Attempting forceful kill due to error...")
                 vllm_process.kill()
                 vllm_process.wait()
                 print("Server killed forcefully after error.")
         except Exception as kill_e:
             print(f"Error during forceful kill: {kill_e}")
    for _ in range(10):
        torch.cuda.empty_cache()
        gc.collect()
pass


synthetic_config_string = """\
# Master configuration file for Synthetic Data Kit

# Global paths configuration
paths:
  # Input data locations
  input:
    pdf: "data/pdf"
    html: "data/html"
    youtube: "data/youtube"
    docx: "data/docx"
    ppt: "data/ppt"
    txt: "data/txt"

  # Output locations
  output:
    parsed: "data/output"      # Where parsed text files are saved
    generated: "data/generated" # Where generated content is saved
    cleaned: "data/cleaned"     # Where cleaned content is saved
    final: "data/final"         # Where final formatted content is saved

# VLLM server configuration
vllm:
  api_base: "http://localhost:8000/v1" # Base URL for VLLM API
  port: 8000                           # Port for VLLM server
  model: "{model_name}"                # Default model to use
  max_retries: 3                       # Number of retries for API calls
  retry_delay: 1.0                     # Initial delay between retries (seconds)

# Ingest configuration
ingest:
  default_format: "txt"  # Default output format for parsed files
  youtube_captions: "auto"  # Options: "auto", "manual" - caption preference

# LLM generation parameters
generation:
  temperature: {temperature}     # Higher = more creative, lower = more deterministic
  top_p: {top_p}                 # Nucleus sampling parameter
  chunk_size: {chunk_size}       # Size of text chunks for processing
  overlap: {overlap}             # Overlap between chunks to maintain context
  max_tokens: {max_tokens}       # Maximum tokens in LLM responses
  num_pairs: {default_num_pairs} # Default number of QA pairs to generate

# Content cleanup parameters
cleanup:
  threshold: {cleanup_threshold}       # Default quality threshold (1-10)
  batch_size: {cleanup_batch_size}     # Number of items per batch for rating
  temperature: {cleanup_temperature}   # Temperature for rating (lower = more consistent)

# Format conversion parameters
format:
  default: "jsonl"   # Default output format
  include_metadata: true  # Include metadata in output files
  pretty_json: true  # Use indentation in JSON output

# Prompts for different tasks
prompts:
  # Summary generation prompt
  summary: |
    Summarize this document in 3-5 sentences, focusing on the main topic and key concepts.

  # QA pair generation prompt
  qa_generation: |
    Create {num_pairs} question-answer pairs from this text for LLM training.

    Rules:
    1. Questions must be about important facts in the text
    2. Answers must be directly supported by the text
    3. Return JSON format only:

    [
      {{
        "question": "Question 1?",
        "answer": "Answer 1."
      }},
      {{
        "question": "Question 2?",
        "answer": "Answer 2."
      }}
    ]

    Text:
    {text}

  # QA pair rating prompt
  qa_rating: |
    Rate each of these question-answer pairs for quality and return exactly this JSON format:

    [
      {{"question": "same question text", "answer": "same answer text", "rating": n}}
    ]

    Where n is a number from 1-10.

    DO NOT include any text outside of the JSON array, just return valid JSON:

    {pairs}"""


def configure_synthetic_data_kit(
    model_name = "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
    temperature = 0.7,
    top_p = 0.95,
    chunk_size = 4000,
    overlap = 200,
    max_tokens = 512,
    default_num_pairs = 25,
    cleanup_threshold = 1.0,
    cleanup_batch_size = 4,
    cleanup_temperature = 0.3,
):
    config = synthetic_config_string\
        .replace("{model_name}", str(model_name))\
        .replace("{temperature}", str(temperature))\
        .replace("{top_p}", str(top_p))\
        .replace("{chunk_size}", str(chunk_size))\
        .replace("{overlap}", str(overlap))\
        .replace("{max_tokens}", str(max_tokens))\
        .replace("{default_num_pairs}", str(default_num_pairs))\
        .replace("{cleanup_threshold}", str(cleanup_threshold))\
        .replace("{cleanup_batch_size}", str(cleanup_batch_size))\
        .replace("{cleanup_temperature}", str(cleanup_temperature))

    return config
pass
