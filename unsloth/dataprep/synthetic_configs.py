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

synthetic_qa_config = """\
# Master configuration file for Synthetic Data Kit

# Global paths configuration
paths:
  # Input data locations
  input:
    pdf: "{data_output_location}/pdf"
    html: "{data_output_location}/html"
    youtube: "{data_output_location}/youtube"
    docx: "{data_output_location}/docx"
    ppt: "{data_output_location}/ppt"
    txt: "{data_output_location}/txt"

  # Output locations
  output:
    parsed: "{data_output_location}/output"      # Where parsed text files are saved
    generated: "{data_output_location}/generated" # Where generated content is saved
    cleaned: "{data_output_location}/cleaned"     # Where cleaned content is saved
    final: "{data_output_location}/final"         # Where final formatted content is saved

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