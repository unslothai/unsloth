from typing import Optional


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
    "SyntheticDataKit",
]
import subprocess
import time
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import requests
import torch
import gc
import time
from unsloth_zoo.vllm_utils import (
    load_vllm,
    patch_vllm,
    delete_vllm,
)
import numpy as np

from .synthetic_configs import (
    synthetic_qa_config,
)

class SyntheticDataKit:
    """
    A class that provides tools for generating synthetic data using a pretrained language model.
    
    Args:
        model_name (`str`):
            Name or path of the pretrained model.
        max_seq_length (`int`):
            Maximum sequence length for the model.
        gpu_memory_utilization (`float`):
            Fraction of GPU memory to use for the model.
        float8_kv_cache (`bool`):
            Whether to use float8 precision for the KV cache.
        conservativeness (`float`):
            Level of conservativeness for the generation process.
        token (`str`, *optional*):
            Authentication token for private models.
    """
    def __init__(
        self,
        model_name: str               = "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        max_seq_length: int           = 2048,
        gpu_memory_utilization: float = 0.98,
        float8_kv_cache: bool         = False,
        conservativeness: float       = 1.0,
        token: Optional[str]          = None,
        **kwargs,
    ):
        assert(type(model_name) is str)
        assert(type(max_seq_length) is int)
        assert(type(gpu_memory_utilization) is float)
        assert(type(float8_kv_cache) is bool)
        assert(type(conservativeness) is float)
        assert(token is None or type(token) is str)

        self.model_name = model_name
        self.max_seq_length = max_seq_length

        from transformers import AutoConfig, AutoTokenizer
        self.config = AutoConfig.from_pretrained(
            model_name,
            token = token,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token = token,
        )
        patch_vllm(debug = False)
        engine_args = load_vllm(
            model_name             = model_name,
            config                 = self.config,
            gpu_memory_utilization = gpu_memory_utilization,
            max_seq_length         = max_seq_length,
            disable_log_stats      = True,
            float8_kv_cache        = float8_kv_cache,
            conservativeness       = conservativeness,
            return_args            = True,
            enable_lora            = False,
            use_bitsandbytes       = False,
            **kwargs,
        )

        if "device" in engine_args: del engine_args["device"]
        if "model"  in engine_args: del engine_args["model"]
        if "compilation_config" in engine_args:
            # Cannot parse in vllm serve
            engine_args["compilation_config"] = 3

        subprocess_commands = [
            "vllm", "serve", str(model_name),
        ]
        for key, value in engine_args.items():
            flag  = key.replace("_", "-")
            which = str(value).replace("torch.", "")
            if which == "True":
                # Ignore --enforce-eager True
                subprocess_commands += ["--" + flag,]
            elif which == "False":
                # Ignore flag
                pass
            else:
                subprocess_commands += ["--" + flag, which,]
        pass
        vllm_process = subprocess.Popen(
            subprocess_commands,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            start_new_session = True,
        )
        self.vllm_process = vllm_process

        ready_message_part = b"Starting vLLM API server on"
        ready = False
        while vllm_process.poll() is None:
            output = vllm_process.stdout.readline()
            if not output:
                print("Stdout stream ended before readiness message detected.")
                break
            output_str = output.decode('utf-8', errors='ignore').strip()
            if "platform is" not in output_str:
                print(f"vLLM STDOUT: {output_str}")
            if ready_message_part in output:
                print(f"\n--- vLLM Server Ready (Detected: '{ready_message_part.decode()}') ---")
                ready = True
                break
            pass
        pass
        if vllm_process is None:
            raise RuntimeError("Unsloth: vllm_process failed to load!")
        trial = 0
        while not self.check_vllm_status():
            if trial >= 100:
                raise RuntimeError("Unsloth: vllm_process failed to load!")
            trial += 1
            time.sleep(1)
        return
    pass

    @staticmethod
    def from_pretrained(
        model_name: str               = "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        max_seq_length: int           = 2048,
        gpu_memory_utilization: float = 0.9,
        float8_kv_cache: bool         = False,
        conservativeness: float       = 1.0,
        token: Optional[str]          = None,
        **kwargs,
    ) -> SyntheticDataKit:
        """
        Load a pretrained SyntheticDataKit instance.
        
        Args:
            model_name (`str`):
                Name or path of the pretrained model.
            max_seq_length (`int`):
                Maximum sequence length for the model.
            gpu_memory_utilization (`float`):
                Fraction of GPU memory to use for the model.
            float8_kv_cache (`bool`):
                Whether to use float8 precision for the KV cache.
            conservativeness (`float`):
                Level of conservativeness for the generation process.
            token (`str`, *optional*):
                Authentication token for private models.
            **kwargs:
                Additional keyword arguments to pass to the SyntheticDataKit constructor.
        
        Returns:
            `SyntheticDataKit`: A new instance of SyntheticDataKit with the specified configuration.
        """
        return SyntheticDataKit(
            model_name = model_name,
            max_seq_length = max_seq_length,
            gpu_memory_utilization = gpu_memory_utilization,
            float8_kv_cache = float8_kv_cache,
            conservativeness = conservativeness,
            token = token,
            **kwargs,
        )
    pass

    @staticmethod
    def check_vllm_status() -> bool:
        """
        Check if the vLLM server is running and accessible.
        
        Returns:
            `bool`: True if the server is running and accessible, False otherwise.
        """
        try:
            response = requests.get("http://localhost:8000/metrics")
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            return False
        pass
    pass

    def cleanup(self) -> None:
        """
        Clean up resources by terminating the vLLM server and releasing GPU memory.
        
        This method attempts to gracefully terminate the server process, falling back to a forceful kill if necessary.
        It also clears the CUDA cache and garbage collects to free up memory.
        """
        if not hasattr(self, "vllm_process"): return

        vllm_process = self.vllm_process
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

        # Delete vLLM module as well
        delete_vllm(llm = None)
    pass

    def __enter__(self) -> SyntheticDataKit: return self
    def __exit__(self, *exc) -> None: self.cleanup()
    def __del__(self) -> None: self.cleanup()

    def chunk_data(self, filename: str = None) -> list[str]:
        """
        Split a text file into chunks suitable for question-answer generation.
        
        The chunk length is calculated as the model's max context length minus twice the 
        max generation tokens and an additional 128-token buffer to reduce errors:
        chunk_length = max_seq_length - (max_generation_tokens * 2) - 128
        
        Args:
            filename (`str`):
                Path to the input text file.
        
        Returns:
            `list[str]`: List of paths to the generated chunk files.
        """
        # Chunks data by max tokens and generation length
        assert(filename is not None)
        assert(os.path.exists(filename))
        assert(hasattr(self, "tokenizer"))
        if not hasattr(self, "max_seq_length"):
            raise RuntimeError("Please use SynthetidDataKit.from_pretrained(...) first!")
        if not hasattr(self, "overlap") or not hasattr(self, "max_generation_tokens"):
            raise RuntimeError("Please use prepare_qa_generation first!")

        with open(filename, "r") as f: text = f.read()

        max_tokens = self.max_seq_length - self.max_generation_tokens*2 - 128 # -128 to reduce errors
        if max_tokens <= 5:
            raise RuntimeError("Generation length is way too long!")
        input_ids = self.tokenizer(text, add_special_tokens = False).input_ids

        # Get left and right boundaries
        length = len(input_ids)
        n_chunks = int(np.ceil(length / (max_tokens - self.overlap)))
        boundaries = np.ceil(np.linspace(0, length - self.overlap, n_chunks)).astype(int)
        boundaries = np.stack((boundaries[:-1], (boundaries + self.overlap)[1:])).T
        boundaries = np.minimum(boundaries, length).tolist()

        # Get extension of filename like .txt
        filename, extension = os.path.splitext(filename)
        if filename.endswith("/"): filename = filename[:-1]

        all_filenames = []
        for i, (left, right) in enumerate(boundaries):
            chunked_text = self.tokenizer.decode(input_ids[left : right])
            new_filename = f"{filename}_{i}{extension}"
            all_filenames.append(new_filename)
            with open(new_filename, "w") as f: f.write(chunked_text)
        pass
        return all_filenames
    pass

    def prepare_qa_generation(
        self,
        output_folder: str         = "data",
        max_generation_tokens: int = 512,
        temperature: float         = 0.7,
        top_p: float               = 0.95,
        overlap: int               = 64,
        default_num_pairs: int = 25,
        cleanup_threshold: float   = 1.0,
        cleanup_batch_size: int    = 4,
        cleanup_temperature: float = 0.3,
    ) -> None:
        """
        Prepare the configuration for question-answer generation.
        
        Args:
            output_folder (`str`):
                Directory where generated data will be saved.
            max_generation_tokens (`int`):
                Maximum number of tokens to generate for each question-answer pair.
            temperature (`float`):
                Sampling temperature for generation.
            top_p (`float`):
                Nucleus sampling parameter.
            overlap (`int`):
                Number of tokens to overlap between chunks.
            default_num_pairs (`int`):
                Default number of question-answer pairs to generate per chunk.
            cleanup_threshold (`float`):
                Threshold for filtering low-quality generated pairs.
            cleanup_batch_size (`int`):
                Batch size for the cleanup process.
            cleanup_temperature (`float`):
                Temperature for the cleanup generation process.
        """
        assert(hasattr(self, "model_name"))
        assert(hasattr(self, "max_seq_length"))
        assert(max_generation_tokens < self.max_seq_length)

        locations = "pdf,html,youtube,docx,ppt,txt,output,generated,cleaned,final"
        locations = locations.split(",")
        for path in locations:
            os.makedirs(os.path.join(output_folder, path), exist_ok = True)
        pass

        self.max_generation_tokens = max_generation_tokens

        config = synthetic_qa_config\
            .replace("{data_output_location}", str(output_folder))\
            .replace("{model_name}", str(self.model_name))\
            .replace("{temperature}", str(temperature))\
            .replace("{top_p}", str(top_p))\
            .replace("{chunk_size}", str(self.max_seq_length - max_generation_tokens*2 - 2))\
            .replace("{overlap}", str(overlap))\
            .replace("{max_tokens}", str(max_generation_tokens))\
            .replace("{default_num_pairs}", str(default_num_pairs))\
            .replace("{cleanup_threshold}", str(cleanup_threshold))\
            .replace("{cleanup_batch_size}", str(cleanup_batch_size))\
            .replace("{cleanup_temperature}", str(cleanup_temperature))

        with open("synthetic_data_kit_config.yaml", "w") as f: f.write(config)

        self.overlap = overlap
    pass
pass
