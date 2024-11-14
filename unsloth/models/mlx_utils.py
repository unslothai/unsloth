# Copyright © 2023-2024 Apple Inc.

import glob
import json
import logging
from pathlib import Path
from typing import Generator

import mlx.core as mx
import mlx.nn as nn
from . import mlx_models as models
import transformers
from huggingface_hub import snapshot_download,create_repo
from unsloth.save import MODEL_CARD

def fetch_from_hub(hf_path: str):
    model_path = snapshot_download(
        repo_id=hf_path,
        allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
    )
    weight_files = glob.glob(f"{model_path}/*.safetensors")
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    config = transformers.AutoConfig.from_pretrained(hf_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        hf_path,
    )
    return weights, config.to_dict(), tokenizer


def upload_to_hub(name_or_path,model_type,username,path: str, name: str, token : str):
    import os
   
    from huggingface_hub import HfApi, ModelCard, logging

    repo_id = f"{name}"

    try:
        create_repo(
            repo_id   = repo_id,
            token     = token,
            repo_type = "model",
            exist_ok  = False,
            private   = None,
        ) 
    except:
        pass
    
    try:
        content = MODEL_CARD.format(
            username   = username,
            base_model = name_or_path,
            model_type = model_type,
            method     = "",
            extra      = "unsloth",
        )
        card = ModelCard(content)
        card.push_to_hub(repo_id, token = token)
    except:
        pass
    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True,token = token)
    api.upload_folder(
        folder_path=path,
        path_in_repo = ".",
        token=token,
        repo_id=repo_id,
        commit_message  = "(Trained with Unsloth)",
        repo_type="model"
    )
    


def make_shards(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def save_model(save_dir: str, weights, tokenizer, config):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights, max_file_size_gibibyte=5)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, shards_count)
        mx.save_safetensors(
            str(save_dir / shard_name), shard, metadata={"format": "mlx"}
        )
        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    tokenizer.save_pretrained(save_dir)
    with open(save_dir / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }
    with open(save_dir / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def load(path_or_hf_repo: str, tokenizer_config={}):
    # If the path exists, it will try to load model form it
    # otherwise download and cache from the hf_repo and cache
    
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
            )
        )

    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        quantization = config.get("quantization", None)

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    model_args = models.ModelArgs.from_dict(config)
    model = models.Model(model_args)
    if quantization is not None:
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in weights
        )
        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, **tokenizer_config
    )
    return model, tokenizer, config


def generate(
    prompt: mx.array, model: nn.Module, temp: float = 0.0
) -> Generator[mx.array, None, None]:
    """
    Generate text based on the given prompt and model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling. If temp is 0, use max sampling.

    Yields:
        mx.array: The generated text.
    """

    def sample(logits: mx.array) -> mx.array:
        return (
            mx.argmax(logits, axis=-1)
            if temp == 0
            else mx.random.categorical(logits * (1 / temp))
        )

    y = prompt
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y

