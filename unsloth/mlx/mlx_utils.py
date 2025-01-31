import gc
import glob
import shutil
import json
import logging
from pathlib import Path
from typing import Generator, Optional,Type, Callable, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from unsloth.models.loader_utils import get_model_name
from .models import llama as models
import transformers
from huggingface_hub import snapshot_download,create_repo
from unsloth.save import MODEL_CARD
from mlx.utils import tree_flatten, tree_unflatten
from .trainer.utils import  load_adapters

MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
}


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

    shards = make_shards(weights, max_file_size_gibibyte=1)
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

def _get_classes(config: dict):
    model_type = config["model_type"]    
    if model_type != "llama" and MODEL_REMAPPING.get(model_type,model_type) != "llama":
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)

    return models.Model, models.ModelArgs

def load(model_path: str, tokenizer_config={},    
         get_model_classes: Callable[[dict], Tuple[Type[nn.Module], Type]] = _get_classes,):
    
    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        quantization = config.get("quantization", None)

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        # Try weight for back-compat
        weight_files = glob.glob(str(model_path / "weight*.safetensors"))

    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    model_class, model_args_class = get_model_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
    
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

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

    # mx.eval(model.parameters())
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, **tokenizer_config
    )
    return model, tokenizer, config


def save_config(
    config: dict,
    config_path: Union[str, Path],
) -> None:
    """Save the model configuration to the ``config_path``.

    The final configuration will be sorted before saving for better readability.

    Args:
        config (dict): The model configuration.
        config_path (Union[str, Path]): Model configuration file path.
    """
    # Clean unused keys
    config.pop("_name_or_path", None)

    # sort the config for better readability
    config = dict(sorted(config.items()))

    # write the updated config to the config_path (if provided)
    with open(config_path, "w") as fid:
        json.dump(config, fid, indent=4)



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

def save_merged_model(args):
    model_name = get_model_name(args.model_name,args.load_in_4bit)
    model_path = get_model_path(model_name)
    model, tokenizer, config = load(model_path)
    model.freeze()

    # Load the LoRA adapter weights which we assume should exist by this point
    if not Path(args.save_path,args.adapter_file).is_file():
        raise ValueError(
        f"Adapter file {args.adapter_file} missing. ")
    
    model = load_adapters(model, args.save_path,args.adapter_file)

    fused_linears = [
        (n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")
    ]

    if fused_linears:
        model.update_modules(tree_unflatten(fused_linears))

    weights = dict(tree_flatten(model.parameters()))

    save_model(args.save_path, weights, tokenizer, config)
   
    mx.metal.clear_cache()
    del model
    gc.collect()


def push_to_hub(args,name, model_type):
        if args.push_model:
            from huggingface_hub import whoami
            try: 
                username = whoami(token = args.hub_token)["name"]
            except:
                raise RuntimeError(
                    "Unsloth: Please supply a token!\n"\
                    "Go to https://huggingface.co/settings/tokens"
                )
            pass
        pass

        if  args.push_model and args.hub_path is not None:
            hf_path = args.hub_path
            if not Path(args.model_name).exists():
                # If the model path doesn't exist, assume it's an HF repo
                hf_path = args.model_name
            elif hf_path is None:
                raise ValueError(
                    "Must provide original Hugging Face repo to upload local model."
                )
            upload_to_hub(name,model_type,username,args.save_path, args.hub_path,args.hub_token)


def get_model_path(path_or_hf_repo: str, revision: Optional[str] = None) -> Path:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        try:
            model_path = Path(
                snapshot_download(
                    repo_id=path_or_hf_repo,
                    revision=revision,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.py",
                        "tokenizer.model",
                        "*.tiktoken",
                        "*.txt",
                    ],
                )
            )
        except:
            raise FileNotFoundError(
                f"Model not found for path or HF repo: {path_or_hf_repo}.\n"
                "Please make sure you specified the local path or Hugging Face"
                " repo id correctly.\nIf you are trying to access a private or"
                " gated Hugging Face repo, make sure you are authenticated:\n"
                "https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login"
            ) from None
    return model_path



def load_pretrained(
    model_name: str,
    tokenizer_config={},
    model_config={},
    dtype= None,
    load_in_4bit=True
):
    model_name = get_model_name(model_name,load_in_4bit)
    model_path = get_model_path(model_name)

    model,tokenizer, config = load(model_path, tokenizer_config)

    return model, tokenizer, config