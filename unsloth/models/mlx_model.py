import os
import json
from typing import Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass

from mlx_lm import load
from mlx_lm.tuner import (
    train,
    TrainingArgs,
    datasets,
    linear_to_lora_layers,
)
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from ..device_type import DEVICE_TYPE


@dataclass
class MLXTrainingArguments:
    """training arguments for MLX models."""

    adapter_file: str = "adapters.safetensors"
    max_seq_length: int = 2048
    grad_checkpoint: bool = True
    grad_accumulation_steps: int = 1
    iters: int = 100
    batch_size: int = 4
    val_batches: int = 10 

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_file": self.adapter_file,
            "max_seq_length": self.max_seq_length,
            "grad_checkpoint": self.grad_checkpoint,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "iters": self.iters,
            "batch_size": self.batch_size,
            "val_batches": self.val_batches,
        }


class MLXLoraConfig:

    def __init__(
        self,
        rank: int = 8,
        scale: float = 20.0,
        dropout: float = 0.0,
        num_layers: int = 8,
    ):
        self.rank = rank
        self.scale = scale
        self.dropout = dropout
        self.num_layers = num_layers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "lora_parameters": {
                "rank": self.rank,
                "scale": self.scale,
                "dropout": self.dropout,
            },
        }

    def save(self, adapter_path: str):
        os.makedirs(adapter_path, exist_ok=True)
        config_path = os.path.join(adapter_path, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)



class MLXTrainer:
    def prepare_model_for_training(
        self,
        model: Any,
        lora_config: Optional[MLXLoraConfig] = None,
    ) -> Any:
        if lora_config is None:
            lora_config = MLXLoraConfig()

        model.freeze()

        linear_to_lora_layers(
            model,
            lora_config.num_layers,
            lora_config.to_dict()["lora_parameters"],
        )

        num_train_params = sum(
            v.size for _, v in tree_flatten(model.trainable_parameters())
        )
        print(f"number of trainable parameters: {num_train_params}")

        model.train()

        return model

    def _train(
        self,
        model: Any,
        training_args: Union[MLXTrainingArguments, Dict[str, Any]],
        train_dataset: Any,
        val_dataset: Any = None,
        learning_rate: float = 1e-5,
    ):
        if isinstance(training_args, MLXTrainingArguments):
            args_dict = training_args.to_dict()
        else:
            args_dict = training_args

        args = TrainingArgs(**args_dict)

        optimizer = optim.Adam(learning_rate=learning_rate)

        train_set = datasets.CacheDataset(train_dataset)
        val_set = datasets.CacheDataset(val_dataset) if val_dataset else None

        train(
            model=model,
            args=args,
            optimizer=optimizer,
            train_dataset=train_set,
            val_dataset=val_set,
        )



class FastMLXModel:
    @staticmethod
    def from_pretrained(
        model_name: str,
        **kwargs,
    ) -> Tuple[Any, Any]:
        print(f"Unsloth: Loading model with MLX: {model_name}")

        model, tokenizer = load(model_name)
        return model, tokenizer

    @staticmethod
    def for_inference(
        model_name: str,
        adapter_path: Optional[str] = None,
    ) -> Any:
        if adapter_path:
            model, _ = load(model_name, adapter_path=adapter_path)
        else:
            model, _ = load(model_name)

        return model

    @staticmethod
    def train(
        model: Any,
        train_set: Any,
        val_set: Any,
        lora_config: Optional[MLXLoraConfig] = None,
        iterations: int = 100,
        learning_rate: float = 1e-5,
    ):
        if DEVICE_TYPE != "mps":
            raise RuntimeError("This function requires running on Apple Silicon")

        trainer = MLXTrainer()

        if lora_config is None:
            lora_config = MLXLoraConfig()

        trainer.prepare_model_for_training(model, lora_config)

        trainer._train(
            model=model,
            training_args=MLXTrainingArguments(iters=iterations),
            train_dataset=train_set,
            val_dataset=val_set,
            learning_rate=learning_rate,
        )

        return model
