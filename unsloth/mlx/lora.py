from pathlib import Path
import math
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.core as mx
from .trainer.trainer import TrainingArgs, TrainingCallback, train
from .trainer.utils import (
    build_schedule,
    linear_to_lora_layers,
    print_trainable_parameters,
)
from .mlx_utils import  save_config

def train_model(
    args,
    model: nn.Module,
    tokenizer,
    train_set,
    valid_set,
    training_callback: TrainingCallback = None,
):
    model.freeze()
    linear_to_lora_layers(
        model,
        min(args.r,len(model.layers)/2),
        {"rank": args.r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "scale": float(args.lora_alpha)/math.sqrt(float(args.r)) if args.use_rslora else float(args.lora_alpha)/float(args.r)},
    )
    print_trainable_parameters(model)

    adapter_path = Path(args.save_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    adapter_file = adapter_path / args.adapter_file
    config = {
        "num_layers" : min(args.r,len(model.layers)/2),
        "lora_parameters" : {"rank": args.r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "scale": float(args.lora_alpha)/math.sqrt(float(args.r)) if args.use_rslora else float(args.lora_alpha)/float(args.r)}
    }
    save_config(config, adapter_path / "adapter_config.json")

    # init training args
    training_args = TrainingArgs(
        batch_size=args.per_device_train_batch_size,
        iters=args.max_steps,
        val_batches=25,
        steps_per_report=args.logging_steps,
        steps_per_eval=200,
        steps_per_save=100,
        adapter_file=adapter_file,
        max_seq_length=args.max_seq_length,
        grad_checkpoint=args.use_gradient_checkpointing,
    )

    mx.random.seed(args.seed)
    model.train()
    if args.lr_scheduler_type == "linear":
        arguments = [args.learning_rate,0.0,args.max_steps-args.warmup_steps]
    elif args.lr_scheduler_type == "exponential_decay":
        arguments = [args.learning_rate,args.weight_decay]
    elif args.lr_scheduler_type == "step_decay":
        arguments = [args.learning_rate,args.weight_decay,args.warmup_steps]
    elif args.lr_scheduler_type == "cosine_decay":
        arguments = [args.learning_rate,args.max_steps]
    else:
        arguments = [args.learning_rate]

    schedule_config = {
            "name": "linear_schedule" if args.lr_scheduler_type == "linear" else args.lr_scheduler_type,
            "warmup": args.warmup_steps,
            "arguments": arguments,
        }
    
    lr = build_schedule(schedule_config) if args.lr_scheduler_type else args.learning_rate

    if args.optim.lower().startswith("sgd"):
        opt = optim.SGD(learning_rate=(lr), weight_decay=args.weight_decay)
    elif args.optim.lower().startswith("rmsprop"):
        opt = optim.RMSprop(learning_rate=(lr))
    elif args.optim.lower().startswith("adagrad"):
        opt = optim.Adagrad(learning_rate=(lr))
    elif args.optim.lower().startswith("adaDelta"):
        opt = optim.AdaDelta(learning_rate=(lr))
    elif args.optim.lower().startswith("adamw"):
        opt = optim.AdamW(learning_rate=(lr),weight_decay=args.weight_decay)
    elif args.optim.lower().startswith("adam"):
        opt = optim.Adam(learning_rate=(lr))
    elif args.optim.lower().startswith("adamax"):
        opt = optim.Adamax(learning_rate=(lr))
    elif args.optim.lower().startswith("lion"):
        opt = optim.Lion(learning_rate=(lr), weight_decay=args.weight_decay)
    elif args.optim.lower().startswith("adafactor"):
        opt = optim.Adafactor(learning_rate=(lr), weight_decay= args.weight_decay)
    else:
        raise ValueError("The Optimizer type provided is not supported")
        
    # Train model
    train(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizer=opt,
        train_dataset=train_set,
        val_dataset=valid_set,
        training_callback=training_callback,
    )

