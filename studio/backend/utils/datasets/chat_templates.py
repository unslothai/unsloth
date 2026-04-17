# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Chat template application utilities for dataset processing.

This module contains functions for applying chat templates to datasets
and generating dataset info summaries.
"""

from .format_detection import detect_dataset_format, detect_multimodal_dataset, detect_custom_format_heuristic
from .model_mappings import MODEL_TO_TEMPLATE_MAPPER
from loggers import get_logger
logger = get_logger(__name__)




DEFAULT_ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def get_tokenizer_chat_template(tokenizer, model_name):
    """
    Gets appropriate chat template for tokenizer based on model.
    Uses Unsloth's get_chat_template if model is in the mapper.

    Args:
        tokenizer: HuggingFace tokenizer
        model_name: Model class name (e.g., "Gemma3ForCausalLM")

    Returns:
        tokenizer: Tokenizer with appropriate chat template applied
    """
    try:
        from unsloth.chat_templates import get_chat_template
    except ImportError:
        # Unsloth not available, return tokenizer as-is
        return tokenizer

    # Normalize model_name to lowercase for matching
    model_name_lower = model_name.lower()

    # Check if model matches any template in mapper
    matched_template = None

    # Direct match in MODEL_TO_TEMPLATE_MAPPER
    if model_name_lower in MODEL_TO_TEMPLATE_MAPPER:
        matched_template = MODEL_TO_TEMPLATE_MAPPER[model_name_lower]
        logger.info(f"📝 Applying Unsloth chat template: {matched_template}")
        try:
            tokenizer = get_chat_template(
                tokenizer,
                chat_template = matched_template,
            )
        except Exception as e:
            logger.info(f"⚠️ Failed to apply Unsloth template '{matched_template}': {e}")
            logger.info(f"   Falling back to tokenizer's default chat template")
    else:
        # Check if tokenizer actually has a chat_template set
        has_chat_template = (
            hasattr(tokenizer, 'chat_template')
            and tokenizer.chat_template is not None
        )
        if has_chat_template:
            logger.info(f"📝 Using tokenizer's own chat template (no Unsloth template match)")
        else:
            # Base model with no chat template — apply default ChatML
            logger.info(f"📝 No chat template found — applying default ChatML template (base model)")
            try:
                tokenizer = get_chat_template(
                    tokenizer,
                    chat_template = "chatml",
                )
            except Exception as e:
                logger.info(f"⚠️ Failed to apply default ChatML template: {e}")
                logger.info(f"   Falling back to tokenizer as-is")

    return tokenizer


def get_dataset_info_summary(dataset_info):
    """
    Returns a human-readable summary for UI display.
    """
    detected_format = dataset_info["detected_format"]
    final_format = dataset_info["final_format"]

    format_descriptions = {
        "alpaca": "Alpaca format (instruction/input/output)",
        "sharegpt": "ShareGPT format (needs standardization)",
        "chatml_messages": "ChatML format (messages column) - OpenAI compatible",
        "chatml_conversations": "ChatML format (conversations column) - HuggingFace standard",
        "unknown": "Unknown format"
    }

    return {
        "detected_format": detected_format,
        "final_format": final_format,
        "detected_description": format_descriptions.get(detected_format, "Unknown"),
        "final_description": format_descriptions.get(final_format, "Unknown"),
        "chat_column": dataset_info["chat_column"],
        "is_standardized": dataset_info["is_standardized"],
        "warnings": dataset_info.get("warnings", []),
        "ready_for_training": dataset_info["is_standardized"] and final_format != "unknown"
    }


def apply_chat_template_to_dataset(
    dataset_info,
    tokenizer,
    model_name = None,
    custom_prompt_template = None,
    add_eos_token = False,
    remove_bos_prefix = False,
    custom_format_mapping = None,
    auto_detect_mapping = True,
    batch_size = 1000,
    num_proc = None,
    progress_callback = None,
):
    """
    Applies chat template to dataset based on its format.

    Args:
        dataset_info: Output from format_dataset() with metadata
        tokenizer: Tokenizer with chat template
        custom_prompt_template: Optional string template for custom formatting
        add_eos_token: If True, appends tokenizer.eos_token to each text
        remove_bos_prefix: If True, removes '<bos>' prefix (for Gemma, etc.)
        custom_format_mapping: Dict mapping custom columns to standard format
        batch_size: Batch size for processing
        num_proc: Number of processes

    Returns:
        dict with dataset, success status, warnings, and errors
    """
    dataset = dataset_info["dataset"]
    final_format = dataset_info["final_format"]
    chat_column = dataset_info["chat_column"]
    is_standardized = dataset_info["is_standardized"]

    warnings = list(dataset_info.get("warnings", []))
    errors = []

    # Get EOS token if needed
    eos_token = ""
    if add_eos_token:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
            eos_token = tokenizer.eos_token
        else:
            warnings.append("add_eos_token=True but tokenizer has no eos_token")

    # CUSTOM FORMAT MAPPING (for non-standard datasets)
    if final_format == "unknown":
        # Try auto-detection if no custom mapping provided
        if custom_format_mapping is None and auto_detect_mapping:
            # Check if format_dataset already tried and failed
            if not dataset_info.get("auto_detection_attempted", False):
                custom_format_mapping = detect_custom_format_heuristic(dataset)
                if custom_format_mapping:
                    warnings.append(f"Auto-detected column mapping: {custom_format_mapping}")
                else:
                    errors.append("Could not auto-detect format mapping")
                    return {
                        "dataset": dataset,
                        "success": False,
                        "warnings": warnings,
                        "errors": errors
                    }
            else:
                # Already failed once in format_dataset, don't retry
                errors.append(
                    "Format remains unknown after detection attempts. "
                    "Please provide custom_format_mapping to specify column roles manually."
                )
                return {
                    "dataset": dataset,
                    "success": False,
                    "warnings": warnings,
                    "errors": errors
                }

        if custom_format_mapping:
            warnings.append(f"Applying custom format mapping: {custom_format_mapping}")
            is_user_provided = dataset_info.get("custom_format_mapping") is not None

            def _apply_custom_mapping(examples):
                conversations = []
                num_examples = len(examples[list(examples.keys())[0]])

                # Only preserve unmapped columns if auto-detected
                preserved_columns = {}
                if not is_user_provided:
                    all_columns = set(examples.keys())
                    mapped_columns = set(custom_format_mapping.keys())
                    non_mapped_columns = all_columns - mapped_columns

                    for col in non_mapped_columns:
                        preserved_columns[col] = examples[col]

                for i in range(num_examples):
                    convo = []
                    role_order = ['system', 'user', 'assistant']

                    for target_role in role_order:
                        for col_name, role in custom_format_mapping.items():
                            if role == target_role and col_name in examples:
                                content = examples[col_name][i]

                                if is_user_provided:
                                    # User explicitly mapped - include even if empty
                                    convo.append({"role": role, "content": str(content) if content else ""})
                                else:
                                    # Auto-detected - skip empty
                                    if content and str(content).strip():
                                        convo.append({"role": role, "content": str(content)})

                    conversations.append(convo)

                result = {"conversations": conversations}
                if not is_user_provided:
                    result.update(preserved_columns)
                return result

            try:
                dataset = dataset.map(_apply_custom_mapping, batched = True, batch_size = batch_size)
                # Update to use conversations format
                final_format = "chatml_conversations"
                chat_column = "conversations"
                is_standardized = True
                warnings.append("Successfully converted to ChatML format via custom mapping")
            except Exception as e:
                errors.append(f"Custom format mapping failed: {e}")
                return {
                    "dataset": dataset,
                    "success": False,
                    "warnings": warnings,
                    "errors": errors
                }

    # ALPACA FORMAT
    if final_format == "alpaca":

        # Set alpaca chat template on tokenizer for saving (if not already set)
        # This ensures the template is saved with the model for inference
        if not (hasattr(tokenizer, 'chat_template') and tokenizer.chat_template):
            try:
                from unsloth.chat_templates import get_chat_template
                tokenizer = get_chat_template(tokenizer, chat_template = "alpaca")
                logger.info(f"📝 Set alpaca chat template on tokenizer for model saving")
            except Exception as e:
                logger.info(f"⚠️ Could not set alpaca template on tokenizer: {e}")

        # Use custom template if provided
        def _format_alpaca_custom(examples):
            texts = []
            for i in range(len(examples["instruction"])):
                fields = {
                    "instruction": examples["instruction"][i],
                    "input": examples.get("input", [""] * len(examples["instruction"]))[i],
                    "output": examples["output"][i]
                }

                try:
                    text = DEFAULT_ALPACA_TEMPLATE.format(fields["instruction"], fields["input"], fields["output"])
                    text += eos_token
                    texts.append(text)
                except KeyError as e:
                    errors.append(f"Custom template missing field: {e}")
                    texts.append("")

            return {"text": texts}

        formatted_fn = _format_alpaca_custom

        try:
            dataset_map_kwargs = {
                'batched': True,
                'batch_size': batch_size,
            }

            try:
                from torch.utils.data import IterableDataset
                _is_torch_iterable = isinstance(dataset, IterableDataset)
            except ImportError:
                _is_torch_iterable = False

            if not _is_torch_iterable:
                from utils.hardware import dataset_map_num_proc
                if num_proc is None or type(num_proc) is not int:
                    num_proc = dataset_map_num_proc()
                else:
                    num_proc = dataset_map_num_proc(num_proc)
                dataset_map_kwargs['num_proc'] = num_proc
                dataset_map_kwargs['desc'] = "Applying template to Alpaca format"

            formatted_dataset = dataset.map(formatted_fn, **dataset_map_kwargs)

            return {
                "dataset": formatted_dataset,
                "success": True,
                "warnings": warnings,
                "errors": errors
            }
        except Exception as e:
            errors.append(f"Failed to format Alpaca dataset: {e}")
            return {
                "dataset": dataset,
                "success": False,
                "warnings": warnings,
                "errors": errors
            }

    # CHATML FORMATS
    elif final_format in ["chatml_messages", "chatml_conversations"]:

        if not is_standardized:
            warnings.append("Dataset may not be fully standardized")

        # Apply Unsloth chat template if model matches
        if model_name:
            tokenizer = get_tokenizer_chat_template(tokenizer, model_name)

        def _format_chatml(examples):
            convos = examples[chat_column]
            texts = []

            for convo in convos:
                try:
                    text = tokenizer.apply_chat_template(
                        convo,
                        tokenize = False,
                        add_generation_prompt = False
                    )

                    if remove_bos_prefix:
                        text = text.removeprefix('<bos>')
                    text += eos_token

                    texts.append(text)
                except Exception as e:
                    if len(texts) == 0:
                        warnings.append(f"Chat template failed: {e}")
                    texts.append("")

            return {"text": texts}

        try:
            try:
                from torch.utils.data import IterableDataset
                _is_torch_iterable = isinstance(dataset, IterableDataset)
            except ImportError:
                _is_torch_iterable = False

            dataset_map_kwargs = {
                'batched': True,
                'batch_size': batch_size,
            }

            if not _is_torch_iterable:
                from utils.hardware import dataset_map_num_proc
                if num_proc is None or type(num_proc) is not int:
                    num_proc = dataset_map_num_proc()
                else:
                    num_proc = dataset_map_num_proc(num_proc)
                dataset_map_kwargs['num_proc'] = num_proc
                dataset_map_kwargs['desc'] = f"Applying chat template to {final_format}"

            # Monitor tqdm progress from dataset.map() and relay to callback
            _tqdm_monitor_stop = None
            if progress_callback and not _is_torch_iterable:
                import threading
                from tqdm.auto import tqdm as _tqdm_cls

                _tqdm_monitor_stop = threading.Event()
                _total = len(dataset) if hasattr(dataset, "__len__") else 0
                _desc = f"Applying chat template to {final_format}"

                def _poll_tqdm():
                    while not _tqdm_monitor_stop.is_set():
                        for bar in list(getattr(_tqdm_cls, "_instances", set())):
                            try:
                                n = bar.n or 0
                                total = bar.total or _total
                                if total > 0 and n > 0:
                                    pct = min(int(n * 100 / total), 100)
                                    progress_callback(
                                        status_message = f"{_desc}... {pct}% ({n:,}/{total:,})"
                                    )
                            except (AttributeError, ReferenceError):
                                pass
                        _tqdm_monitor_stop.wait(3)

                threading.Thread(target = _poll_tqdm, daemon = True).start()

            formatted_dataset = dataset.map(_format_chatml, **dataset_map_kwargs)

            if _tqdm_monitor_stop is not None:
                _tqdm_monitor_stop.set()

            return {
                "dataset": formatted_dataset,
                "success": True,
                "warnings": warnings,
                "errors": errors
            }
        except Exception as e:
            errors.append(f"Failed to format ChatML dataset: {e}")
            return {
                "dataset": dataset,
                "success": False,
                "warnings": warnings,
                "errors": errors
            }

    # UNKNOWN FORMAT
    else:
        errors.append(
            f"Cannot apply chat template to format: {final_format}. "
            f"This should not happen after custom mapping."
        )
        return {
            "dataset": dataset,
            "success": False,
            "warnings": warnings,
            "errors": errors
        }
