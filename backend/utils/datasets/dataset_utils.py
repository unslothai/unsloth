import torch
from torch.utils.data import IterableDataset
try:
    from deepseek_ocr.modeling_deepseekocr import (
        format_messages,
        text_encode,
        BasicImageTransform,
        dynamic_preprocess,
    )
    DEEPSEEK_OCR_AVAILABLE = True
except ImportError:
    DEEPSEEK_OCR_AVAILABLE = False
    format_messages = None
    text_encode = None
    BasicImageTransform = None
    dynamic_preprocess = None
    import logging
    logging.getLogger(__name__).warning(
        "DeepSeek OCR module not found. Will auto-install if needed."
    )
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
import io




DEFAULT_ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""



def standardize_chat_format(
    dataset,
    tokenizer=None,
    aliases_for_system=["system",],
    aliases_for_user=["user", "human", "input",],
    aliases_for_assistant=["gpt", "assistant", "output",],
    batch_size=1000,
    num_proc=None,
):
    """
    Our own standardization function that handles BOTH messages and conversations.
    Converts non-standard role names and keys to standard format.
    """
    import collections
    import itertools
    from datasets import IterableDataset

    # Check if vision tokenizer is used
    is_vlm = False
    if tokenizer is not None:
        if hasattr(tokenizer, "image_processor") or hasattr(tokenizer, "tokenizer"):
            is_vlm = True

    column_names = set(next(iter(dataset)).keys())

    #   Check for both 'conversations' and 'messages'
    chat_column = None
    if "conversations" in column_names:
        chat_column = "conversations"
    elif "messages" in column_names:
        chat_column = "messages"
    elif "texts" in column_names:
        chat_column = "texts"
    else:
        return dataset  # No chat column found

    # Inspect structure
    examples = itertools.islice(dataset, 10)
    uniques = collections.defaultdict(list)
    for example in examples:
        for message in example[chat_column]:
            for key, value in message.items():
                if type(value) is not str:
                    continue  # Skip non-string values
                uniques[key].append(value)

    if len(uniques.keys()) != 2:
        return dataset  # Unexpected structure

    keys = list(uniques.keys())
    length_first  = len(set(uniques[keys[0]]))
    length_second = len(set(uniques[keys[1]]))

    # Determine which is role and which is content
    if length_first < length_second:
        role_key    = keys[0]
        content_key = keys[1]
    else:
        role_key    = keys[1]
        content_key = keys[0]

    # Mapping for aliases
    aliases_mapping = {}
    for x in aliases_for_system:    aliases_mapping[x] = "system"
    for x in aliases_for_user:      aliases_mapping[x] = "user"
    for x in aliases_for_assistant: aliases_mapping[x] = "assistant"

    def _standardize_dataset(examples):
        convos = examples[chat_column]
        all_convos = []
        for convo in convos:
            new_convo = []
            for message in convo:
                # Get original role and content
                original_role = message.get(role_key, "")
                original_content = message.get(content_key, "")

                # Map to standard role name
                standard_role = aliases_mapping.get(original_role, original_role)

                # Handle VLM format
                if is_vlm:
                    original_content = [{"type": "text", "text": original_content}]

                # Create dict with EXPLICIT ORDER
                new_message = {"role": standard_role, "content": original_content}
                new_convo.append(new_message)

            all_convos.append(new_convo)

        return {chat_column: all_convos}


    dataset_map_kwargs = {
        'batched': True,
        'batch_size': batch_size,
    }

    if not isinstance(dataset, IterableDataset):
        from multiprocessing import cpu_count

        if num_proc is None or type(num_proc) is not int:
            num_proc = cpu_count()

        dataset_map_kwargs['num_proc'] = num_proc
        dataset_map_kwargs['desc'] = "Standardizing chat format"

    return dataset.map(_standardize_dataset, **dataset_map_kwargs)
pass


def detect_dataset_format(dataset):
    """
    Detects dataset format by inspecting structure.

    Returns:
        dict: {
            "format": "alpaca" | "sharegpt" | "chatml" | "unknown",
            "chat_column": "messages" | "conversations" | None,
            "needs_standardization": bool,
            "sample_keys": list of keys found in messages (for debugging)
        }
    """
    column_names = set(next(iter(dataset)).keys())

    # Check for Alpaca
    alpaca_columns = {"instruction", "output"}
    if alpaca_columns.issubset(column_names):
        return {
            "format": "alpaca",
            "chat_column": None,
            "needs_standardization": False,
            "sample_keys": []
        }

    # Check for chat-based formats (messages or conversations)
    chat_column = None
    if "messages" in column_names:
        chat_column = "messages"
    elif "conversations" in column_names:
        chat_column = "conversations"
    elif "texts" in column_names:
        chat_column = "texts"

    if chat_column:
        # Inspect the structure to determine if ShareGPT or ChatML
        try:
            sample = next(iter(dataset))
            chat_data = sample[chat_column]

            if chat_data and len(chat_data) > 0:
                first_msg = chat_data[0]
                msg_keys = set(first_msg.keys())

                # ShareGPT uses "from" and "value"
                if "from" in msg_keys or "value" in msg_keys:
                    return {
                        "format": "sharegpt",
                        "chat_column": chat_column,
                        "needs_standardization": True,
                        "sample_keys": list(msg_keys)
                    }

                # ChatML uses "role" and "content"
                elif "role" in msg_keys and "content" in msg_keys:
                    return {
                        "format": "chatml",
                        "chat_column": chat_column,
                        "needs_standardization": False,
                        "sample_keys": list(msg_keys)
                    }

                # Unknown structure but has chat column
                else:
                    return {
                        "format": "unknown",
                        "chat_column": chat_column,
                        "needs_standardization": None,
                        "sample_keys": list(msg_keys)
                    }
        except Exception as e:
            return {
                "format": "unknown",
                "chat_column": chat_column,
                "needs_standardization": None,
                "sample_keys": [],
                "error": str(e)
            }

    # No recognized format
    return {
        "format": "unknown",
        "chat_column": None,
        "needs_standardization": None,
        "sample_keys": []
    }


def format_dataset(
    dataset,
    format_type           = "auto",
    tokenizer             = None,
    aliases_for_system    = ["system",],
    aliases_for_user      = ["user", "human", "input",],
    aliases_for_assistant = ["gpt", "assistant", "output",],
    batch_size            = 1000,
    num_proc              = None,
    auto_detect_custom    = True,
):
    """
    Formats dataset and returns metadata.

    Returns:
        dict: {
            "dataset": processed dataset,
            "detected_format": original format detected,
            "final_format": final format after processing,
            "chat_column": column name with chat data,
            "is_standardized": whether role names are standardized,
            "warnings": list of warning messages
        }
    """

    # Detect multimodal first
    multimodal_info = detect_multimodal_dataset(dataset)

    # Detect current format
    detected = detect_dataset_format(dataset)
    warnings = []

     # Add multimodal warning if detected
    if multimodal_info["is_multimodal"]:
        warnings.append(
            f"Multimodal dataset detected. Found columns: {multimodal_info['multimodal_columns']}"
        )

    # AUTO MODE: Keep format but standardize if needed
    if format_type == "auto":

        # Alpaca - keep as is
        if detected["format"] == "alpaca":
            return {
                "dataset": dataset,
                "detected_format": "alpaca",
                "final_format": "alpaca",
                "chat_column": None,
                "is_standardized": True,
                "is_multimodal": multimodal_info["is_multimodal"],
                "multimodal_info": multimodal_info,
                "warnings": []
            }

        # ShareGPT - needs standardization
        elif detected["format"] == "sharegpt":
            try:
                standardized = standardize_chat_format(
                    dataset, tokenizer, aliases_for_system,
                    aliases_for_user, aliases_for_assistant,
                    batch_size, num_proc
                )
                return {
                    "dataset": standardized,
                    "detected_format": "sharegpt",
                    "final_format": f"chatml_{detected['chat_column']}",
                    "chat_column": detected["chat_column"],
                    "is_standardized": True,
                    "is_multimodal": multimodal_info["is_multimodal"],
                    "multimodal_info": multimodal_info,
                    "warnings": []
                }
            except Exception as e:
                warnings.append(f"Failed to standardize ShareGPT format: {e}")
                return {
                    "dataset": dataset,
                    "detected_format": "sharegpt",
                    "final_format": "sharegpt",
                    "chat_column": detected["chat_column"],
                    "is_standardized": False,
                    "is_multimodal": multimodal_info["is_multimodal"],
                    "multimodal_info": multimodal_info,
                    "warnings": warnings
                }

        elif detected["format"] == "chatml" and detected["chat_column"] in ["conversations", "messages", "texts"]:
            return {
                "dataset": dataset,
                "detected_format": f"chatml_{detected['chat_column']}",
                "final_format": f"chatml_{detected['chat_column']}",
                "chat_column": detected["chat_column"],
                "is_standardized": True,
                "is_multimodal": multimodal_info["is_multimodal"],
                "multimodal_info": multimodal_info,
                "warnings": warnings
            }


        # Unknown - try standardization, if fails pass as is
        else:
            warnings.append(f"Unknown format detected. Keys found: {detected['sample_keys']}")

            # NEW: Try heuristic detection
            if auto_detect_custom:
                custom_mapping = detect_custom_format_heuristic(dataset)
                if custom_mapping:
                    warnings.append(f"Auto-detected column mapping: {custom_mapping}")


                    def _apply_auto_mapping(examples):
                        conversations = []
                        num_examples = len(examples[list(examples.keys())[0]])

                        # NEW: Check if this is user-provided or auto-detected
                        is_user_provided = custom_format_mapping is not None  # Passed explicitly

                        # Preserve non-mapped columns ONLY if auto-detected
                        preserved_columns = {}
                        if not is_user_provided:  # Only preserve for auto-detection
                            all_columns = set(examples.keys())
                            mapped_columns = set(custom_mapping.keys())
                            non_mapped_columns = all_columns - mapped_columns

                            for col in non_mapped_columns:
                                preserved_columns[col] = examples[col]

                        for i in range(num_examples):
                            convo = []

                            # Enforce standard role order
                            role_order = ['system', 'user', 'assistant']

                            for target_role in role_order:
                                for col_name, role in custom_mapping.items():
                                    if role == target_role and col_name in examples:
                                        content = examples[col_name][i]

                                        # NEW: Different behavior based on mapping source
                                        if is_user_provided:
                                            # User explicitly mapped this - always include even if empty
                                            convo.append({"role": role, "content": str(content) if content else ""})
                                        else:
                                            # Auto-detected - skip empty (original behavior)
                                            if content and str(content).strip():
                                                convo.append({"role": role, "content": str(content)})

                            conversations.append(convo)

                        result = {"conversations": conversations}

                        # Only add preserved columns if auto-detected
                        if not is_user_provided:
                            result.update(preserved_columns)

                        return result


                    try:
                        dataset = dataset.map(_apply_auto_mapping, batched=True, batch_size=batch_size)
                        return {
                            "dataset": dataset,
                            "detected_format": "unknown",
                            "final_format": "chatml_conversations",
                            "chat_column": "conversations",
                            "is_standardized": True,
                            "is_multimodal": multimodal_info["is_multimodal"],
                            "multimodal_info": multimodal_info,
                            "warnings": warnings
                        }
                    except Exception as e:
                        warnings.append(f"Auto-detection failed: {e}")

            # Try standardization as a last resort
            if detected["chat_column"]:
                try:
                    standardized = standardize_chat_format(
                        dataset, tokenizer, aliases_for_system,
                        aliases_for_user, aliases_for_assistant,
                        batch_size, num_proc
                    )
                    warnings.append("Successfully standardized unknown format")
                    return {
                        "dataset": standardized,
                        "detected_format": "unknown",
                        "final_format": f"chatml_{detected['chat_column']}",
                        "chat_column": detected["chat_column"],
                        "is_standardized": True,
                        "is_multimodal": multimodal_info["is_multimodal"],
                        "multimodal_info": multimodal_info,
                        "warnings": warnings
                    }
                except Exception as e:
                    warnings.append(f"Could not standardize: {e}. Passing dataset as-is.")

            # Return as-is with warnings
            return {
                "dataset": dataset,
                "detected_format": "unknown",
                "final_format": "unknown",
                "chat_column": detected["chat_column"],
                "is_standardized": False,
                "is_multimodal": multimodal_info["is_multimodal"],
                "multimodal_info": multimodal_info,
                "warnings": warnings
            }

    # ALPACA MODE: Convert to Alpaca
    elif format_type == "alpaca":

        if detected["format"] == "alpaca":
            return {
                "dataset": dataset,
                "detected_format": "alpaca",
                "final_format": "alpaca",
                "chat_column": None,
                "is_standardized": True,
                "is_multimodal": multimodal_info["is_multimodal"],
                "multimodal_info": multimodal_info,
                "warnings": []
            }

        elif detected["format"] in ["sharegpt", "chatml"]:
            # First standardize if ShareGPT
            if detected["format"] == "sharegpt":
                dataset = standardize_chat_format(
                    dataset, tokenizer, aliases_for_system,
                    aliases_for_user, aliases_for_assistant,
                    batch_size, num_proc
                )

            # Then convert to Alpaca
            converted = convert_chatml_to_alpaca(dataset, batch_size, num_proc)
            return {
                "dataset": converted,
                "detected_format": detected["format"],
                "final_format": "alpaca",
                "chat_column": None,
                "is_standardized": True,
                "is_multimodal": multimodal_info["is_multimodal"],
                "multimodal_info": multimodal_info,
                "warnings": []
            }

        else:
            warnings.append(f"Cannot convert unknown format to Alpaca")
            return {
                "dataset": dataset,
                "detected_format": "unknown",
                "final_format": "unknown",
                "chat_column": detected["chat_column"],
                "is_standardized": False,
                "is_multimodal": multimodal_info["is_multimodal"],
                "multimodal_info": multimodal_info,
                "warnings": warnings
            }

    # CHATML MODE: Convert to ChatML
    elif format_type in ["chatml", "conversational"]:

        if detected["format"] == "alpaca":
            converted = convert_alpaca_to_chatml(dataset, batch_size, num_proc)
            return {
                "dataset": converted,
                "detected_format": "alpaca",
                "final_format": "chatml_conversations",
                "chat_column": "conversations",
                "is_standardized": True,
                "is_multimodal": multimodal_info["is_multimodal"],
                "multimodal_info": multimodal_info,
                "warnings": []
            }

        elif detected["format"] == "sharegpt":
            standardized = standardize_chat_format(
                dataset, tokenizer, aliases_for_system,
                aliases_for_user, aliases_for_assistant,
                batch_size, num_proc
            )
            return {
                "dataset": standardized,
                "detected_format": "sharegpt",
                "final_format": f"chatml_{detected['chat_column']}",
                "chat_column": detected["chat_column"],
                "is_standardized": True,
                "is_multimodal": multimodal_info["is_multimodal"],
                "multimodal_info": multimodal_info,
                "warnings": []
            }

        elif detected["format"] == "chatml":
            return {
                "dataset": dataset,
                "detected_format": f"chatml_{detected['chat_column']}",
                "final_format": f"chatml_{detected['chat_column']}",
                "chat_column": detected["chat_column"],
                "is_standardized": True,
                "is_multimodal": multimodal_info["is_multimodal"],
                "multimodal_info": multimodal_info,
                "warnings": []
            }

        else:
            warnings.append(f"Unknown format, attempting standardization")
            try:
                standardized = standardize_chat_format(
                    dataset, tokenizer, aliases_for_system,
                    aliases_for_user, aliases_for_assistant,
                    batch_size, num_proc
                )
                return {
                    "dataset": standardized,
                    "detected_format": "unknown",
                    "final_format": f"chatml_{detected['chat_column']}",
                    "chat_column": detected["chat_column"],
                    "is_standardized": True,
                    "is_multimodal": multimodal_info["is_multimodal"],
                    "multimodal_info": multimodal_info,
                    "warnings": warnings
                }
            except Exception as e:
                warnings.append(f"Standardization failed: {e}")
                return {
                    "dataset": dataset,
                    "detected_format": "unknown",
                    "final_format": "unknown",
                    "chat_column": detected["chat_column"],
                    "is_standardized": False,
                    "is_multimodal": multimodal_info["is_multimodal"],
                    "multimodal_info": multimodal_info,
                    "warnings": warnings
                }

    else:
        raise ValueError(f"Unknown format_type: {format_type}")
pass


def convert_chatml_to_alpaca(dataset, batch_size=1000, num_proc=None):
    """
    Converts ChatML format (messages OR conversations) to Alpaca format.
    Handles both standardized and ShareGPT formats.

    Supports:
    - "messages" or "conversations" column
    - "role"/"content" (standard) or "from"/"value" (ShareGPT)
    """
    def _convert(examples):
        # Auto-detect which column name is used
        chatml_data = examples.get("messages") or examples.get("conversations") or examples.get("texts")

        if chatml_data is None:
            raise ValueError("No 'messages' or 'conversations' or 'texts' column found.")

        instructions = []
        outputs = []
        inputs = []

        for convo in chatml_data:
            instruction = ""
            output = ""

            for msg in convo:
                # Handle both standard and ShareGPT formats
                role = msg.get("role") or msg.get("from")
                content = msg.get("content") or msg.get("value")

                # Get first user message as instruction
                if role in ["user", "human", "input"] and not instruction:
                    instruction = content
                # Get first assistant message as output
                elif role in ["assistant", "gpt", "output"] and not output:
                    output = content
                    break  # Stop after first assistant response

            instructions.append(instruction)
            inputs.append("")  # Alpaca typically has empty input
            outputs.append(output)

        return {
            "instruction": instructions,
            "input": inputs,
            "output": outputs
        }

    dataset_map_kwargs = {
        'batched': True,
        'batch_size': batch_size,
    }

    if not isinstance(dataset, IterableDataset):
        from multiprocessing import cpu_count

        if num_proc is None or type(num_proc) is not int:
            num_proc = cpu_count()

        dataset_map_kwargs['num_proc'] = num_proc
        dataset_map_kwargs['desc'] = "Converting ChatML to Alpaca format"

    return dataset.map(_convert, **dataset_map_kwargs)


def convert_alpaca_to_chatml(dataset, batch_size=1000, num_proc=None):
    """
    Converts Alpaca format to ChatML format.

    Output format: Uses 'conversations' column with standard 'role'/'content' structure.
    """
    def _convert(examples):
        conversations = []

        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples.get("input", [""] * len(examples["instruction"]))[i]
            output = examples["output"][i]

            # Combine instruction and input (if exists) for user message
            if input_text and input_text.strip():
                user_content = f"{instruction}\n\n{input_text}".strip()
            else:
                user_content = instruction

            # Build conversation in standard ChatML format
            convo = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]
            conversations.append(convo)

        return {"conversations": conversations}

    dataset_map_kwargs = {
        'batched': True,
        'batch_size': batch_size,
    }

    if not isinstance(dataset, IterableDataset):
        from multiprocessing import cpu_count

        if num_proc is None or type(num_proc) is not int:
            num_proc = cpu_count()

        dataset_map_kwargs['num_proc'] = num_proc
        dataset_map_kwargs['desc'] = "Converting Alpaca to ChatML format"

    return dataset.map(_convert, **dataset_map_kwargs)


def apply_chat_template_to_dataset(
    dataset_info,
    tokenizer,
    model_name = None,
    custom_prompt_template=None,
    add_eos_token=False,
    remove_bos_prefix=False,
    custom_format_mapping=None,
    auto_detect_mapping=True,
    batch_size=1000,
    num_proc=None,
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
        # NEW: Try auto-detection if no custom mapping provided
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
            dataset = dataset.map(_apply_custom_mapping, batched=True, batch_size=batch_size)
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
                    text = DEFAULT_ALPACA_TEMPLATE.format(fields["instruction"],fields["input"],fields["output"])
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

            if not isinstance(dataset, IterableDataset):
                from multiprocessing import cpu_count
                if num_proc is None or type(num_proc) is not int:
                    num_proc = cpu_count()
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
                        tokenize=False,
                        add_generation_prompt=False
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
            dataset_map_kwargs = {
                'batched': True,
                'batch_size': batch_size,
            }

            if not isinstance(dataset, IterableDataset):
                from multiprocessing import cpu_count
                if num_proc is None or type(num_proc) is not int:
                    num_proc = cpu_count()
                dataset_map_kwargs['num_proc'] = num_proc
                dataset_map_kwargs['desc'] = f"Applying chat template to {final_format}"

            formatted_dataset = dataset.map(_format_chatml, **dataset_map_kwargs)

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


def format_and_template_dataset(
    dataset,
    model_name,
    tokenizer,
    is_vlm = False,
    format_type="auto",
    # VLM-specific parameters
    vlm_instruction=None,  # Now optional - will auto-generate
    vlm_text_column=None,
    vlm_image_column=None,
    dataset_name=None,

    custom_prompt_template=None,
    add_eos_token=False,
    remove_bos_prefix=False,
    custom_format_mapping=None,
    auto_detect_custom=True,      # NEW
    auto_detect_mapping=True,     # NEW
    aliases_for_system=["system",],
    aliases_for_user=["user", "human", "input",],
    aliases_for_assistant=["gpt", "assistant", "output",],
    batch_size=1000,
    num_proc=None,
):
    """
    Convenience function that combines format_dataset and apply_chat_template_to_dataset.
    Perfect for UI workflows - one function does everything!

    Returns:
        dict: {
            "dataset": Final dataset with 'text' column,
            "detected_format": Original format,
            "final_format": Format after processing,
            "success": Whether template application succeeded,
            "warnings": List of warnings,
            "errors": List of errors,
            "summary": Human-readable summary
        }
    """

    # VLM FLOW
    if is_vlm:
        warnings = []
        errors = []

        multimodal_info = detect_multimodal_dataset(dataset)
        vlm_structure = detect_vlm_dataset_structure(dataset)

        # Handle Llava format
        if vlm_structure["format"] == "vlm_messages_llava":
            try:
                dataset = convert_llava_to_vlm_format(dataset)
                warnings.append("Converted from Llava format (image indices) to standard VLM format")
            except Exception as e:
                errors.append(f"Failed to convert Llava format: {e}")
                import traceback
                traceback.print_exc()

                return {
                    "dataset": dataset,
                    "detected_format": "vlm_messages_llava",
                    "final_format": "vlm_conversion_failed",
                    "is_vlm": True,
                    "success": False,
                    "warnings": warnings,
                    "errors": errors,
                }

        # Handle simple format
        elif vlm_structure["needs_conversion"]:
            # ... existing simple conversion code
            if vlm_text_column is None:
                vlm_text_column = vlm_structure["text_column"]
            if vlm_image_column is None:
                vlm_image_column = vlm_structure["image_column"]

            if vlm_text_column is None or vlm_image_column is None:
                errors.append(
                    f"Could not auto-detect image/text columns. Found: {vlm_structure}. "
                )
                return {
                    "dataset": dataset,
                    "detected_format": "vlm_unknown",
                    "final_format": "vlm_unknown",
                    "is_vlm": True,
                    "success": False,
                    "warnings": warnings,
                    "errors": errors,
                }

            try:
                dataset = convert_to_vlm_format(
                    dataset,
                    instruction=vlm_instruction,
                    text_column=vlm_text_column,
                    image_column=vlm_image_column,
                    dataset_name=dataset_name,
                )

                if vlm_instruction:
                    warnings.append(f"Using user-provided instruction: '{vlm_instruction}'")
                else:
                    warnings.append("Auto-generated instruction based on dataset analysis")

            except Exception as e:
                errors.append(f"Failed to convert to VLM format: {e}")
                import traceback
                traceback.print_exc()

                return {
                    "dataset": dataset,
                    "detected_format": vlm_structure["format"],
                    "final_format": "vlm_conversion_failed",
                    "is_vlm": True,
                    "success": False,
                    "warnings": warnings,
                    "errors": errors,
                }

        # Already in standard VLM format
        elif vlm_structure["format"] == "vlm_messages":
            dataset = [sample for sample in dataset]
            warnings.append("Dataset already in standard VLM messages format")

        # Return as list
        return {
            "dataset": dataset,
            "detected_format": vlm_structure["format"],
            "final_format": "vlm_messages",
            "chat_column": "messages",
            "is_vlm": True,
            "is_multimodal": multimodal_info["is_multimodal"],
            "multimodal_info": multimodal_info,
            "vlm_structure": vlm_structure,
            "success": True,
            "warnings": warnings,
            "errors": errors,
        }

    # LLM FLOW (Existing code)
    else:
        # Step 1: Format the dataset
        dataset_info = format_dataset(
            dataset,
            format_type=format_type,
            tokenizer=tokenizer,
            auto_detect_custom = auto_detect_custom,
            aliases_for_system=aliases_for_system,
            aliases_for_user=aliases_for_user,
            aliases_for_assistant=aliases_for_assistant,
            batch_size=batch_size,
            num_proc=num_proc,
        )

        # Step 2: Apply chat template
        if "gemma" in model_name.lower() and not dataset_info["is_multimodal"] and (format_type != "alpaca" or (format_type == "auto" and dataset_info["detected_format"] != "alpaca")):
            print("remove_bos_prefix is true")
            remove_bos_prefix = True
        template_result = apply_chat_template_to_dataset(
            dataset_info=dataset_info,
            tokenizer=tokenizer,
            model_name = model_name,
            custom_prompt_template=custom_prompt_template,
            add_eos_token=add_eos_token,
            remove_bos_prefix=remove_bos_prefix,
            custom_format_mapping=custom_format_mapping,
            auto_detect_mapping = auto_detect_mapping,
            batch_size=batch_size,
            num_proc=num_proc,
        )

        # Step 3: Generate summary
        summary = get_dataset_info_summary(dataset_info)

        # Combine results
        all_warnings = dataset_info.get("warnings", []) + template_result.get("warnings", [])
        all_errors = template_result.get("errors", [])

        return {
            "dataset": template_result["dataset"],
            "detected_format": dataset_info["detected_format"],
            "final_format": dataset_info["final_format"],
            "chat_column": dataset_info.get("chat_column"),
            "is_vlm": False,  # This is LLM flow
            "success": template_result["success"],
            "warnings": all_warnings,
            "errors": all_errors,
            "summary": summary,
        }


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



def detect_custom_format_heuristic(dataset):
    """
    Smart detection with priority scoring.

    Strategy for ambiguous keywords like 'task':
    1. Detect assistant first (unambiguous)
    2. Detect user using high-priority keywords first
    3. Check REMAINING columns for system keywords (including 'task')
    4. Only if no system match, use 'task' as fallback user
    """
    sample = next(iter(dataset))
    all_columns = list(sample.keys())

    mapping = {}

    # Keywords
    assistant_words = [
        'output', 'answer', 'response', 'assistant', 'completion',
        'expected', 'recommendation', 'reply', 'result', 'target',
        'solution', 'explanation', 'solve'
    ]

    # Split into high/low priority
    user_words_high_priority = [
        'input', 'question', 'query', 'prompt', 'instruction',
        'request', 'snippet', 'user', 'text',
        'problem', 'exercise'
    ]
    user_words_low_priority = ['task']  # Ambiguous - can be user OR system
    user_words = user_words_high_priority + user_words_low_priority

    system_words = [
        'system', 'context', 'description', 'persona', 'role',
        'template', 'task'  # Also in system
    ]

    # Metadata columns to ignore
    metadata_exact_match = {
        'id', 'idx', 'index', 'key', 'timestamp', 'date',
        'metadata', 'source', 'kind', 'type', 'category',
        'score', 'label', 'tag', 'inference_mode'
    }

    metadata_prefix_patterns = [
        'problem_type', 'problem_source',
        'generation_model', 'pass_rate',
    ]

    priority_patterns = {
        'generated': 100,
        'gen_': 90,
        'model_': 80,
        'predicted': 70,
        'completion': 60,
    }

    def has_keyword(col_name, keywords):
        """Check if any keyword appears in column name."""
        col_lower = col_name.lower()
        col_normalized = col_lower.replace('_', '').replace('-', '').replace(' ', '')

        for keyword in keywords:
            if keyword in col_lower or keyword in col_normalized:
                return True
        return False

    def is_metadata(col_name):
        """Check if column is likely metadata."""
        col_lower = col_name.lower()

        if col_lower in metadata_exact_match:
            return True

        if col_lower in metadata_prefix_patterns:
            return True

        for pattern in metadata_prefix_patterns:
            if col_lower.startswith(pattern.split('_')[0] + '_') and col_lower != pattern:
                if '_' in col_lower:
                    prefix = col_lower.split('_')[0]
                    if prefix in ['generation', 'pass', 'inference']:
                        return True

        if len(col_lower) <= 2 and not col_lower in ['qa', 'q', 'a']:
            return True

        return False

    def get_priority_score(col_name):
        """Calculate priority score based on column name patterns."""
        col_lower = col_name.lower()
        score = 0

        for pattern, pattern_score in priority_patterns.items():
            if pattern in col_lower:
                score += pattern_score

        return score

    def get_content_length(col_name):
        """Get average content length for this column."""
        try:
            if col_name in sample and sample[col_name]:
                content = str(sample[col_name])
                return len(content)
            return 0
        except:
            return 0

    def score_column(col_name, keywords, role_type, num_candidates):
        """Score a column for how likely it is to be a particular role."""
        if not has_keyword(col_name, keywords):
            return 0

        score = 0
        score += 10

        #   NEW: Penalize ambiguous keywords when scoring for user
        if role_type == 'user':
            col_lower = col_name.lower()
            # If column is ONLY "task" (or task_xxx), give it lower priority for user role
            if 'task' in col_lower and not any(kw in col_lower for kw in user_words_high_priority):
                score -= 15  # Significant penalty so other user columns win

        priority_bonus = get_priority_score(col_name)
        score += priority_bonus

        if role_type in ['assistant', 'user']:
            avg_length = get_content_length(col_name)

            if num_candidates > 1:
                if avg_length > 1000:
                    score += 50
                elif avg_length > 200:
                    score += 30
                elif avg_length > 50:
                    score += 10
                elif avg_length < 50:
                    score -= 20
            else:
                if avg_length > 1000:
                    score += 50
                elif avg_length > 200:
                    score += 30
                elif avg_length > 50:
                    score += 10

        return score

    # Filter out metadata columns
    content_columns = [col for col in all_columns if not is_metadata(col)]

    # Count candidates first
    assistant_potential = [col for col in content_columns if has_keyword(col, assistant_words)]
    user_potential = [col for col in content_columns if has_keyword(col, user_words)]

    # STEP 1: Find best ASSISTANT column
    assistant_candidates = []
    for col in assistant_potential:
        score = score_column(col, assistant_words, 'assistant', len(assistant_potential))
        if score > 0:
            assistant_candidates.append((col, score))

    if assistant_candidates:
        assistant_candidates.sort(key=lambda x: x[1], reverse=True)
        assistant_col = assistant_candidates[0][0]
        mapping[assistant_col] = 'assistant'
    else:
        assistant_col = None

    # STEP 2: Find best USER column (with penalty for ambiguous keywords)
    user_candidates = []
    for col in user_potential:
        if col == assistant_col:
            continue
        score = score_column(col, user_words, 'user', len(user_potential))
        if score > 0:
            user_candidates.append((col, score))

    if user_candidates:
        user_candidates.sort(key=lambda x: x[1], reverse=True)
        user_col = user_candidates[0][0]
        mapping[user_col] = 'user'
    else:
        user_col = None

    # STEP 3: Check ALL remaining columns for SYSTEM matches (priority check)
    remaining_columns = [col for col in content_columns if col not in mapping]

    system_col = None
    for col in remaining_columns:
        if has_keyword(col, system_words):
            # Found a system match in remaining columns
            mapping[col] = 'system'
            system_col = col
            break

    # STEP 4: Handle any additional remaining columns
    if system_col:
        remaining_columns = [col for col in remaining_columns if col != system_col]

    if len(remaining_columns) >= 1:
        remaining_col = remaining_columns[0]

        # If no strong keyword match, decide based on what's missing
        if not has_keyword(remaining_col, user_words + assistant_words):
            mapping[remaining_col] = 'system'
        elif user_col is None:
            # No user column yet, assign this as user
            mapping[remaining_col] = 'user'
        else:
            # Already have user + assistant, treat as system context
            mapping[remaining_col] = 'system'

    # VALIDATION: Ensure we have at least user + assistant
    has_user = any(role == 'user' for role in mapping.values())
    has_assistant = any(role == 'assistant' for role in mapping.values())

    if not has_user and len(remaining_columns) > 0:
        for col in remaining_columns:
            if col not in mapping:
                mapping[col] = 'user'
                has_user = True
                break

    if has_user and has_assistant:
        return mapping

    return None

def detect_multimodal_dataset(dataset):
    """
    Detects if dataset contains multimodal data (images/vision).

    Returns:
        dict: {
            "is_multimodal": bool,
            "multimodal_columns": list of column names containing image data,
            "modality_types": list of detected types (e.g., ["image", "pixel"])
        }
    """
    sample = next(iter(dataset))
    column_names = list(sample.keys())

    # Keywords that indicate multimodal/image data
    multimodal_keywords = ['image', 'img', 'pixel']

    multimodal_columns = []
    modality_types = set()

    for col_name in column_names:
        col_lower = col_name.lower()

        for keyword in multimodal_keywords:
            if keyword in col_lower:
                multimodal_columns.append(col_name)
                modality_types.add(keyword)
                break  # Don't check other keywords for this column

    return {
        "is_multimodal": len(multimodal_columns) > 0,
        "multimodal_columns": multimodal_columns,
        "modality_types": list(modality_types)
    }
pass

def detect_vlm_dataset_structure(dataset):
    """
    Detects if VLM dataset is:
    - Standard VLM messages format (image objects in content)
    - Llava format (image indices + separate images column)
    - Simple format needing conversion (image + text columns)
    """
    try:
        sample = next(iter(dataset))
    except StopIteration:
        return {
            "format": "unknown",
            "needs_conversion": None,
            "image_column": None,
            "text_column": None,
            "messages_column": None,
        }

    column_names = set(sample.keys())

    # Check if has messages column
    if "messages" in column_names:
        messages = sample["messages"]

        if messages and len(messages) > 0:
            first_msg = messages[0]
            if "content" in first_msg:
                content = first_msg["content"]

                if isinstance(content, list) and len(content) > 0:
                    if isinstance(content[0], dict) and "type" in content[0]:

                        # Check for llava format
                        has_index = any('index' in item for item in content if isinstance(item, dict))
                        has_images_column = 'images' in column_names

                        if has_index and has_images_column:
                            return {
                                "format": "vlm_messages_llava",
                                "needs_conversion": True,
                                "messages_column": "messages",
                                "image_column": "images",
                                "text_column": None,
                            }

                        # Standard VLM format
                        has_image = any('image' in item for item in content if isinstance(item, dict))
                        if has_image:
                            return {
                                "format": "vlm_messages",
                                "needs_conversion": False,
                                "messages_column": "messages",
                                "image_column": None,
                                "text_column": None,
                            }

    # Find image and text columns using metadata filtering

    # Define metadata patterns to EXCLUDE
    metadata_patterns = {
        'suffixes': ['_id', '_url', '_name', '_filename', '_uri', '_link', '_key', '_index'],
        'prefixes': ['id_', 'url_', 'name_', 'filename_', 'uri_', 'link_', 'key_', 'index_'],
    }

    # Image-related keywords
    image_keywords = ['image', 'img', 'photo', 'picture', 'pic', 'visual', 'scan']

    # Text-related keywords
    text_keywords = ['text', 'caption', 'description', 'answer', 'output', 'response', 'label']

    def is_metadata_column(col_name):
        """Check if column name looks like metadata."""
        col_lower = col_name.lower()

        # Check suffixes
        if any(col_lower.endswith(suffix) for suffix in metadata_patterns['suffixes']):
            return True

        # Check prefixes
        if any(col_lower.startswith(prefix) for prefix in metadata_patterns['prefixes']):
            return True

        return False

    def find_image_column():
        """Find image column by filtering out metadata and checking keywords."""
        candidates = []

        for col in column_names:
            col_lower = col.lower()

            # Check if contains image keywords
            if any(keyword in col_lower for keyword in image_keywords):
                # Verify it actually contains image data
                sample_value = sample[col]

                # PIL Image object (highest priority - even if name suggests metadata)
                if hasattr(sample_value, 'size') and hasattr(sample_value, 'mode'):
                    candidates.append((col, 100))  # High priority - actual PIL Image

                # String (could be path) - but lower priority if name is metadata-like
                elif isinstance(sample_value, str):
                    if is_metadata_column(col):
                        candidates.append((col, 30))  # Lower priority for metadata names
                    else:
                        candidates.append((col, 50))  # Medium priority

                # Dict with image data
                elif isinstance(sample_value, dict) and ('bytes' in sample_value or 'path' in sample_value):
                    candidates.append((col, 75))  # High-medium priority

        # Return highest priority candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    def find_text_column():
        """Find text column by filtering out metadata and checking keywords."""
        candidates = []

        for col in column_names:
            # Skip metadata columns
            if is_metadata_column(col):
                continue

            col_lower = col.lower()

            # Check if contains text keywords
            if any(keyword in col_lower for keyword in text_keywords):
                # Verify it's actually text
                sample_value = sample[col]

                if isinstance(sample_value, str) and len(sample_value) > 0:
                    # Longer text = higher priority (likely content, not just a label)
                    priority = min(len(sample_value), 1000)  # Cap at 1000
                    candidates.append((col, priority))

        # Return highest priority candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    found_image = find_image_column()
    found_text = find_text_column()

    if found_image and found_text:
        return {
            "format": "simple_image_text",
            "needs_conversion": True,
            "image_column": found_image,
            "text_column": found_text,
            "messages_column": None,
        }

    return {
        "format": "unknown",
        "needs_conversion": None,
        "image_column": found_image,
        "text_column": found_text,
        "messages_column": None,
    }
pass

def convert_to_vlm_format(
    dataset,
    instruction=None,
    text_column="text",
    image_column="image",
    dataset_name=None,
):
    """
    Converts simple {image, text} format to VLM messages format.

    Returns a LIST, not a HuggingFace Dataset (to preserve PIL Images).

    Returns:
        list: List of dicts with 'messages' field
    """
    from PIL import Image

    # Generate smart instruction if not provided
    if instruction is None:
        instruction_info = generate_smart_vlm_instruction(
            dataset,
            text_column=text_column,
            image_column=image_column,
            dataset_name=dataset_name,
        )

        instruction = instruction_info["instruction"]
        instruction_column = instruction_info.get("instruction_column")
        uses_dynamic = instruction_info["uses_dynamic_instruction"]

        print(f"📝 Auto-detected instruction type: {instruction_info['instruction_type']}")
        print(f"📝 Confidence: {instruction_info['confidence']:.2f}")
        if not uses_dynamic:
            print(f"📝 Using instruction: '{instruction}'")
        else:
            print(f"📝 Using dynamic instructions from column: '{instruction_column}'")
    else:
        instruction_column = None
        uses_dynamic = False

    def _convert_single_sample(sample):
        """Convert a single sample to VLM format."""
        # Get image (might be PIL Image or path)
        image_data = sample[image_column]

        # Handle image paths
        if isinstance(image_data, str):
            image_data = Image.open(image_data).convert("RGB")

        # Get text
        text_data = sample[text_column]

        # Get instruction (static or dynamic)
        if uses_dynamic and instruction_column:
            current_instruction = sample[instruction_column]
        else:
            current_instruction = instruction

        # Build VLM messages - simple structure
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": current_instruction},
                    {"type": "image", "image": image_data}  # PIL object
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text_data}
                ]
            }
        ]

        # Return dict with messages
        return {"messages": messages}

    # Use list comprehension and return the LIST directly
    print(f"🔄 Converting {len(dataset)} samples to VLM format...")
    converted_list = [_convert_single_sample(sample) for sample in dataset]

    print(f"Converted {len(converted_list)} samples")

    # Return list, NOT Dataset
    return converted_list
pass

def generate_smart_vlm_instruction(
    dataset,
    text_column="text",
    image_column="image",
    dataset_name=None,
):
    """
    Generate smart, context-aware instruction for VLM datasets using heuristics.

    Strategy:
    1. Check for explicit question/instruction columns → use that
    2. Infer from text column name + sample content
    3. Analyze dataset name for task hints
    4. Fall back to generic instruction

    Returns:
        dict: {
            "instruction": str or None,  # None means use column content
            "instruction_type": "explicit" | "inferred" | "generic",
            "uses_dynamic_instruction": bool,  # True if instruction varies per sample
            "confidence": float,  # 0.0 to 1.0
        }
    """
    import re

    column_names = set(next(iter(dataset)).keys())
    sample = next(iter(dataset))

    # ===== LEVEL 1: Explicit Instruction Columns =====
    # Check for columns that contain per-sample instructions
    question_columns = ["question", "query", "prompt", "instruction", "user_prompt"]

    for col in question_columns:
        if col in column_names:
            # Check if this column has varied content (not just empty/same)
            sample_content = sample[col]
            if sample_content and str(sample_content).strip():
                return {
                    "instruction": None,  # Signal to use column content
                    "instruction_column": col,
                    "instruction_type": "explicit",
                    "uses_dynamic_instruction": True,
                    "confidence": 1.0,
                }

    # ===== LEVEL 2: Infer from Column Names + Content =====
    text_col_lower = text_column.lower()

    # Sample the text content to detect patterns
    text_sample = str(sample.get(text_column, ""))[:500]  # First 500 chars

    # Task-specific keywords and their instructions
    task_patterns = {
        # OCR / Transcription
        "ocr": {
            "keywords": ["ocr", "transcribe", "transcript"],
            "content_hints": [r"[A-Za-z\u0600-\u06FF]{10,}"],  # Long text passages (Latin/Arabic)
            "instruction": "Transcribe all the text shown in this image.",
            "confidence": 0.9,
        },

        # LaTeX / Math
        "latex": {
            "keywords": ["latex", "math", "formula", "equation"],
            "content_hints": [r"\\[a-z]+\{", r"\^", r"_", r"\\frac"],  # LaTeX commands
            "instruction": "Convert this image to LaTeX notation.",
            "confidence": 0.95,
        },

        # Caption / Description
        "caption": {
            "keywords": ["caption", "description", "describe"],
            "content_hints": [],
            "instruction": "Provide a detailed description of this image.",
            "confidence": 0.85,
        },

        # Medical / Radiology
        "medical": {
            "keywords": ["medical", "radiology", "xray", "ct", "mri", "scan", "diagnosis"],
            "content_hints": [r"\b(lesion|radiograph|patient|diagnosis|findings)\b"],
            "instruction": "Analyze this medical image and describe the key findings.",
            "confidence": 0.9,
        },

        # Code / Programming
        "code": {
            "keywords": ["code", "program", "function", "algorithm"],
            "content_hints": [r"def |class |function|import |return "],
            "instruction": "Explain what this code visualization shows.",
            "confidence": 0.85,
        },

        # Chart / Graph
        "chart": {
            "keywords": ["chart", "graph", "plot", "visualization", "diagram"],
            "content_hints": [r"\b(axis|legend|bar|line|pie|scatter)\b"],
            "instruction": "Describe this chart or graph, including key data points and trends.",
            "confidence": 0.85,
        },

        # Document / Text Recognition
        "document": {
            "keywords": ["document", "page", "paragraph", "article"],
            "content_hints": [r"\n.*\n.*\n"],  # Multi-line text
            "instruction": "Extract and transcribe the text from this document image.",
            "confidence": 0.85,
        },
    }

    # Check column name matches
    best_match = None
    best_score = 0.0

    for task_name, task_info in task_patterns.items():
        score = 0.0

        # Check column name
        if any(keyword in text_col_lower for keyword in task_info["keywords"]):
            score += 0.5

        # Check dataset name if provided
        if dataset_name and any(keyword in dataset_name.lower() for keyword in task_info["keywords"]):
            score += 0.3

        # Check content patterns
        for pattern in task_info["content_hints"]:
            if re.search(pattern, text_sample, re.IGNORECASE):
                score += 0.4
                break

        if score > best_score:
            best_score = score
            best_match = task_info

    if best_match and best_score > 0.5:  # Confidence threshold
        return {
            "instruction": best_match["instruction"],
            "instruction_column": None,
            "instruction_type": "inferred",
            "uses_dynamic_instruction": False,
            "confidence": min(best_score, best_match["confidence"]),
        }

    # ===== LEVEL 3: Analyze Dataset Name =====
    if dataset_name:
        name_lower = dataset_name.lower()

        # Common dataset name patterns
        if "vqa" in name_lower or "question" in name_lower:
            return {
                "instruction": "Answer the question about this image.",
                "instruction_column": None,
                "instruction_type": "inferred",
                "uses_dynamic_instruction": False,
                "confidence": 0.75,
            }

        if "coco" in name_lower or "flickr" in name_lower:
            return {
                "instruction": "Provide a detailed caption for this image.",
                "instruction_column": None,
                "instruction_type": "inferred",
                "uses_dynamic_instruction": False,
                "confidence": 0.75,
            }

    # ===== LEVEL 4: Generic Fallback =====
    return {
        "instruction": "Describe this image in detail.",
        "instruction_column": None,
        "instruction_type": "generic",
        "uses_dynamic_instruction": False,
        "confidence": 0.5,
    }
pass

def convert_llava_to_vlm_format(dataset):
    """
    Converts Llava format to standard VLM format.

    Llava format:
    - messages: [{'content': [{'type': 'image', 'index': 0}, {'type': 'text', 'text': '...'}]}]
    - images: [PIL_Image1, PIL_Image2, ...]

    Standard VLM format:
    - messages: [{'content': [{'type': 'image', 'image': PIL_Image}, {'type': 'text', 'text': '...'}]}]
    """
    from PIL import Image

    print(f"🔄 Converting {len(dataset)} samples from Llava format to standard VLM format...")

    def _convert_single_sample(sample):
        """Convert a single llava sample to standard VLM format."""
        messages = sample["messages"]
        images = sample.get("images", [])

        # Process each message
        new_messages = []
        for msg in messages:
            new_content = []

            for item in msg["content"]:
                if item["type"] == "image":
                    # Replace index with actual PIL image
                    if "index" in item and item["index"] is not None:
                        img_idx = item["index"]
                        if img_idx < len(images):
                            pil_image = images[img_idx]
                            # Ensure it's PIL
                            if isinstance(pil_image, str):
                                pil_image = Image.open(pil_image).convert("RGB")

                            new_content.append({
                                "type": "image",
                                "image": pil_image  # Actual PIL object
                            })
                    else:
                        # No index, try to use first image
                        if len(images) > 0:
                            pil_image = images[0]
                            if isinstance(pil_image, str):
                                pil_image = Image.open(pil_image).convert("RGB")

                            new_content.append({
                                "type": "image",
                                "image": pil_image
                            })

                elif item["type"] == "text":
                    # Keep text as-is (only type + text)
                    new_content.append({
                        "type": "text",
                        "text": item.get("text", "")
                    })

            new_messages.append({
                "role": msg["role"],
                "content": new_content
            })

        return {"messages": new_messages}

    # Convert using list comprehension
    converted_list = [_convert_single_sample(sample) for sample in dataset]

    print(f"Converted {len(converted_list)} samples")
    return converted_list
pass

@dataclass
class DeepSeekOCRDataCollator:
    """
    Args:
        tokenizer: Tokenizer
        model: Model
        image_size: Size for image patches (default: 640)
        base_size: Size for global view (default: 1024)
        crop_mode: Whether to use dynamic cropping for large images
        train_on_responses_only: If True, only train on assistant responses (mask user prompts)
    """
    tokenizer: Any
    model: Any
    image_size: int = 640
    base_size: int = 1024
    crop_mode: bool = True
    image_token_id: int = 128815
    train_on_responses_only: bool = True

    def __init__(
        self,
        tokenizer,
        model,
        image_size: int = 640,
        base_size: int = 1024,
        crop_mode: bool = True,
        train_on_responses_only: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.image_token_id = 128815
        self.dtype = model.dtype  # Get dtype from model
        self.train_on_responses_only = train_on_responses_only

        self.image_transform = BasicImageTransform(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            normalize=True
        )
        self.patch_size = 16
        self.downsample_ratio = 4

        # Get BOS token ID from tokenizer
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            self.bos_id = tokenizer.bos_token_id
        else:
            self.bos_id = 0
            print(f"Warning: tokenizer has no bos_token_id, using default: {self.bos_id}")

    def deserialize_image(self, image_data) -> Image.Image:
        """Convert image data (bytes dict or PIL Image) to PIL Image in RGB mode"""
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        elif isinstance(image_data, dict) and 'bytes' in image_data:
            image_bytes = image_data['bytes']
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

    def calculate_image_token_count(self, image: Image.Image, crop_ratio: Tuple[int, int]) -> int:
        """Calculate the number of tokens this image will generate"""
        num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

        width_crop_num, height_crop_num = crop_ratio

        if self.crop_mode:
            img_tokens = num_queries_base * num_queries_base + 1
            if width_crop_num > 1 or height_crop_num > 1:
                img_tokens += (num_queries * width_crop_num + 1) * (num_queries * height_crop_num)
        else:
            img_tokens = num_queries * num_queries + 1

        return img_tokens

    def process_image(self, image: Image.Image) -> Tuple[List, List, List, List, Tuple[int, int]]:
        """
        Process a single image based on crop_mode and size thresholds

        Returns:
            Tuple of (images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio)
        """
        images_list = []
        images_crop_list = []
        images_spatial_crop = []

        if self.crop_mode:
            # Determine crop ratio based on image size
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image, min_num=2, max_num=9,
                    image_size=self.image_size, use_thumbnail=False
                )

            # Process global view with padding
            global_view = ImageOps.pad(
                image, (self.base_size, self.base_size),
                color=tuple(int(x * 255) for x in self.image_transform.mean)
            )
            images_list.append(self.image_transform(global_view).to(self.dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            # Process local views (crops) if applicable
            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(
                        self.image_transform(crop_img).to(self.dtype)
                    )

            # Calculate image tokens
            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

            tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                    num_queries * height_crop_num)

        else:  # crop_mode = False
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])

            # For smaller base sizes, resize; for larger, pad
            if self.base_size <= 640:
                resized_image = image.resize((self.base_size, self.base_size), Image.Resampling.LANCZOS)
                images_list.append(self.image_transform(resized_image).to(self.dtype))
            else:
                global_view = ImageOps.pad(
                    image, (self.base_size, self.base_size),
                    color=tuple(int(x * 255) for x in self.image_transform.mean)
                )
                images_list.append(self.image_transform(global_view).to(self.dtype))

            num_queries = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries + [self.image_token_id]) * num_queries
            tokenized_image += [self.image_token_id]

        return images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio

    def process_single_sample(self, messages: List[Dict]) -> Dict[str, Any]:
            """
            Process a single conversation into model inputs.
            """

            # --- 1. Setup ---
            images = []
            for message in messages:
                if "images" in message and message["images"]:
                    for img_data in message["images"]:
                        if img_data is not None:
                            pil_image = self.deserialize_image(img_data)
                            images.append(pil_image)

            if not images:
                raise ValueError("No images found in sample. Please ensure all samples contain images.")

            tokenized_str = []
            images_seq_mask = []
            images_list, images_crop_list, images_spatial_crop = [], [], []

            prompt_token_count = -1 # Index to start training
            assistant_started = False
            image_idx = 0

            # Add BOS token at the very beginning
            tokenized_str.append(self.bos_id)
            images_seq_mask.append(False)

            for message in messages:
                role = message["role"]
                content = message["content"]

                # Check if this is the assistant's turn
                if role == "<|Assistant|>":
                    if not assistant_started:
                        # This is the split point. All tokens added *so far*
                        # are part of the prompt.
                        prompt_token_count = len(tokenized_str)
                        assistant_started = True

                    # Append the EOS token string to the *end* of assistant content
                    content = f"{content.strip()} {self.tokenizer.eos_token}"

                # Split this message's content by the image token
                text_splits = content.split('<image>')

                for i, text_sep in enumerate(text_splits):
                    # Tokenize the text part
                    tokenized_sep = text_encode(self.tokenizer, text_sep, bos=False, eos=False)
                    tokenized_str.extend(tokenized_sep)
                    images_seq_mask.extend([False] * len(tokenized_sep))

                    # If this text is followed by an <image> tag
                    if i < len(text_splits) - 1:
                        if image_idx >= len(images):
                            raise ValueError(
                                f"Data mismatch: Found '<image>' token but no corresponding image."
                            )

                        # Process the image
                        image = images[image_idx]
                        img_list, crop_list, spatial_crop, tok_img, _ = self.process_image(image)

                        images_list.extend(img_list)
                        images_crop_list.extend(crop_list)
                        images_spatial_crop.extend(spatial_crop)

                        # Add image placeholder tokens
                        tokenized_str.extend(tok_img)
                        images_seq_mask.extend([True] * len(tok_img))

                        image_idx += 1 # Move to the next image

            # --- 3. Validation and Final Prep ---
            if image_idx != len(images):
                raise ValueError(
                    f"Data mismatch: Found {len(images)} images but only {image_idx} '<image>' tokens were used."
                )

            # If we never found an assistant message, we're in a weird state
            # (e.g., user-only prompt). We mask everything.
            if not assistant_started:
                print("Warning: No assistant message found in sample. Masking all tokens.")
                prompt_token_count = len(tokenized_str)

            # Prepare image tensors
            images_ori = torch.stack(images_list, dim=0)
            images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype=torch.long)

            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim=0)
            else:
                images_crop = torch.zeros((1, 3, self.base_size, self.base_size), dtype=self.dtype)

            return {
                "input_ids": torch.tensor(tokenized_str, dtype=torch.long),
                "images_seq_mask": torch.tensor(images_seq_mask, dtype=torch.bool),
                "images_ori": images_ori,
                "images_crop": images_crop,
                "images_spatial_crop": images_spatial_crop_tensor,
                "prompt_token_count": prompt_token_count, # This is now accurate
            }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples"""
        batch_data = []

        # Process each sample
        for feature in features:
            try:
                processed = self.process_single_sample(feature['messages'])
                batch_data.append(processed)
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

        if not batch_data:
            raise ValueError("No valid samples in batch")

        # Extract lists
        input_ids_list = [item['input_ids'] for item in batch_data]
        images_seq_mask_list = [item['images_seq_mask'] for item in batch_data]
        prompt_token_counts = [item['prompt_token_count'] for item in batch_data]

        # Pad sequences
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        images_seq_mask = pad_sequence(images_seq_mask_list, batch_first=True, padding_value=False)

        # Create labels
        labels = input_ids.clone()

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Mask image tokens (model shouldn't predict these)
        labels[images_seq_mask] = -100

        # Mask user prompt tokens when train_on_responses_only=True (only train on assistant responses)
        if self.train_on_responses_only:
            for idx, prompt_count in enumerate(prompt_token_counts):
                if prompt_count > 0:
                    labels[idx, :prompt_count] = -100

        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Prepare images batch (list of tuples)
        images_batch = []
        for item in batch_data:
            images_batch.append((item['images_crop'], item['images_ori']))

        # Stack spatial crop info
        images_spatial_crop = torch.cat([item['images_spatial_crop'] for item in batch_data], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }




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
        print(f"📝 Applying Unsloth chat template: {matched_template}")
        try:
            tokenizer = get_chat_template(
                tokenizer,
                chat_template=matched_template,
            )
        except Exception as e:
            print(f"⚠️ Failed to apply Unsloth template '{matched_template}': {e}")
            print(f"   Falling back to tokenizer's default chat template")
    else:
        print(f"📝 Using tokenizer's default chat template (no Unsloth template match)")

    return tokenizer
pass


TEMPLATE_TO_MODEL_MAPPER = {
    "phi-3.5": (
        "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",
        "microsoft/Phi-3.5-mini-instruct",
    ),
    "phi-3": (
        "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        "unsloth/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
        "unsloth/Phi-3-medium-4k-instruct",
        "microsoft/Phi-3-medium-4k-instruct",
        "unsloth/Phi-3-mini-4k-instruct-v0-bnb-4bit",
        "unsloth/Phi-3-mini-4k-instruct-v0",
    ),
    "phi-4": (
        "unsloth/phi-4-unsloth-bnb-4bit",
        "unsloth/phi-4",
        "microsoft/phi-4",
        "unsloth/phi-4-bnb-4bit",
        "unsloth/phi-4-reasoning-unsloth-bnb-4bit",
        "unsloth/phi-4-reasoning",
        "microsoft/Phi-4-reasoning",
        "unsloth/phi-4-reasoning-bnb-4bit",
        "unsloth/phi-4-reasoning-plus-unsloth-bnb-4bit",
        "unsloth/phi-4-reasoning-plus",
        "microsoft/Phi-4-reasoning-plus",
        "unsloth/phi-4-reasoning-plus-bnb-4bit",
        "unsloth/phi-4-mini-reasoning-unsloth-bnb-4bit",
        "unsloth/phi-4-mini-reasoning",
        "microsoft/Phi-4-mini-reasoning",
        "unsloth/phi-4-mini-reasoning-bnb-4bit",
        "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit",
        "unsloth/Phi-4-mini-instruct",
        "microsoft/Phi-4-mini-instruct",
        "unsloth/Phi-4-mini-instruct-bnb-4bit",
    ),
    "mistral": (
        "unsloth/mistral-7b-instruct-v0.1-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "unsloth/Mixtral-8x7B-Instruct-v0.1-unsloth-bnb-4bit",
        "unsloth/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "unsloth/Mixtral-8x7B-Instruct-v0.1-bnb-4bit",
        "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
        "unsloth/Mistral-Nemo-Instruct-2407",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "unsloth/Mistral-Large-Instruct-2407-bnb-4bit",
        "mistralai/Mistral-Large-Instruct-2407",
        "unsloth/Mistral-Small-Instruct-2409-bnb-4bit",
        "unsloth/Mistral-Small-Instruct-2409",
        "mistralai/Mistral-Small-Instruct-2409",
        "unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit",
        "unsloth/Mistral-Small-24B-Instruct-2501",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit",
        "unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit",
        "unsloth/Mistral-Small-3.1-24B-Instruct-2503",
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "unsloth/Mistral-Small-3.1-24B-Instruct-2503-bnb-4bit",
        "unsloth/Mistral-Small-3.2-24B-Instruct-2506-unsloth-bnb-4bit",
        "unsloth/Mistral-Small-3.2-24B-Instruct-2506",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "unsloth/Mistral-Small-3.2-24B-Instruct-2506-bnb-4bit",
    ),
    "llama": (
        "meta-llama/Llama-2-13b-chat-hf",
        "unsloth/llama-2-7b-chat-bnb-4bit",
        "unsloth/llama-2-7b-chat",
        "meta-llama/Llama-2-7b-chat-hf",
    ),
    "llama3": (
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "unsloth/llama-3-8b-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "unsloth/llama-3-70b-Instruct-bnb-4bit",
        "meta-llama/Meta-Llama-3-70B-Instruct",
    ),
    "llama-3.1": (
        "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        "unsloth/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit",
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "unsloth/Llama-3.1-Storm-8B-bnb-4bit",
        "unsloth/Llama-3.1-Storm-8B",
        "akjindal53244/Llama-3.1-Storm-8B",
        "unsloth/Hermes-3-Llama-3.1-8B-bnb-4bit",
        "unsloth/Hermes-3-Llama-3.1-8B",
        "NousResearch/Hermes-3-Llama-3.1-8B",
        "unsloth/Hermes-3-Llama-3.1-70B-bnb-4bit",
        "unsloth/Hermes-3-Llama-3.1-70B",
        "NousResearch/Hermes-3-Llama-3.1-70B",
        "unsloth/Hermes-3-Llama-3.1-405B-bnb-4bit",
        "NousResearch/Hermes-3-Llama-3.1-405B",
        "unsloth/Llama-3.1-Nemotron-70B-Instruct-bnb-4bit",
        "unsloth/Llama-3.1-Nemotron-70B-Instruct",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "unsloth/Llama-3.1-Tulu-3-8B-bnb-4bit",
        "unsloth/Llama-3.1-Tulu-3-8B",
        "allenai/Llama-3.1-Tulu-3-8B",
        "unsloth/Llama-3.1-Tulu-3-70B-bnb-4bit",
        "unsloth/Llama-3.1-Tulu-3-70B",
        "allenai/Llama-3.1-Tulu-3-70B",
    ),
    "llama-3.2": (
        "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit",
        "unsloth/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit",
        "unsloth/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct",
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
    ),
    "llama-3.3": (
        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "unsloth/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ),
    "gemma": (
        "unsloth/gemma-7b-it-bnb-4bit",
        "unsloth/gemma-7b-it",
        "google/gemma-7b-it",
        "google/gemma-2b-it",
        "unsloth/gemma-1.1-2b-it-bnb-4bit",
        "unsloth/gemma-1.1-2b-it",
        "google/gemma-1.1-2b-it",
        "unsloth/gemma-1.1-7b-it-bnb-4bit",
        "unsloth/gemma-1.1-7b-it",
        "google/gemma-1.1-7b-it",
    ),
    "gemma2": (
        "unsloth/gemma-2-9b-it-bnb-4bit",
        "unsloth/gemma-2-9b-it",
        "google/gemma-2-9b-it",
        "unsloth/gemma-2-27b-it-bnb-4bit",
        "unsloth/gemma-2-27b-it",
        "google/gemma-2-27b-it",
        "unsloth/gemma-2-2b-it-bnb-4bit",
        "unsloth/gemma-2-2b-it",
        "google/gemma-2-2b-it",
    ),
    "gemma-3": (
        "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-1b-it",
        "google/gemma-3-1b-it",
        "unsloth/gemma-3-1b-it-bnb-4bit",
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-4b-it",
        "google/gemma-3-4b-it",
        "unsloth/gemma-3-4b-it-bnb-4bit",
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-12b-it",
        "google/gemma-3-12b-it",
        "unsloth/gemma-3-12b-it-bnb-4bit",
        "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-27b-it",
        "google/gemma-3-27b-it",
        "unsloth/gemma-3-27b-it-bnb-4bit",
        "unsloth/gemma-3-270m-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-270m-it",
        "google/gemma-3-270m-it",
        "unsloth/gemma-3-270m-it-bnb-4bit",
        "unsloth/gemma-3-270m-unsloth-bnb-4bit",
        "unsloth/medgemma-4b-it-unsloth-bnb-4bit",
        "unsloth/medgemma-4b-it",
        "google/medgemma-4b-it",
        "unsloth/medgemma-4b-it-bnb-4bit",
        "unsloth/medgemma-27b-text-it-unsloth-bnb-4bit",
        "unsloth/medgemma-27b-text-it",
        "google/medgemma-27b-text-it",
        "unsloth/medgemma-27b-text-it-bnb-4bit",
    ),
    "gemma3n": (
        "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E4B-it",
        "google/gemma-3n-E4B-it",
        "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E2B-it",
        "google/gemma-3n-E2B-it",
        "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    ),
    "qwen2.5": (
        "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "unsloth/Qwen2.5-0.5B-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Math-1.5B-Instruct",
        "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Math-7B-Instruct",
        "Qwen/Qwen2.5-Math-7B-Instruct",
        "unsloth/Qwen2.5-Math-72B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Math-72B-Instruct",
        "Qwen/Qwen2.5-Math-72B-Instruct",
        "unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-3B-Instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-14B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "unsloth/Qwen2.5-VL-32B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-72B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "unsloth/Qwen2.5-VL-72B-Instruct-bnb-4bit",
        "unsloth/OpenThinker-7B-unsloth-bnb-4bit",
        "unsloth/OpenThinker-7B",
        "open-thoughts/OpenThinker-7B",
        "unsloth/OpenThinker-7B-bnb-4bit",
    ),
    "qwen3": (
        "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        "unsloth/Qwen3-0.6B",
        "Qwen/Qwen3-0.6B",
        "unsloth/Qwen3-0.6B-bnb-4bit",
        "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        "unsloth/Qwen3-1.7B",
        "Qwen/Qwen3-1.7B",
        "unsloth/Qwen3-1.7B-bnb-4bit",
        "unsloth/Qwen3-4B-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B",
        "Qwen/Qwen3-4B",
        "unsloth/Qwen3-4B-bnb-4bit",
        "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        "unsloth/Qwen3-8B",
        "Qwen/Qwen3-8B",
        "unsloth/Qwen3-8B-bnb-4bit",
        "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        "unsloth/Qwen3-14B",
        "Qwen/Qwen3-14B",
        "unsloth/Qwen3-14B-bnb-4bit",
        "unsloth/Qwen3-32B-unsloth-bnb-4bit",
        "unsloth/Qwen3-32B",
        "Qwen/Qwen3-32B",
        "unsloth/Qwen3-32B-bnb-4bit",
        "unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit",
        "unsloth/Qwen3-30B-A3B",
        "Qwen/Qwen3-30B-A3B",
        "unsloth/Qwen3-30B-A3B-bnb-4bit",
    ),
    "qwen3-instruct": (
        "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507",
        "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
        "unsloth/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "unsloth/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507",
        "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
    ),
    "qwen3-thinking": (
        "unsloth/QwQ-32B-Preview-bnb-4bit",
        "unsloth/QwQ-32B-Preview",
        "Qwen/QwQ-32B-Preview",
        "unsloth/QwQ-32B-unsloth-bnb-4bit",
        "unsloth/QwQ-32B",
        "Qwen/QwQ-32B",
        "unsloth/QwQ-32B-bnb-4bit",
        "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-4B-Thinking-2507",
        "unsloth/Qwen3-4B-Thinking-2507-bnb-4bit",
        "unsloth/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
    ),
    "zephyr": (
        "unsloth/zephyr-sft-bnb-4bit",
        "unsloth/zephyr-sft",
        "HuggingFaceH4/mistral-7b-sft-beta",
    ),
    "chatml": (
        "unsloth/yi-6b-bnb-4bit",
        "unsloth/yi-6b",
        "01-ai/Yi-6B",
        "unsloth/Hermes-2-Pro-Mistral-7B-bnb-4bit",
        "unsloth/Hermes-2-Pro-Mistral-7B",
        "NousResearch/Hermes-2-Pro-Mistral-7B",
        "unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit",
        "unsloth/OpenHermes-2.5-Mistral-7B",
        "teknium/OpenHermes-2.5-Mistral-7B",
    ),
    "gpt-oss": (
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-20b",
        "openai/gpt-oss-20b",
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-120b",
        "openai/gpt-oss-120b",
        "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    ),
    "starling": (
        "unsloth/Starling-LM-7B-beta-bnb-4bit",
        "unsloth/Starling-LM-7B-beta",
        "Nexusflow/Starling-LM-7B-beta",
    ),
    "yi-chat": (
        "unsloth/yi-34b-chat-bnb-4bit",
        "01-ai/Yi-6B-Chat",
        "01-ai/Yi-34B-Chat",
    )
}

MODEL_TO_TEMPLATE_MAPPER = {}

for key, values in TEMPLATE_TO_MODEL_MAPPER.items():
    for value in values:
        MODEL_TO_TEMPLATE_MAPPER[value] = key
    pass

    # Get lowercased
    lowered_key = key.lower()
    for value in values:
        MODEL_TO_TEMPLATE_MAPPER[value.lower()] = lowered_key
    pass
pass


TEMPLATE_TO_RESPONSES_MAPPER = {
    "gemma-3": {
        "instruction": "<start_of_turn>user\n",
        "response": "<start_of_turn>model\n",
    },
    "gemma3n": {
        "instruction": "<start_of_turn>user\n",
        "response": "<start_of_turn>model\n",
    },
    "qwen3-instruct": {
        "instruction": "<|im_start|>user\n",
        "response": "<|im_start|>assistant\n",
    },
    "qwen3-thinking": {
        "instruction": "<|im_start|>user\n",
        "response": "<|im_start|>assistant\n<think>\n",
    },
    "qwen3": {
        "instruction": "<|im_start|>user\n",
        "response": "<|im_start|>assistant\n",
    },
    "qwen2.5": {
        "instruction": "<|im_start|>user\n",
        "response": "<|im_start|>assistant\n",
    },
    "llama-3.2": {
        "instruction": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "llama-3.3": {
        "instruction": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "llama-3.1": {
        "instruction": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "llama3": {
        "instruction": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "phi-3": {
        "instruction": "<|user|>\n",
        "response": "<|assistant|>\n",
    },
    "phi-3.5": {
        "instruction": "<|user|>\n",
        "response": "<|assistant|>\n",
    },
    "phi-4": {
        "instruction": "<|im_start|>user<|im_sep|>",
        "response": "<|im_start|>assistant<|im_sep|>",
    },
    "mistral": {
        "instruction": "[INST] ",
        "response": " [/INST]",
    },
    "llama": {
        "instruction": "[INST] ",
        "response": " [/INST]",
    },
    "chatml": {
        "instruction": "<|im_start|>user\n",
        "response": "<|im_start|>assistant\n",
    },
    "zephyr": {
        "instruction": "<|user|>\n",
        "response": "<|assistant|>\n",
    },
    "unsloth": {
        "instruction": ">>> User: ",
        "response": ">>> Assistant: ",
    },
    "vicuna": {
        "instruction": "USER: ",
        "response": "ASSISTANT: ",
    },
    "alpaca": {
        "instruction": "### Instruction:\n",
        "response": "### Response:\n",
    },
    "gemma": {
        "instruction": "<start_of_turn>user\n",
        "response": "<start_of_turn>model\n",
    },
    "gemma2": {
        "instruction": "<start_of_turn>user\n",
        "response": "<start_of_turn>model\n",
    },
    "gpt-oss": {
        "instruction": "<|start|>user<|message|>",
        "response": "<|start|>assistant<|channel|>final<|message|>",
    },
    "lfm-2": {
        "instruction": "<|im_start|>user\n",
        "response": "<|im_start|>assistant\n",
    },
    "starling": {
        "instruction": "GPT4 Correct User: ",
        "response": "GPT4 Correct Assistant: ",
    },
    "yi-chat": {
        "instruction": "<|im_start|>user\n",
        "response": "<|im_start|>assistant\n",
    },
}
