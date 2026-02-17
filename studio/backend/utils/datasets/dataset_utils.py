"""
Dataset utilities for format detection, conversion, and template application.

This module provides the main entry points for dataset processing:
- check_dataset_format: Lightweight check if manual mapping is needed (for frontend)
- format_dataset: Detects and normalizes dataset formats
- format_and_template_dataset: End-to-end processing with chat template application

All internal utilities have been moved to separate modules:
- format_detection: detect_dataset_format, detect_multimodal_dataset, etc.
- format_conversion: standardize_chat_format, convert_chatml_to_alpaca, etc.
- chat_templates: apply_chat_template_to_dataset, get_tokenizer_chat_template, etc.
- vlm_processing: generate_smart_vlm_instruction
- data_collators: DeepSeekOCRDataCollator, VLMDataCollator
- model_mappings: TEMPLATE_TO_MODEL_MAPPER
"""

# Import from modular files
from .format_detection import (
    detect_dataset_format,
    detect_multimodal_dataset,
    detect_vlm_dataset_structure,
    detect_custom_format_heuristic,
)
from .format_conversion import (
    standardize_chat_format,
    convert_chatml_to_alpaca,
    convert_alpaca_to_chatml,
    convert_to_vlm_format,
    convert_llava_to_vlm_format,
    get_expected_chat_column,
    rename_chat_column_in_list,
)
from .chat_templates import (
    apply_chat_template_to_dataset,
    get_dataset_info_summary,
    get_tokenizer_chat_template,
    DEFAULT_ALPACA_TEMPLATE,
)
from .vlm_processing import generate_smart_vlm_instruction
from .data_collators import DeepSeekOCRDataCollator, VLMDataCollator
from .model_mappings import TEMPLATE_TO_MODEL_MAPPER


def check_dataset_format(dataset, is_vlm: bool = False) -> dict:
    """
    Lightweight format check without processing - for frontend validation.
    
    Use this to quickly determine if user needs to manually map columns
    before calling the full format_and_template_dataset().
    
    Args:
        dataset: HuggingFace dataset
        is_vlm: Whether this is a Vision-Language Model dataset
    
    Returns:
        dict: {
            "requires_manual_mapping": bool - True if user must map columns,
            "detected_format": str - The detected format,
            "columns": list - Available column names for mapping UI,
            "suggested_mapping": dict or None - Auto-detected mapping if available,
            "detected_image_column": str or None - For VLM only,
            "detected_text_column": str or None - For VLM only,
        }
    """
    columns = list(dataset.column_names) if hasattr(dataset, 'column_names') else list(next(iter(dataset)).keys())
    
    # Auto-detect multimodal data regardless of is_vlm flag
    multimodal_info = detect_multimodal_dataset(dataset)
    if multimodal_info["is_multimodal"]:
        is_vlm = True  # Route to VLM detection automatically
    
    if is_vlm:
        vlm_structure = detect_vlm_dataset_structure(dataset)
        requires_mapping = vlm_structure["format"] == "unknown"
        
        return {
            "requires_manual_mapping": requires_mapping,
            "detected_format": vlm_structure["format"],
            "columns": columns,
            "suggested_mapping": None,
            "detected_image_column": vlm_structure.get("image_column"),
            "detected_text_column": vlm_structure.get("text_column"),
            "is_multimodal": multimodal_info["is_multimodal"],
            "multimodal_columns": multimodal_info.get("multimodal_columns"),
        }
    else:
        # LLM flow
        detected = detect_dataset_format(dataset)
        
        # If format is unknown, try heuristic detection
        if detected["format"] == "unknown":
            heuristic_mapping = detect_custom_format_heuristic(dataset)
            if heuristic_mapping:
                # Heuristic succeeded - no manual mapping needed
                return {
                    "requires_manual_mapping": False,
                    "detected_format": "custom_heuristic",
                    "columns": columns,
                    "suggested_mapping": heuristic_mapping,
                    "detected_image_column": None,
                    "detected_text_column": None,
                    "is_multimodal": False,
                    "multimodal_columns": None,
                }
            else:
                # Both detection and heuristic failed
                return {
                    "requires_manual_mapping": True,
                    "detected_format": "unknown",
                    "columns": columns,
                    "suggested_mapping": None,
                    "detected_image_column": None,
                    "detected_text_column": None,
                    "is_multimodal": False,
                    "multimodal_columns": None,
                }
        
        # Known format detected
        return {
            "requires_manual_mapping": False,
            "detected_format": detected["format"],
            "columns": columns,
            "suggested_mapping": None,
            "detected_image_column": None,
            "detected_text_column": None,
            "is_multimodal": False,
            "multimodal_columns": None,
        }

def _apply_user_mapping(dataset, mapping: dict, batch_size: int = 1000):
    """
    Apply user-provided column mapping to convert dataset to conversations format.
    
    Args:
        dataset: HuggingFace dataset
        mapping: Dict like {"question": "user", "answer": "assistant", "context": "system"}
        batch_size: Batch size for processing
    
    Returns:
        Dataset with single 'conversations' column (no extra columns preserved)
    """
    def _convert(examples):
        num_examples = len(examples[list(examples.keys())[0]])
        conversations = []
        
        for i in range(num_examples):
            convo = []
            role_order = ['system', 'user', 'assistant']
            
            for target_role in role_order:
                for col_name, role in mapping.items():
                    if role == target_role and col_name in examples:
                        content = examples[col_name][i]
                        # User explicitly mapped - always include even if empty
                        convo.append({"role": role, "content": str(content) if content else ""})
            
            conversations.append(convo)
        
        # ONLY return conversations - no extra columns
        return {"conversations": conversations}
    
    return dataset.map(_convert, batched=True, batch_size=batch_size, remove_columns=dataset.column_names)


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
    custom_format_mapping = None,
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
            "requires_manual_mapping": True if format detection failed and user must map columns,
            "warnings": list of warning messages
        }
    """

    # Detect multimodal first (needed for all flows)
    multimodal_info = detect_multimodal_dataset(dataset)

    # NEW: If user provided explicit mapping, skip detection and apply directly
    if custom_format_mapping:
        try:
            mapped_dataset = _apply_user_mapping(dataset, custom_format_mapping, batch_size)
            return {
                "dataset": mapped_dataset,
                "detected_format": "user_mapped",
                "final_format": "chatml_conversations",
                "chat_column": "conversations",
                "is_standardized": True,
                "requires_manual_mapping": False,
                "is_multimodal": multimodal_info["is_multimodal"],
                "multimodal_info": multimodal_info,
                "warnings": [f"Applied user-provided column mapping: {custom_format_mapping}"]
            }
        except Exception as e:
            return {
                "dataset": dataset,
                "detected_format": "user_mapped",
                "final_format": "unknown",
                "chat_column": None,
                "is_standardized": False,
                "requires_manual_mapping": True,
                "is_multimodal": multimodal_info["is_multimodal"],
                "multimodal_info": multimodal_info,
                "warnings": [f"Failed to apply user mapping: {e}"]
            }


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
                "requires_manual_mapping": False,
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
                    "requires_manual_mapping": False,
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
                    "requires_manual_mapping": True,
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
                "requires_manual_mapping": False,
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
                            "requires_manual_mapping": False,
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
                        "requires_manual_mapping": False,
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
                "requires_manual_mapping": True,
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
                "requires_manual_mapping": False,
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
                "requires_manual_mapping": False,
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
                "requires_manual_mapping": True,
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
                "requires_manual_mapping": False,
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
                "requires_manual_mapping": False,
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
                "requires_manual_mapping": False,
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
                    "requires_manual_mapping": False,
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
                    "requires_manual_mapping": True,
                    "is_multimodal": multimodal_info["is_multimodal"],
                    "multimodal_info": multimodal_info,
                    "warnings": warnings
                }

    else:
        raise ValueError(f"Unknown format_type: {format_type}")


def format_and_template_dataset(
    dataset,
    model_name,
    tokenizer,
    model=None,
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
    auto_detect_custom=True,
    auto_detect_mapping=True,
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
            "requires_manual_mapping": True if format detection failed and user must map columns,
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

        # NEW: If user provided explicit mapping for VLM, use it directly
        if custom_format_mapping:
            # Expect mapping like: {"image_col": "image", "caption_col": "text"}
            user_vlm_image_column = None
            user_vlm_text_column = None
            
            for col, role in custom_format_mapping.items():
                if role == "image":
                    user_vlm_image_column = col
                elif role in ["text", "user", "caption", "assistant"]:
                    user_vlm_text_column = col
            
            if user_vlm_image_column and user_vlm_text_column:
                try:
                    dataset = convert_to_vlm_format(
                        dataset,
                        instruction=vlm_instruction,
                        text_column=user_vlm_text_column,
                        image_column=user_vlm_image_column,
                        dataset_name=dataset_name,
                    )
                    warnings.append(f"Applied user VLM mapping: image='{user_vlm_image_column}', text='{user_vlm_text_column}'")
                    
                    return {
                        "dataset": dataset,
                        "detected_format": "user_mapped",
                        "final_format": "vlm_messages",
                        "chat_column": "messages",
                        "is_vlm": True,
                        "is_multimodal": True,
                        "multimodal_info": multimodal_info,
                        "success": True,
                        "requires_manual_mapping": False,
                        "warnings": warnings,
                        "errors": [],
                    }
                except Exception as e:
                    errors.append(f"Failed to apply user VLM mapping: {e}")
                    return {
                        "dataset": dataset,
                        "detected_format": "user_mapped",
                        "final_format": "vlm_conversion_failed",
                        "is_vlm": True,
                        "success": False,
                        "requires_manual_mapping": True,
                        "warnings": warnings,
                        "errors": errors,
                    }
            else:
                errors.append(
                    f"Invalid VLM mapping: need 'image' and 'text' roles. Got: {custom_format_mapping}"
                )
                return {
                    "dataset": dataset,
                    "detected_format": "user_mapped",
                    "final_format": "vlm_unknown",
                    "is_vlm": True,
                    "success": False,
                    "requires_manual_mapping": True,
                    "warnings": warnings,
                    "errors": errors,
                }

        # Auto-detect VLM structure
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
                    "requires_manual_mapping": True,
                    "warnings": warnings,
                    "errors": errors,
                }

        # Handle simple format
        elif vlm_structure["needs_conversion"]:
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
                    "requires_manual_mapping": True,
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
                    "requires_manual_mapping": True,
                    "warnings": warnings,
                    "errors": errors,
                }

        # Already in standard VLM format
        elif vlm_structure["format"] == "vlm_messages":
            dataset = [sample for sample in dataset]
            warnings.append("Dataset already in standard VLM messages format")

        # Defensive: rename chat column if model expects a different name
        expected_col = get_expected_chat_column(model) if model is not None else None
        # VLM data is a list of dicts — check what key the first item uses
        current_col = "messages"  # default from our converters
        if isinstance(dataset, list) and len(dataset) > 0:
            sample_keys = dataset[0].keys()
            if "conversations" in sample_keys:
                current_col = "conversations"
            elif "messages" in sample_keys:
                current_col = "messages"

        if expected_col and expected_col != current_col:
            warnings.append(
                f"Model expects '{expected_col}' but dataset has '{current_col}' — renaming."
            )
            dataset = rename_chat_column_in_list(dataset, current_col, expected_col)
            current_col = expected_col

        # Return as list
        return {
            "dataset": dataset,
            "detected_format": vlm_structure["format"],
            "final_format": "vlm_messages",
            "chat_column": current_col,
            "is_vlm": True,
            "is_multimodal": multimodal_info["is_multimodal"],
            "multimodal_info": multimodal_info,
            "vlm_structure": vlm_structure,
            "success": True,
            "requires_manual_mapping": False,
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
            auto_detect_custom=auto_detect_custom,
            custom_format_mapping=custom_format_mapping,
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
            model_name=model_name,
            custom_prompt_template=custom_prompt_template,
            add_eos_token=add_eos_token,
            remove_bos_prefix=remove_bos_prefix,
            custom_format_mapping=custom_format_mapping,
            auto_detect_mapping=auto_detect_mapping,
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
            "requires_manual_mapping": dataset_info.get("requires_manual_mapping", False),
            "warnings": all_warnings,
            "errors": all_errors,
            "summary": summary,
        }
