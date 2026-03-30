# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Format conversion utilities for dataset processing.

This module contains functions for converting between dataset formats
(Alpaca, ShareGPT, ChatML) and standardizing chat formats.
"""

import os

from datasets import IterableDataset
from loggers import get_logger

logger = get_logger(__name__)


def standardize_chat_format(
    dataset,
    tokenizer = None,
    aliases_for_system = [
        "system",
    ],
    aliases_for_user = [
        "user",
        "human",
        "input",
    ],
    aliases_for_assistant = [
        "gpt",
        "assistant",
        "output",
    ],
    batch_size = 1000,
    num_proc = None,
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
    length_first = len(set(uniques[keys[0]]))
    length_second = len(set(uniques[keys[1]]))

    # Determine which is role and which is content
    if length_first < length_second:
        role_key = keys[0]
        content_key = keys[1]
    else:
        role_key = keys[1]
        content_key = keys[0]

    # Mapping for aliases
    aliases_mapping = {}
    for x in aliases_for_system:
        aliases_mapping[x] = "system"
    for x in aliases_for_user:
        aliases_mapping[x] = "user"
    for x in aliases_for_assistant:
        aliases_mapping[x] = "assistant"

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
        "batched": True,
        "batch_size": batch_size,
    }

    if not isinstance(dataset, IterableDataset):
        from utils.hardware import dataset_map_num_proc

        if num_proc is None or type(num_proc) is not int:
            num_proc = dataset_map_num_proc()
        else:
            num_proc = dataset_map_num_proc(num_proc)

        dataset_map_kwargs["num_proc"] = num_proc
        dataset_map_kwargs["desc"] = "Standardizing chat format"

    return dataset.map(_standardize_dataset, **dataset_map_kwargs)


def convert_chatml_to_alpaca(dataset, batch_size = 1000, num_proc = None):
    """
    Converts ChatML format (messages OR conversations) to Alpaca format.
    Handles both standardized and ShareGPT formats.

    Supports:
    - "messages" or "conversations" column
    - "role"/"content" (standard) or "from"/"value" (ShareGPT)
    """
    try:
        from torch.utils.data import IterableDataset

        _is_torch_iterable = isinstance(dataset, IterableDataset)
    except ImportError:
        _is_torch_iterable = False

    def _convert(examples):
        # Auto-detect which column name is used
        chatml_data = (
            examples.get("messages")
            or examples.get("conversations")
            or examples.get("texts")
        )

        if chatml_data is None:
            raise ValueError(
                "No 'messages' or 'conversations' or 'texts' column found."
            )

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

        return {"instruction": instructions, "input": inputs, "output": outputs}

    dataset_map_kwargs = {
        "batched": True,
        "batch_size": batch_size,
    }

    if not _is_torch_iterable:
        from utils.hardware import dataset_map_num_proc

        if num_proc is None or type(num_proc) is not int:
            num_proc = dataset_map_num_proc()
        else:
            num_proc = dataset_map_num_proc(num_proc)

        dataset_map_kwargs["num_proc"] = num_proc
        dataset_map_kwargs["desc"] = "Converting ChatML to Alpaca format"

    return dataset.map(_convert, **dataset_map_kwargs)


def convert_alpaca_to_chatml(dataset, batch_size = 1000, num_proc = None):
    """
    Converts Alpaca format to ChatML format.

    Output format: Uses 'conversations' column with standard 'role'/'content' structure.
    """
    try:
        from torch.utils.data import IterableDataset

        _is_torch_iterable = isinstance(dataset, IterableDataset)
    except ImportError:
        _is_torch_iterable = False

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
                {"role": "assistant", "content": output},
            ]
            conversations.append(convo)

        return {"conversations": conversations}

    dataset_map_kwargs = {
        "batched": True,
        "batch_size": batch_size,
    }

    if not _is_torch_iterable:
        from utils.hardware import dataset_map_num_proc

        if num_proc is None or type(num_proc) is not int:
            num_proc = dataset_map_num_proc()
        else:
            num_proc = dataset_map_num_proc(num_proc)

        dataset_map_kwargs["num_proc"] = num_proc
        dataset_map_kwargs["desc"] = "Converting Alpaca to ChatML format"

    return dataset.map(_convert, **dataset_map_kwargs)


def _format_eta(seconds):
    """Format seconds into a human-readable ETA string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, _ = divmod(remainder, 60)
        return f"{h}h {m}m"


def convert_to_vlm_format(
    dataset,
    instruction = None,
    text_column = "text",
    image_column = "image",
    dataset_name = None,
    progress_callback = None,
):
    """
    Converts simple {image, text} format to VLM messages format.

    Returns a LIST, not a HuggingFace Dataset (to preserve PIL Images).

    For URL-based image datasets, runs a 200-sample parallel probe first to
    estimate download speed and failure rate, then reports time estimate or
    warning through progress_callback before proceeding with the full conversion.

    Args:
        progress_callback: Optional callable(status_message=str) to report
                          progress to the training overlay.

    Returns:
        list: List of dicts with 'messages' field
    """
    from PIL import Image
    from .vlm_processing import generate_smart_vlm_instruction

    def _notify(msg):
        """Send status update to the training overlay if callback is available."""
        if progress_callback:
            progress_callback(status_message = msg)

    # Generate smart instruction if not provided
    if instruction is None:
        instruction_info = generate_smart_vlm_instruction(
            dataset,
            text_column = text_column,
            image_column = image_column,
            dataset_name = dataset_name,
        )

        instruction = instruction_info["instruction"]
        instruction_column = instruction_info.get("instruction_column")
        uses_dynamic = instruction_info["uses_dynamic_instruction"]

        logger.info(
            f"📝 Auto-detected instruction type: {instruction_info['instruction_type']}"
        )
        logger.info(f"📝 Confidence: {instruction_info['confidence']:.2f}")
        if not uses_dynamic:
            logger.info(f"📝 Using instruction: '{instruction}'")
        else:
            logger.info(
                f"📝 Using dynamic instructions from column: '{instruction_column}'"
            )
    else:
        instruction_column = None
        uses_dynamic = False

    def _convert_single_sample(sample):
        """Convert a single sample to VLM format."""
        # Get image (might be PIL Image, local path, URL, or bare filename)
        image_data = sample[image_column]

        if isinstance(image_data, str):
            if image_data.startswith(("http://", "https://")):
                import fsspec
                from io import BytesIO

                with fsspec.open(image_data, "rb", expand = True) as f:
                    image_data = Image.open(BytesIO(f.read())).convert("RGB")
            elif _image_lookup is not None and image_data in _image_lookup:
                # Bare filename → resolve via HF repo lookup
                from huggingface_hub import hf_hub_download

                local_path = hf_hub_download(
                    dataset_name,
                    _image_lookup[image_data],
                    repo_type = "dataset",
                )
                image_data = Image.open(local_path).convert("RGB")
            else:
                image_data = Image.open(image_data).convert("RGB")

        # Get text (if list of strings, pick a random one — e.g. multiple captions)
        text_data = sample[text_column]
        if isinstance(text_data, list) and len(text_data) > 0:
            import random

            text_data = random.choice(text_data)

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
                    {"type": "image", "image": image_data},  # PIL object
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": text_data}]},
        ]

        # Return dict with messages
        return {"messages": messages}

    total = len(dataset)
    first_image = next(iter(dataset))[image_column]
    has_urls = isinstance(first_image, str) and first_image.startswith(
        ("http://", "https://")
    )

    # ── Bare-filename detection: images stored as filenames (e.g. "img_001.png")
    #    that don't exist locally.  Build a basename→repo_path lookup so we can
    #    resolve them via hf_hub_download during conversion.
    _image_lookup = None
    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff")
    if (
        not has_urls
        and isinstance(first_image, str)
        and not os.path.exists(first_image)
        and dataset_name
    ):
        try:
            from huggingface_hub import HfApi

            _notify("Resolving image filenames from HF repo...")
            logger.info(
                f"🔍 Image column contains bare filenames (e.g. '{first_image}') — building repo lookup..."
            )
            repo_files = HfApi().list_repo_files(dataset_name, repo_type = "dataset")
            _image_lookup = {
                os.path.basename(f): f
                for f in repo_files
                if any(f.lower().endswith(ext) for ext in _IMAGE_EXTS)
            }
            if first_image in _image_lookup:
                logger.info(
                    f"✅ Matched {len(_image_lookup)} image files in repo (e.g. '{first_image}' → '{_image_lookup[first_image]}')"
                )
            else:
                logger.info(
                    f"⚠️ Built lookup with {len(_image_lookup)} images but '{first_image}' not found — falling back to local open"
                )
                _image_lookup = None
        except Exception as e:
            logger.info(f"⚠️ Failed to build HF repo image lookup: {e}")
            _image_lookup = None

    # ── URL probe: 200 samples with parallel workers to estimate speed + failure rate ──
    PROBE_SIZE = 200
    MAX_FAIL_RATE = 0.3

    if has_urls and total > PROBE_SIZE:
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from utils.hardware import safe_thread_num_proc

        num_workers = safe_thread_num_proc()
        _notify(f"Probing {PROBE_SIZE} image URLs with {num_workers} workers...")
        logger.info(
            f"🔍 Probing {PROBE_SIZE}/{total} image URLs with {num_workers} workers..."
        )

        probe_samples = [dataset[i] for i in range(PROBE_SIZE)]
        probe_ok = 0
        probe_fail = 0
        probe_start = time.time()

        with ThreadPoolExecutor(max_workers = num_workers) as executor:
            futures = {
                executor.submit(_convert_single_sample, s): s for s in probe_samples
            }
            for future in as_completed(futures):
                try:
                    future.result()
                    probe_ok += 1
                except Exception:
                    probe_fail += 1

        probe_elapsed = time.time() - probe_start
        probe_total = probe_ok + probe_fail
        fail_rate = probe_fail / probe_total if probe_total > 0 else 0
        throughput = probe_total / probe_elapsed if probe_elapsed > 0 else 0

        if fail_rate >= MAX_FAIL_RATE:
            issues = [
                f"{fail_rate:.0%} of the first {PROBE_SIZE} image URLs failed to download ({probe_fail}/{probe_total})",
                "Images are external URLs, not embedded in the dataset",
            ]
            # Try LLM-friendly warning
            friendly = None
            try:
                from .llm_assist import llm_generate_dataset_warning

                friendly = llm_generate_dataset_warning(
                    issues,
                    dataset_name = dataset_name,
                    modality = "vision",
                    column_names = [image_column, text_column],
                )
            except Exception:
                pass
            msg = friendly or (
                f"⚠️ {fail_rate:.0%} of the first {PROBE_SIZE} images failed to download "
                f"({probe_fail}/{probe_total}). "
                "This dataset has too many broken or unreachable image URLs. "
                "Consider using a dataset with embedded images instead."
            )
            logger.info(msg)
            _notify(msg)
            raise ValueError(msg)

        # Estimate total time for remaining samples
        remaining = total - PROBE_SIZE
        estimated_seconds = remaining / throughput if throughput > 0 else 0
        eta_str = _format_eta(estimated_seconds)

        info_msg = (
            f"Downloading {total:,} images ({num_workers} workers, ~{throughput:.1f} img/s). "
            f"Estimated time: ~{eta_str}"
        )
        if probe_fail > 0:
            info_msg += f" | {fail_rate:.0%} broken URLs will be skipped"

        logger.info(
            f"✅ Probe passed: {probe_ok}/{probe_total} ok, {probe_fail} failed ({fail_rate:.0%}), {throughput:.1f} img/s"
        )
        logger.info(f"⏱️ Estimated time for {total:,} samples: ~{eta_str}")
        _notify(info_msg)

    # ── Full conversion with progress ──
    from tqdm import tqdm

    logger.info(f"🔄 Converting {total} samples to VLM format...")
    converted_list = []
    failed_count = 0

    if has_urls:
        # Parallel conversion for URL-based datasets
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from utils.hardware import safe_thread_num_proc

        num_workers = safe_thread_num_proc()
        batch_size = 500
        start_time = time.time()

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_samples = [dataset[i] for i in range(batch_start, batch_end)]

            with ThreadPoolExecutor(max_workers = num_workers) as executor:
                futures = {
                    executor.submit(_convert_single_sample, s): i
                    for i, s in enumerate(batch_samples)
                }
                batch_results = [None] * len(batch_samples)
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        batch_results[idx] = future.result()
                    except Exception as e:
                        failed_count += 1
                        if failed_count == 1:
                            logger.info(
                                f"First VLM conversion failure: {type(e).__name__}: {e}"
                            )

            converted_list.extend(r for r in batch_results if r is not None)

            # Progress update every batch
            elapsed = time.time() - start_time
            done = batch_end
            rate = done / elapsed if elapsed > 0 else 0
            remaining_time = (total - done) / rate if rate > 0 else 0
            eta_str = _format_eta(remaining_time)
            progress_msg = f"Downloading images: {done:,}/{total:,} ({done*100//total}%) | ~{eta_str} remaining | {failed_count} skipped"
            logger.info(
                f"  [{done}/{total}] {rate:.1f} img/s, {failed_count} failed, ETA {eta_str}"
            )
            _notify(progress_msg)
    else:
        # Sequential conversion for local/embedded images (fast, no I/O bottleneck)
        pbar = tqdm(dataset, total = total, desc = "Converting VLM samples", unit = "sample")
        for sample in pbar:
            try:
                converted_list.append(_convert_single_sample(sample))
            except Exception as e:
                failed_count += 1
                if failed_count == 1:
                    # Log the first failure to aid debugging
                    logger.info(
                        f"First VLM conversion failure: {type(e).__name__}: {e}"
                    )
            pbar.set_postfix(ok = len(converted_list), failed = failed_count, refresh = False)
        pbar.close()

    if failed_count > 0:
        fail_rate = failed_count / total
        logger.info(
            f"⚠️ Skipped {failed_count}/{total} ({fail_rate:.0%}) samples with broken/unreachable images"
        )
        # For datasets that skipped the probe (small URL datasets), check fail rate now
        if has_urls and fail_rate >= MAX_FAIL_RATE:
            issues = [
                f"{fail_rate:.0%} of images failed to download ({failed_count}/{total})",
                "Images are external URLs, not embedded in the dataset",
            ]
            friendly = None
            try:
                from .llm_assist import llm_generate_dataset_warning

                friendly = llm_generate_dataset_warning(
                    issues,
                    dataset_name = dataset_name,
                    modality = "vision",
                    column_names = [image_column, text_column],
                )
            except Exception:
                pass
            msg = friendly or (
                f"⚠️ {fail_rate:.0%} of images failed to download ({failed_count}/{total}). "
                "This dataset has too many broken or unreachable image URLs. "
                "Consider using a dataset with embedded images instead."
            )
            _notify(msg)
            raise ValueError(msg)

    if len(converted_list) == 0:
        issues = [
            f"All {total} samples failed during VLM conversion — no usable images found",
            f"Image column '{image_column}' may contain URLs that are no longer accessible, "
            "or local file paths that don't exist",
        ]
        friendly = None
        try:
            from .llm_assist import llm_generate_dataset_warning

            friendly = llm_generate_dataset_warning(
                issues,
                dataset_name = dataset_name,
                modality = "vision",
                column_names = [image_column, text_column],
            )
        except Exception:
            pass
        raise ValueError(
            friendly
            or (
                f"All {total} samples failed during VLM conversion — no usable images found. "
                "This dataset may contain only image URLs that are no longer accessible."
            )
        )

    logger.info(f"✅ Converted {len(converted_list)}/{total} samples")
    _notify(f"Converted {len(converted_list):,}/{total:,} images successfully")

    # Return list, NOT Dataset
    return converted_list


def convert_sharegpt_with_images_to_vlm_format(
    dataset,
    image_column = "image",
    messages_column = "conversations",
    dataset_name = None,
    progress_callback = None,
):
    """
    Converts ShareGPT/ChatML datasets that have a separate image column and
    ``<image>`` placeholders inside the conversation text.

    Example input::

        {
            "image": "sam/images/sa_545504.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\\nWhat is this photo about?"},
                {"from": "gpt",   "value": "The image captures..."}
            ]
        }

    Returns a list of dicts in standard VLM messages format (PIL Images inline).
    """
    from PIL import Image
    from tqdm import tqdm

    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff")
    _ROLE_MAP = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }

    def _notify(msg):
        if progress_callback:
            progress_callback(status_message = msg)

    # ── Resolve image loading strategy (same 3-tier as convert_to_vlm_format) ──
    total = len(dataset)
    first_image = next(iter(dataset))[image_column]

    _image_lookup = None
    if (
        isinstance(first_image, str)
        and not first_image.startswith(("http://", "https://"))
        and not os.path.exists(first_image)
        and dataset_name
    ):
        try:
            from huggingface_hub import HfApi

            _notify("Resolving image filenames from HF repo...")
            logger.info(
                f"🔍 Image column contains bare filenames (e.g. '{first_image}') — building repo lookup..."
            )
            repo_files = HfApi().list_repo_files(dataset_name, repo_type = "dataset")
            _image_lookup = {
                os.path.basename(f): f
                for f in repo_files
                if any(f.lower().endswith(ext) for ext in _IMAGE_EXTS)
            }
            # Also add the full relative paths as keys (for paths like "sam/images/sa_545504.jpg")
            for f in repo_files:
                if any(f.lower().endswith(ext) for ext in _IMAGE_EXTS):
                    _image_lookup[f] = f
            if first_image in _image_lookup:
                logger.info(
                    f"✅ Matched {len(_image_lookup)} image files in repo (e.g. '{first_image}' → '{_image_lookup[first_image]}')"
                )
            else:
                logger.info(
                    f"⚠️ Built lookup with {len(_image_lookup)} images but '{first_image}' not found — falling back to local open"
                )
                _image_lookup = None
        except Exception as e:
            logger.info(f"⚠️ Failed to build HF repo image lookup: {e}")
            _image_lookup = None

    def _resolve_image(image_data):
        """Resolve image data to a PIL Image object."""
        if hasattr(image_data, "size") and hasattr(image_data, "mode"):
            return image_data  # Already PIL
        if isinstance(image_data, str):
            if image_data.startswith(("http://", "https://")):
                import fsspec
                from io import BytesIO

                with fsspec.open(image_data, "rb", expand = True) as f:
                    return Image.open(BytesIO(f.read())).convert("RGB")
            elif _image_lookup is not None and image_data in _image_lookup:
                from huggingface_hub import hf_hub_download

                local_path = hf_hub_download(
                    dataset_name,
                    _image_lookup[image_data],
                    repo_type = "dataset",
                )
                return Image.open(local_path).convert("RGB")
            else:
                return Image.open(image_data).convert("RGB")
        if isinstance(image_data, dict) and (
            "bytes" in image_data or "path" in image_data
        ):
            if image_data.get("bytes"):
                from io import BytesIO

                return Image.open(BytesIO(image_data["bytes"])).convert("RGB")
            if image_data.get("path"):
                return Image.open(image_data["path"]).convert("RGB")
        raise ValueError(f"Cannot resolve image: {type(image_data)}")

    def _convert_single_sample(sample):
        """Convert a single ShareGPT+image sample to standard VLM format."""
        pil_image = _resolve_image(sample[image_column])
        conversation = sample[messages_column]

        new_messages = []
        for msg in conversation:
            role_raw = msg.get("from") or msg.get("role", "user")
            role = _ROLE_MAP.get(role_raw.lower(), role_raw.lower())
            text = msg.get("value") or msg.get("content") or ""

            # Split on <image> to interleave text and image content blocks
            if "<image>" in text:
                parts = text.split("<image>")
                content = []
                for i, part in enumerate(parts):
                    part = part.strip()
                    if part:
                        content.append({"type": "text", "text": part})
                    if i < len(parts) - 1:
                        content.append({"type": "image", "image": pil_image})
                # If <image> was the entire text, content might just be the image
                if not content:
                    content.append({"type": "image", "image": pil_image})
            else:
                content = [{"type": "text", "text": text}]

            new_messages.append({"role": role, "content": content})

        return {"messages": new_messages}

    # ── Full conversion with progress ──
    logger.info(f"🔄 Converting {total} samples from ShareGPT+image format...")
    converted_list = []
    failed_count = 0

    pbar = tqdm(dataset, total = total, desc = "Converting ShareGPT+image", unit = "sample")
    for sample in pbar:
        try:
            converted_list.append(_convert_single_sample(sample))
        except Exception as e:
            failed_count += 1
            if failed_count == 1:
                logger.info(f"⚠️ First conversion failure: {type(e).__name__}: {e}")
        pbar.set_postfix(ok = len(converted_list), failed = failed_count, refresh = False)
    pbar.close()

    if failed_count > 0:
        logger.info(
            f"⚠️ Skipped {failed_count}/{total} ({failed_count*100//total}%) samples"
        )

    if len(converted_list) == 0:
        raise ValueError(
            f"All {total} samples failed during ShareGPT+image conversion — "
            "no usable samples found."
        )

    logger.info(f"✅ Converted {len(converted_list)}/{total} samples")
    _notify(f"Converted {len(converted_list):,}/{total:,} samples successfully")
    return converted_list


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

    logger.info(
        f"🔄 Converting {len(dataset)} samples from Llava format to standard VLM format..."
    )

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

                            new_content.append(
                                {
                                    "type": "image",
                                    "image": pil_image,  # Actual PIL object
                                }
                            )
                    else:
                        # No index, try to use first image
                        if len(images) > 0:
                            pil_image = images[0]
                            if isinstance(pil_image, str):
                                pil_image = Image.open(pil_image).convert("RGB")

                            new_content.append({"type": "image", "image": pil_image})

                elif item["type"] == "text":
                    # Keep text as-is (only type + text)
                    new_content.append({"type": "text", "text": item.get("text", "")})

            new_messages.append({"role": msg["role"], "content": new_content})

        return {"messages": new_messages}

    # Convert using list comprehension
    converted_list = [_convert_single_sample(sample) for sample in dataset]

    logger.info(f"✅ Converted {len(converted_list)} samples")
    return converted_list
