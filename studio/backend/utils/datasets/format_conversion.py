"""
Format conversion utilities for dataset processing.

This module contains functions for converting between dataset formats
(Alpaca, ShareGPT, ChatML) and standardizing chat formats.
"""

from datasets import IterableDataset


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
        from utils.hardware import safe_num_proc

        if num_proc is None or type(num_proc) is not int:
            num_proc = safe_num_proc()
        else:
            num_proc = safe_num_proc(num_proc)

        dataset_map_kwargs['num_proc'] = num_proc
        dataset_map_kwargs['desc'] = "Standardizing chat format"

    return dataset.map(_standardize_dataset, **dataset_map_kwargs)


def convert_chatml_to_alpaca(dataset, batch_size=1000, num_proc=None):
    """
    Converts ChatML format (messages OR conversations) to Alpaca format.
    Handles both standardized and ShareGPT formats.

    Supports:
    - "messages" or "conversations" column
    - "role"/"content" (standard) or "from"/"value" (ShareGPT)
    """
    from torch.utils.data import IterableDataset

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
        from utils.hardware import safe_num_proc

        if num_proc is None or type(num_proc) is not int:
            num_proc = safe_num_proc()
        else:
            num_proc = safe_num_proc(num_proc)

        dataset_map_kwargs['num_proc'] = num_proc
        dataset_map_kwargs['desc'] = "Converting ChatML to Alpaca format"

    return dataset.map(_convert, **dataset_map_kwargs)


def convert_alpaca_to_chatml(dataset, batch_size=1000, num_proc=None):
    """
    Converts Alpaca format to ChatML format.

    Output format: Uses 'conversations' column with standard 'role'/'content' structure.
    """
    from torch.utils.data import IterableDataset

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
        from utils.hardware import safe_num_proc

        if num_proc is None or type(num_proc) is not int:
            num_proc = safe_num_proc()
        else:
            num_proc = safe_num_proc(num_proc)

        dataset_map_kwargs['num_proc'] = num_proc
        dataset_map_kwargs['desc'] = "Converting Alpaca to ChatML format"

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
    instruction=None,
    text_column="text",
    image_column="image",
    dataset_name=None,
    progress_callback=None,
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
            progress_callback(status_message=msg)

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
        # Get image (might be PIL Image, local path, or URL)
        image_data = sample[image_column]

        if isinstance(image_data, str):
            if image_data.startswith(("http://", "https://")):
                import fsspec
                from io import BytesIO
                with fsspec.open(image_data, "rb", expand=True) as f:
                    image_data = Image.open(BytesIO(f.read())).convert("RGB")
            else:
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

    total = len(dataset)
    first_image = next(iter(dataset))[image_column]
    has_urls = isinstance(first_image, str) and first_image.startswith(("http://", "https://"))

    # ── URL probe: 200 samples with parallel workers to estimate speed + failure rate ──
    PROBE_SIZE = 200
    MAX_FAIL_RATE = 0.3

    if has_urls and total > PROBE_SIZE:
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from utils.hardware import safe_num_proc

        num_workers = safe_num_proc()
        _notify(f"Probing {PROBE_SIZE} image URLs with {num_workers} workers...")
        print(f"🔍 Probing {PROBE_SIZE}/{total} image URLs with {num_workers} workers...")

        probe_samples = [dataset[i] for i in range(PROBE_SIZE)]
        probe_ok = 0
        probe_fail = 0
        probe_start = time.time()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_convert_single_sample, s): s for s in probe_samples}
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
            msg = (
                f"⚠️ {fail_rate:.0%} of the first {PROBE_SIZE} images failed to download "
                f"({probe_fail}/{probe_total}). "
                "This dataset has too many broken or unreachable image URLs. "
                "Consider using a dataset with embedded images instead."
            )
            print(msg)
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

        print(f"✅ Probe passed: {probe_ok}/{probe_total} ok, {probe_fail} failed ({fail_rate:.0%}), {throughput:.1f} img/s")
        print(f"⏱️ Estimated time for {total:,} samples: ~{eta_str}")
        _notify(info_msg)

    # ── Full conversion with progress ──
    from tqdm import tqdm

    print(f"🔄 Converting {total} samples to VLM format...")
    converted_list = []
    failed_count = 0

    if has_urls:
        # Parallel conversion for URL-based datasets
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from utils.hardware import safe_num_proc

        num_workers = safe_num_proc()
        batch_size = 500
        start_time = time.time()

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_samples = [dataset[i] for i in range(batch_start, batch_end)]

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_convert_single_sample, s): i for i, s in enumerate(batch_samples)}
                batch_results = [None] * len(batch_samples)
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        batch_results[idx] = future.result()
                    except Exception:
                        failed_count += 1

            converted_list.extend(r for r in batch_results if r is not None)

            # Progress update every batch
            elapsed = time.time() - start_time
            done = batch_end
            rate = done / elapsed if elapsed > 0 else 0
            remaining_time = (total - done) / rate if rate > 0 else 0
            eta_str = _format_eta(remaining_time)
            progress_msg = f"Downloading images: {done:,}/{total:,} ({done*100//total}%) | ~{eta_str} remaining | {failed_count} skipped"
            print(f"  [{done}/{total}] {rate:.1f} img/s, {failed_count} failed, ETA {eta_str}")
            _notify(progress_msg)
    else:
        # Sequential conversion for local/embedded images (fast, no I/O bottleneck)
        pbar = tqdm(dataset, total=total, desc="Converting VLM samples", unit="sample")
        for sample in pbar:
            try:
                converted_list.append(_convert_single_sample(sample))
            except Exception:
                failed_count += 1
            pbar.set_postfix(ok=len(converted_list), failed=failed_count, refresh=False)
        pbar.close()

    if failed_count > 0:
        fail_rate = failed_count / total
        print(f"⚠️ Skipped {failed_count}/{total} ({fail_rate:.0%}) samples with broken/unreachable images")
        # For datasets that skipped the probe (small URL datasets), check fail rate now
        if has_urls and fail_rate >= MAX_FAIL_RATE:
            msg = (
                f"⚠️ {fail_rate:.0%} of images failed to download ({failed_count}/{total}). "
                "This dataset has too many broken or unreachable image URLs. "
                "Consider using a dataset with embedded images instead."
            )
            _notify(msg)
            raise ValueError(msg)

    if len(converted_list) == 0:
        raise ValueError(
            f"All {total} samples failed during VLM conversion — no usable images found. "
            "This dataset may contain only image URLs that are no longer accessible."
        )

    print(f"✅ Converted {len(converted_list)}/{total} samples")
    _notify(f"Converted {len(converted_list):,}/{total:,} images successfully")

    # Return list, NOT Dataset
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

    print(f"✅ Converted {len(converted_list)} samples")
    return converted_list
