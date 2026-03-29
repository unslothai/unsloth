"""
KV Cache Reuse Benchmark — Multi-Turn Conversation

Demonstrates the speedup from passing pre-computed past_key_values to
model.generate() in a realistic multi-turn chat scenario.

The key insight:
  - Baseline: re-processes ALL history tokens from scratch on every call
  - KV Cache: only processes the NEW tokens, reuses cached KV for history

The longer the conversation history, the bigger the speedup.

Run in Colab (T4/A100):
    python examples/kv_cache_multiturn_benchmark.py
"""

import torch
import time
from unsloth import FastLanguageModel


# ── A realistic 15-turn tech support conversation ──────────────────────
# Each turn has substantive content to build up a large token count.
CONVERSATION_HISTORY = [
    {
        "role": "system",
        "content": (
            "You are a senior machine learning engineer helping a colleague debug "
            "and optimize their deep learning training pipeline. You give detailed, "
            "technically precise answers with code examples when appropriate. "
            "You have deep expertise in PyTorch, transformers, distributed training, "
            "mixed precision, gradient checkpointing, and GPU memory optimization."
        ),
    },
    {
        "role": "user",
        "content": (
            "Hey, I'm training a Llama-3-8B model on a single A100 80GB GPU using "
            "LoRA with rank 16. I'm getting OOM errors when I increase my batch size "
            "beyond 2. My sequences are 2048 tokens long. I'm using bfloat16 and "
            "gradient checkpointing is already enabled. What else can I try?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Several things to investigate:\n\n"
            "1. **LoRA target modules**: If you're applying LoRA to all linear layers "
            "(q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj), that's "
            "7 adapters per layer x 32 layers = 224 adapter pairs. Consider starting with "
            "just q_proj and v_proj (the original LoRA paper's recommendation).\n\n"
            "2. **Gradient accumulation**: Instead of batch_size=4, use batch_size=1 with "
            "gradient_accumulation_steps=4. Same effective batch size, much less memory.\n\n"
            "3. **Sequence packing**: If many of your samples are shorter than 2048, you're "
            "wasting compute on padding. Use a packing collator to concatenate multiple "
            "samples into a single 2048-token sequence.\n\n"
            "4. **Optimizer state**: AdamW keeps 2 state tensors per parameter. Try using "
            "8-bit Adam (bitsandbytes) or Adafactor which uses less memory.\n\n"
            "5. **Check for memory leaks**: Make sure you're not accidentally storing "
            "computation graphs. Use `torch.no_grad()` during evaluation and ensure "
            "`.detach()` is called on any tensors you log."
        ),
    },
    {
        "role": "user",
        "content": (
            "I switched to gradient accumulation of 4 with batch size 1, and I'm using "
            "8-bit Adam now. That freed up enough memory to run. But my training loss "
            "is plateauing around 1.8 after 500 steps. The learning rate is 2e-4 with "
            "a cosine schedule and 10% warmup. The dataset has about 50k instruction-"
            "response pairs. Should I adjust the learning rate or is something else wrong?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "A loss of 1.8 after 500 steps could mean several things:\n\n"
            "1. **Learning rate might be too high for LoRA**: With rank 16, try 1e-4 or "
            "even 5e-5. LoRA adapters have fewer parameters so they can overfit or "
            "oscillate with aggressive LR. The `lora_alpha` also matters — if alpha=16 "
            "and rank=16, the effective scaling is 1.0. Try alpha=32 for a scaling of 2.0.\n\n"
            "2. **Data quality**: Check a random sample of 100 training examples manually. "
            "Common issues: duplicate entries, empty responses, responses that are just "
            "the prompt repeated, or formatting inconsistencies in the chat template.\n\n"
            "3. **Tokenizer mismatch**: Ensure you're using the model's own tokenizer, "
            "not a generic one. Llama-3 uses a different tokenizer than Llama-2.\n\n"
            "4. **Label masking**: Are you masking the loss on the instruction tokens? "
            "If you're computing loss on both instruction and response, the model wastes "
            "capacity predicting the (fixed) instruction text. Use `DataCollatorForCompletionOnlyLM` "
            "or equivalent.\n\n"
            "5. **Baseline check**: What's the loss with LR=0 (no training)? That tells "
            "you the pretrained model's baseline on your data."
        ),
    },
    {
        "role": "user",
        "content": (
            "Great catches. I found two issues: (1) I wasn't masking instruction tokens — "
            "about 60% of each sequence is instruction, so the model was spending most of "
            "its capacity there. (2) I had about 2000 duplicate entries in my 50k dataset. "
            "After fixing both, the loss dropped to 1.2 after 500 steps and is still "
            "decreasing. Now I want to evaluate the model. What metrics should I use "
            "for an instruction-following model, and how do I set up evaluation properly?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "For instruction-following evaluation, I'd recommend a multi-layered approach:\n\n"
            "**Automated metrics:**\n"
            "- **Perplexity on a held-out set**: Keep 5-10% of your data as validation. "
            "Track eval loss every 100 steps. If eval loss starts increasing while train "
            "loss decreases, you're overfitting.\n"
            "- **ROUGE/BERTScore**: For tasks with reference answers. ROUGE-L captures "
            "longest common subsequence, BERTScore captures semantic similarity.\n"
            "- **Pass@k for code tasks**: If your data includes coding problems, measure "
            "functional correctness using unit tests.\n\n"
            "**Benchmark suites:**\n"
            "- **MMLU**: 57-subject multiple choice. Good for measuring general knowledge retention.\n"
            "- **ARC-Challenge**: Science questions requiring reasoning.\n"
            "- **HellaSwag**: Commonsense NLI.\n"
            "- **MT-Bench**: Multi-turn conversation quality, scored by GPT-4. This is "
            "particularly relevant for instruction-following.\n\n"
            "**Human evaluation:**\n"
            "- Create a test set of 50-100 diverse prompts spanning your target use cases.\n"
            "- Do blind A/B comparison between base model and finetuned model.\n"
            "- Rate on: helpfulness, accuracy, harmlessness, conciseness.\n\n"
            "For the automated setup, use `lm-evaluation-harness` from EleutherAI:\n"
            "```bash\n"
            "lm_eval --model hf --model_args pretrained=your_model --tasks mmlu,arc_challenge,hellaswag\n"
            "```"
        ),
    },
    {
        "role": "user",
        "content": (
            "I ran MMLU and got 62.3 (base was 65.1), so there's a slight regression on "
            "general knowledge. MT-Bench score improved from 6.2 to 7.4 though. The eval "
            "loss is still decreasing. Should I be worried about the MMLU drop? Also, I "
            "want to eventually serve this model in production with low latency. What "
            "should I be thinking about for inference optimization?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "**On the MMLU regression:**\n"
            "A ~3 point drop is common and usually acceptable when finetuning for a specific "
            "task. It's called 'catastrophic forgetting' — the model trades some general "
            "knowledge for task-specific capability. To mitigate it:\n"
            "- Mix in 5-10% general instruction data (like Alpaca or SlimOrca) with your "
            "domain data\n"
            "- Use a lower learning rate (less aggressive updates)\n"
            "- Train for fewer steps — the MT-Bench improvement might have peaked earlier "
            "than where you are now, before MMLU started dropping\n\n"
            "**For production inference optimization:**\n\n"
            "1. **Quantization**: Convert to GGUF format with 4-bit quantization (Q4_K_M is "
            "a good balance). This cuts model size from 16GB to ~4.5GB with minimal quality loss.\n\n"
            "2. **KV Cache optimization**: This is critical for multi-turn conversations. "
            "Pre-compute and cache the KV pairs for your system prompt and any static context. "
            "On subsequent user turns, you only need to compute attention for the new tokens. "
            "For a 1000-token system prompt, this can save 40-60% of first-token latency.\n\n"
            "3. **Continuous batching**: Use vLLM or TGI which dynamically batch incoming "
            "requests. This dramatically improves throughput compared to static batching.\n\n"
            "4. **Speculative decoding**: Use a small draft model (e.g., Llama-3-1B) to "
            "propose tokens, then verify with the large model in parallel. Can give 2-3x "
            "speedup on generation.\n\n"
            "5. **PagedAttention**: vLLM's paged KV cache management avoids memory "
            "fragmentation and allows serving more concurrent users."
        ),
    },
    {
        "role": "user",
        "content": (
            "The KV cache optimization point is exactly what I need. My application has a "
            "2000-token system prompt that's the same for every user. Right now, every "
            "request re-processes that entire system prompt. Can you show me how to "
            "pre-compute the KV cache for the system prompt and reuse it across requests? "
            "I'm using the Hugging Face transformers library."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Here's the pattern for KV cache reuse with a static prefix:\n\n"
            "```python\n"
            "import torch\n"
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
            "model = AutoModelForCausalLM.from_pretrained('your-model')\n"
            "tokenizer = AutoTokenizer.from_pretrained('your-model')\n\n"
            "# Step 1: Pre-compute KV cache for the static system prompt\n"
            "system_prompt = 'Your long system prompt here...'\n"
            "system_tokens = tokenizer(system_prompt, return_tensors='pt').to(model.device)\n\n"
            "with torch.no_grad():\n"
            "    outputs = model(**system_tokens, use_cache=True)\n"
            "    cached_kv = outputs.past_key_values  # Save this!\n\n"
            "# Step 2: For each user request, concatenate and pass cached KV\n"
            "user_msg = 'User question here'\n"
            "full_input = tokenizer(system_prompt + user_msg, return_tensors='pt')\n\n"
            "output = model.generate(\n"
            "    **full_input,\n"
            "    past_key_values=cached_kv,  # Reuse pre-computed cache\n"
            "    max_new_tokens=512,\n"
            "    use_cache=True,\n"
            ")\n"
            "```\n\n"
            "**Important details:**\n"
            "- The `past_key_values` must correspond to a prefix of `input_ids`. The model "
            "will skip recomputing attention for those prefix tokens.\n"
            "- You should deep-copy `cached_kv` if serving multiple concurrent requests, "
            "since generation modifies the cache in-place.\n"
            "- With Unsloth, this is even simpler because it handles the cache format "
            "conversion automatically.\n"
            "- For the best latency, also pass only the new tokens as input_ids (not the "
            "full conversation), but this requires careful position_ids management."
        ),
    },
    {
        "role": "user",
        "content": (
            "That makes sense. One more question about the training side — I noticed that "
            "some of my training examples have very long outputs (1500+ tokens) while most "
            "are under 200 tokens. The long examples seem to dominate the loss. Should I "
            "truncate them, or is there a better way to handle this imbalance?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "This is a common issue. Long examples contribute more to the loss because "
            "there are more tokens to predict. Several strategies:\n\n"
            "1. **Loss weighting by sample, not by token**: Instead of summing loss over "
            "all tokens and dividing by total token count, average the per-sample losses. "
            "This gives equal weight to short and long examples. In HF Trainer, set "
            "`num_items_in_batch` in your data collator.\n\n"
            "2. **Truncate smartly**: Don't just chop at 2048 tokens. For long outputs, "
            "consider splitting them into multiple training examples with overlapping context. "
            "For example, a 3000-token response becomes two examples: tokens 0-2048 and "
            "tokens 1024-3072, each with the full instruction as prefix.\n\n"
            "3. **Curriculum learning**: Train on shorter examples first (1-2 epochs), then "
            "introduce longer examples. This helps the model learn the basic task structure "
            "before dealing with complex, long-form outputs.\n\n"
            "4. **Packing with length grouping**: Sort your dataset by length and pack "
            "similarly-sized examples together. This minimizes padding waste and ensures "
            "each batch has a consistent compute cost.\n\n"
            "5. **Analyze the long examples**: Are those 1500+ token outputs actually high "
            "quality? Sometimes long outputs are verbose or contain repetition. You might "
            "get better results by summarizing them to 500 tokens using a stronger model."
        ),
    },
    {
        "role": "user",
        "content": (
            "I checked and about half the long outputs are genuinely detailed (code walkthroughs, "
            "step-by-step math), but the other half are indeed verbose. I'll clean those up. "
            "Now, I want to scale to multi-GPU training with 4x A100s. I've never done "
            "distributed training before. What's the simplest way to go from single-GPU "
            "LoRA training to multi-GPU? Do I need to change my training script significantly?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Going from 1 GPU to 4 GPUs with LoRA is actually straightforward. Here's the "
            "progression from simplest to most complex:\n\n"
            "**Option 1: Accelerate (Simplest)**\n"
            "Your existing script works almost unchanged. Just wrap the launch:\n"
            "```bash\n"
            "accelerate launch --num_processes 4 train.py\n"
            "```\n"
            "Accelerate handles DDP (DistributedDataParallel) automatically. Each GPU gets "
            "a copy of the model and processes different batches. Gradients are synchronized.\n\n"
            "**Option 2: FSDP (Better memory efficiency)**\n"
            "Fully Sharded Data Parallel shards the model across GPUs. With LoRA, this is "
            "mainly useful if the base model barely fits on one GPU. Configure via:\n"
            "```python\n"
            "from accelerate import FullyShardedDataParallelPlugin\n"
            "fsdp_plugin = FullyShardedDataParallelPlugin(\n"
            "    sharding_strategy='FULL_SHARD',\n"
            "    backward_prefetch='BACKWARD_PRE',\n"
            ")\n"
            "```\n\n"
            "**Option 3: DeepSpeed ZeRO Stage 2**\n"
            "Shards optimizer states and gradients across GPUs. Good middle ground:\n"
            "```bash\n"
            "accelerate launch --use_deepspeed --deepspeed_config ds_config.json train.py\n"
            "```\n\n"
            "**My recommendation for your case:**\n"
            "Start with Option 1 (plain DDP via Accelerate). With LoRA rank 16, the adapter "
            "parameters are tiny (~20M), so there's no benefit to sharding them. DDP gives "
            "you nearly linear speedup (3.6-3.8x with 4 GPUs). Only move to FSDP/DeepSpeed "
            "if you increase rank significantly or switch to full finetuning.\n\n"
            "**Things that DO change:**\n"
            "- Effective batch size = per_gpu_batch * num_gpus * gradient_accumulation\n"
            "- Learning rate may need to scale: try linear scaling (LR * num_gpus)\n"
            "- Set `dataloader_num_workers=4` to prevent CPU bottleneck\n"
            "- Use `torch.distributed.barrier()` before evaluation to sync all processes"
        ),
    },
    {
        "role": "user",
        "content": (
            "Perfect. I went with Accelerate DDP and it's working great — getting 3.7x "
            "speedup with 4 GPUs. The model is looking really good now on MT-Bench (7.8). "
            "One last thing: I need to deploy this model behind an API. My requirements "
            "are: p99 latency under 2 seconds for 200-token outputs, support 50 concurrent "
            "users, and the model should be running on a single A100. Is this feasible, and "
            "what serving stack do you recommend?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Let me do the math to check feasibility:\n\n"
            "**Back-of-envelope calculation:**\n"
            "- Llama-3-8B in fp16 on A100 generates ~40-60 tokens/sec per request\n"
            "- With 4-bit quantization: ~80-120 tokens/sec per request\n"
            "- 200 tokens at 100 tok/s = 2 seconds per request (tight but doable for p50)\n"
            "- For p99 under 2s with 50 concurrent users, you NEED continuous batching\n\n"
            "**Recommended stack: vLLM**\n"
            "```bash\n"
            "pip install vllm\n"
            "python -m vllm.entrypoints.openai.api_server \\\n"
            "    --model your-merged-model \\\n"
            "    --quantization awq \\\n"
            "    --max-model-len 4096 \\\n"
            "    --gpu-memory-utilization 0.9 \\\n"
            "    --max-num-seqs 64\n"
            "```\n\n"
            "**Why vLLM:**\n"
            "1. PagedAttention manages KV cache memory efficiently — no fragmentation\n"
            "2. Continuous batching: processes new requests without waiting for long ones "
            "to finish\n"
            "3. Prefix caching: your 2000-token system prompt is cached across users "
            "automatically (exactly what we discussed earlier!)\n"
            "4. OpenAI-compatible API — drop-in replacement for your application\n\n"
            "**Steps to deploy:**\n"
            "1. Merge your LoRA adapter into the base model\n"
            "2. Quantize with AWQ (better quality than GPTQ for serving)\n"
            "3. Load test with `locust` or `k6` to verify p99 meets your SLA\n"
            "4. Add a request queue (Redis/RabbitMQ) in front for burst handling\n\n"
            "**Caveats:**\n"
            "- 50 concurrent users at 200 tokens each = 10,000 tokens of generation "
            "happening simultaneously. The A100's 80GB can hold about 100 concurrent "
            "requests with 4-bit quantization and 4096 context length.\n"
            "- Monitor GPU memory: if KV cache fills up, vLLM will start queuing requests\n"
            "- Set `--enforce-eager` initially to avoid CUDA graph compilation overhead"
        ),
    },
]


def run_benchmark(model, tokenizer, history_turns, new_question, num_runs = 5):
    """
    Run a single benchmark: compare baseline vs KV cache generation.
    Returns (time_baseline, time_kv, num_history_tokens, outputs_match).
    """
    history = CONVERSATION_HISTORY[:history_turns]
    new_msg = [{"role": "user", "content": new_question}]

    # Tokenize history and full conversation
    text_history = tokenizer.apply_chat_template(
        history, tokenize = False, add_generation_prompt = False
    )
    text_full = tokenizer.apply_chat_template(
        history + new_msg, tokenize = False, add_generation_prompt = True
    )

    inputs_history = tokenizer(text_history, return_tensors = "pt").to("cuda")
    inputs_full = tokenizer(text_full, return_tensors = "pt").to("cuda")

    len_history = inputs_history.input_ids.shape[1]
    len_full = inputs_full.input_ids.shape[1]
    len_new = len_full - len_history

    # Verify prefix match
    if not torch.equal(
        inputs_full.input_ids[:, :len_history], inputs_history.input_ids
    ):
        # Re-align if tokenization differs
        inputs_history.input_ids = inputs_full.input_ids[:, :len_history]
        inputs_history.attention_mask = inputs_full.attention_mask[:, :len_history]

    # Pre-compute KV cache (this cost is amortized over many requests)
    with torch.no_grad():
        outputs_history = model(**inputs_history, use_cache = True)
        cached_kv = outputs_history.past_key_values

    gen_kwargs = dict(max_new_tokens = 50, use_cache = True, do_sample = False)

    # Warmup both paths
    model.generate(**inputs_full, max_new_tokens = 1)
    model.generate(**inputs_full, max_new_tokens = 1, past_key_values = cached_kv)
    torch.cuda.synchronize()

    # Benchmark baseline (no KV cache — re-processes all history tokens)
    times_baseline = []
    output_baseline = None
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output_baseline = model.generate(**inputs_full, **gen_kwargs)
        torch.cuda.synchronize()
        times_baseline.append(time.perf_counter() - t0)

    # Benchmark KV cache (reuses pre-computed history)
    times_kv = []
    output_kv = None
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output_kv = model.generate(
            **inputs_full, past_key_values = cached_kv, **gen_kwargs
        )
        torch.cuda.synchronize()
        times_kv.append(time.perf_counter() - t0)

    # Decode outputs
    text_baseline = tokenizer.decode(
        output_baseline[0][len_full:], skip_special_tokens = True
    )
    if output_kv.shape[1] > len_full:
        text_kv = tokenizer.decode(output_kv[0][len_full:], skip_special_tokens = True)
    else:
        text_kv = tokenizer.decode(output_kv[0], skip_special_tokens = True)

    # Use median for stable timing
    time_baseline = sorted(times_baseline)[len(times_baseline) // 2]
    time_kv = sorted(times_kv)[len(times_kv) // 2]

    return {
        "history_tokens": len_history,
        "new_tokens": len_new,
        "total_tokens": len_full,
        "time_baseline": time_baseline,
        "time_kv": time_kv,
        "speedup": time_baseline / time_kv if time_kv > 0 else float("inf"),
        "outputs_match": text_baseline.strip() == text_kv.strip(),
        "text_baseline": text_baseline.strip(),
        "text_kv": text_kv.strip(),
    }


def main():
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = 4096

    print(f"Loading {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    # ── Scaling test: increase history length and measure speedup ──────
    # We test with 4, 8, 12, and all 16 messages of history.
    # Each step roughly doubles the cached token count.
    test_cases = [
        (4, "What should I look at next?"),
        (8, "Can you recap what we've covered so far?"),
        (12, "What's the single most impactful optimization?"),
        (16, "Give me a 3-step action plan to go to production."),
    ]

    print("\n" + "=" * 72)
    print("  KV CACHE REUSE BENCHMARK — Multi-Turn Conversation")
    print("  Comparing: baseline (re-process all) vs cached (reuse history KV)")
    print("=" * 72)

    results = []
    for num_msgs, question in test_cases:
        num_turns = num_msgs // 2  # user+assistant pairs
        print(f"\n{'─' * 72}")
        print(f"  Conversation: {num_msgs} messages ({num_turns} turns)")
        print(f'  New question: "{question}"')
        print(f"{'─' * 72}")

        r = run_benchmark(model, tokenizer, num_msgs, question)
        results.append(r)

        print(f"  History:  {r['history_tokens']:>5} tokens (cached)")
        print(f"  New:      {r['new_tokens']:>5} tokens (processed)")
        print(f"  Total:    {r['total_tokens']:>5} tokens")
        print()
        print(f"  Baseline: {r['time_baseline']:.4f}s")
        print(f"  KV Cache: {r['time_kv']:.4f}s")
        print(f"  Speedup:  {r['speedup']:.2f}x")
        print(f"  Match:    {'YES' if r['outputs_match'] else 'NO'}")

        if not r["outputs_match"]:
            print(f"\n  Baseline output: {r['text_baseline'][:100]}...")
            print(f"  KV Cache output: {r['text_kv'][:100]}...")

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    print(
        f"  {'History':>8} {'New':>6} {'Baseline':>10} {'KV Cache':>10} {'Speedup':>8} {'Match':>6}"
    )
    print(f"  {'tokens':>8} {'tokens':>6} {'(sec)':>10} {'(sec)':>10} {'':>8} {'':>6}")
    print(f"  {'─' * 8} {'─' * 6} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 6}")
    for r in results:
        match = "YES" if r["outputs_match"] else "NO"
        print(
            f"  {r['history_tokens']:>8} {r['new_tokens']:>6} "
            f"{r['time_baseline']:>10.4f} {r['time_kv']:>10.4f} "
            f"{r['speedup']:>7.2f}x {match:>6}"
        )

    print(f"\n  Key takeaway: as conversation history grows, the speedup")
    print(f"  from KV cache reuse increases because the baseline must")
    print(f"  re-process more and more tokens that the KV path skips.\n")


if __name__ == "__main__":
    main()
