import torch
from unsloth import FastLanguageModel
import time


def main():
    # 1. Load Model
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    print(f"Loading {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    # 2. Setup Inputs
    # To demonstrate the value of KV Cache, we need a LONG history.
    # The baseline has to process (History + New) every time.
    # The KV Cache version only processes (New) + (Cached History).

    long_text = """
    The Transformer is a deep learning architecture introduced by Google researchers in the 2017 paper "Attention Is All You Need". It has since become the foundation for many state-of-the-art natural language processing (NLP) models, including BERT, GPT, and T5. Unlike Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks, which process data sequentially, Transformers process the entire input sequence simultaneously using a mechanism called "self-attention". This allows for significantly greater parallelization during training and enables the model to capture long-range dependencies in the text more effectively.

    The core component of the Transformer is the "attention mechanism", which weighs the importance of different words in a sentence relative to each other. For example, in the sentence "The animal didn't cross the street because it was too tired", the attention mechanism helps the model understand that "it" refers to "the animal" rather than "the street". This capability is crucial for tasks like machine translation, text summarization, and question answering.

    Large Language Models (LLMs) like GPT-4 and Llama 3 are built upon the Transformer architecture. These models are trained on massive datasets comprising text from the internet, books, and articles. Through this training, they learn to predict the next word in a sequence, effectively learning the statistical structure of language. This simple objective, when scaled up with billions of parameters and terabytes of data, results in emergent capabilities such as reasoning, coding, and creative writing.

    One of the key challenges in deploying LLMs is their computational cost. Generating text token-by-token requires loading the model's weights into memory for each step. Additionally, the attention mechanism's computational complexity grows quadratically with the sequence length. This is where techniques like KV Caching come into play. KV Caching involves storing the Key (K) and Value (V) matrices computed for previous tokens so that they don't need to be recomputed at every step of generation. This dramatically reduces the computational overhead, especially for long sequences.

    Unsloth is an optimization library designed to make the fine-tuning and inference of these Large Language Models faster and more memory-efficient. By rewriting core kernels in Triton and CUDA, Unsloth achieves significant speedups over standard implementations. It optimizes the backward pass for training and the forward pass for inference, ensuring that researchers and developers can work with state-of-the-art models on consumer-grade hardware.

    The evolution of LLMs is moving at a breakneck pace. We are seeing models that can process millions of tokens of context, multimodal models that can understand images and audio, and agents that can take actions in the real world. Despite these advancements, the fundamental principles of the Transformer and the need for efficient computation remain constant. Tools like Unsloth play a vital role in democratizing access to this powerful technology.

    In recent years, the open-source community has played a pivotal role in advancing LLM technology. Models like Llama, Mistral, and Qwen have matched or exceeded the performance of proprietary models in many benchmarks. This open ecosystem fosters innovation, allowing developers to build specialized applications for healthcare, law, education, and more. However, running these models efficiently remains a hurdle, which is why optimization techniques are more important than ever.

    As we look to the future, we can expect further architectural innovations that may eventually supersede the Transformer. State Space Models (SSMs) like Mamba are gaining traction for their linear scaling properties. Hybrid architectures that combine the best of Transformers and SSMs are also being explored. Regardless of the architecture, the goal remains the same: to create intelligent systems that can understand and generate human language with high accuracy and efficiency.
    """
    messages_history = [
        {"role": "user", "content": f"Here is some context about Unsloth: {long_text}"},
        {
            "role": "assistant",
            "content": "I have read the context. How can I help you?",
        },
    ]
    messages_new = [
        {"role": "user", "content": "Summarize the context in one sentence."},
    ]

    # Prepare text
    text_history = tokenizer.apply_chat_template(
        messages_history, tokenize = False, add_generation_prompt = False
    )

    # Tokenize history first
    inputs_history = tokenizer(text_history, return_tensors = "pt").to("cuda")
    len_history_tokens = inputs_history.input_ids.shape[1]

    # Tokenize full conversation
    text_full = tokenizer.apply_chat_template(
        messages_history + messages_new, tokenize = False, add_generation_prompt = True
    )
    inputs_full = tokenizer(text_full, return_tensors = "pt").to("cuda")
    len_full_tokens = inputs_full.input_ids.shape[1]

    print(f"\nHistory Length (tokens): {len_history_tokens}")
    print(f"Full Prompt Length (tokens): {len_full_tokens}")

    # Verify Prefix Match
    prefix_match = torch.equal(
        inputs_full.input_ids[:, :len_history_tokens], inputs_history.input_ids
    )
    if not prefix_match:
        print("WARNING: Tokenization mismatch! Re-aligning history...")
        inputs_history = tokenizer(text_full, return_tensors = "pt").to("cuda")
        inputs_history.input_ids = inputs_full.input_ids[:, :len_history_tokens]
        inputs_history.attention_mask = inputs_full.attention_mask[
            :, :len_history_tokens
        ]

    # 3. Pre-compute KV Cache for the history
    print("\nPre-computing KV cache for history...")
    with torch.no_grad():
        outputs_history = model(**inputs_history, use_cache = True)
        past_key_values_history = outputs_history.past_key_values
    print("KV Cache computed.")

    # 4. Baseline Generation
    print("\n--- Baseline Generation (No KV passed) ---")

    # Warmup
    model.generate(**inputs_full, max_new_tokens = 1)
    torch.cuda.synchronize()

    start_time = time.time()
    output_baseline = model.generate(
        **inputs_full,
        max_new_tokens = 100,
        use_cache = True,
        do_sample = False,
    )
    torch.cuda.synchronize()
    end_time = time.time()

    time_baseline = end_time - start_time
    new_tokens_baseline = output_baseline[0][len_full_tokens:]
    response_text_baseline = tokenizer.decode(
        new_tokens_baseline, skip_special_tokens = True
    )

    print(f"Time: {time_baseline:.4f}s")
    # print(f"Output: {response_text_baseline.strip()}")

    # 5. KV Cache Generation
    print("\n--- KV Cache Generation (Passing custom KV) ---")

    torch.cuda.synchronize()
    start_time = time.time()
    output_kv = model.generate(
        **inputs_full,
        max_new_tokens = 100,
        past_key_values = past_key_values_history,
        use_cache = True,
        do_sample = False,
    )
    torch.cuda.synchronize()
    end_time = time.time()

    time_kv = end_time - start_time

    # output_kv includes input_ids? Usually yes.
    # But if we passed past_key_values, does it return full sequence?
    # Yes, generate usually returns full sequence.

    # However, if Unsloth sliced the input internally, `generate` might be confused about what "input" was?
    # No, `generate` manages the loop.

    # Let's decode carefully.
    # If output_kv is just new tokens (unlikely), or full.
    if output_kv.shape[1] > len_full_tokens:
        new_tokens_kv = output_kv[0][len_full_tokens:]
    else:
        # Fallback if something weird happened
        new_tokens_kv = output_kv[0]

    response_text_kv = tokenizer.decode(new_tokens_kv, skip_special_tokens = True)

    print(f"Time: {time_kv:.4f}s")
    # print(f"Output: {response_text_kv.strip()}")

    # 6. Comparison
    print("\n--- Comparison ---")
    print(f"Baseline Time: {time_baseline:.4f}s")
    print(f"KV Cache Time: {time_kv:.4f}s")
    print(f"Speedup: {time_baseline / time_kv:.2f}x")

    if response_text_baseline == response_text_kv:
        print("SUCCESS: Outputs match perfectly.")
    else:
        print("WARNING: Outputs do not match.")
        print(f"Baseline: {response_text_baseline}")
        print(f"KV Cache: {response_text_kv}")


if __name__ == "__main__":
    main()
