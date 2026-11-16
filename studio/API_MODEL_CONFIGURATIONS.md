# Select a GGUF configuration through the API

An API client can select different launch configurations of the same downloaded GGUF by choosing a model alias. Each configured alias appears as a separate name in `GET /v1/models`, subject to the underlying model's visibility. Ordinary aliases continue to work unchanged.

This initial version supports GGUF text generation with **Keep multiple models loaded** set to **Off**. Enable automatic model switching in Settings > API and use a credential allowed to load models. A loaded-only credential can use a configuration already loaded but cannot change it. Configured aliases do not download missing models or load non-GGUF models, and are not supported for embeddings.

## Configure aliases

With an owner credential, read `GET /api/settings/model-aliases` first, retain any existing entries, and replace the list with `PUT /api/settings/model-aliases`. The PUT replaces the entire alias list. For example (replace the model id with your downloaded model's exact id from `/v1/models`):

```json
{
  "aliases": [
    {
      "name": "swift-fast",
      "model": "your-org/your-model-GGUF:IQ3_S",
      "load_config": {
        "max_seq_length": 53248,
        "cache_type_kv": "q4_0",
        "speculative_type": "auto"
      }
    },
    {
      "name": "swift-long",
      "model": "your-org/your-model-GGUF:IQ3_S",
      "load_config": {
        "max_seq_length": 98304,
        "cache_type_kv": "q5_1",
        "speculative_type": "ngram"
      }
    }
  ]
}
```

All three `load_config` fields are required. Context must be an integer from 1 to 1,048,576; the model and available memory still constrain what can actually run. Cache types are `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`, `iq4_nl`, or `f32`. Speculation is `auto`, `off`, or `ngram`. Unknown keys are rejected. Omit `load_config` (or send null) for a plain alias. Arbitrary load arguments are not accepted.

The aliases UI preserves API-created configurations when editing names or other entries; configuring these controls initially uses the settings API. The three fields override the model's saved launch settings for that request without changing those saved settings.

## Select a configuration

Use the same server URL and authentication you already use for Studio's OpenAI-compatible API. Pick `swift-fast` or `swift-long` in the client, or submit:

```json
{
  "model": "swift-long",
  "messages": [{"role": "user", "content": "Summarize this document."}],
  "stream": true
}
```

Switching aliases may reload the local inference server and rebuild its cache. It reuses the downloaded weights; it does not keep two model copies resident. Repeating an already loaded configuration avoids a configuration reload. Auto speculation compares the original requested mode, even if the engine chooses a specific algorithm at runtime.

A mismatch with automatic switching disabled or a loaded-only credential returns 409 rather than answering with the wrong settings. A target not available locally returns 404; unsupported backend/multi-model/embedding combinations return 400 before loading. Existing model access restrictions and lifecycle policies still apply. A plain alias or bare model id retains the existing behavior of using that model's currently loaded configuration.

These example context sizes reproduce #17820's intended distinction; they are not a promise that a particular GPU can hold either configuration. For a quick local demonstration, use a small GGUF and windows such as 2048 and 4096, then verify the active context/cache/speculation in Studio status after each request.
