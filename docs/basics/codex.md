# How to Run Local LLMs with OpenAI Codex

This guide provides an overview of running local LLMs with OpenAI Codex. 

## Prerequisites

* Codex CLI `v0.123.0` or later
* `llama.cpp` built locally
* Unsloth model (e.g., `unsloth/Qwen3.6-27B-GGUF`)
* Quantized model (e.g., `Qwen3.6-27B-UD-IQ2_XXS.gguf`)

## Setup

1. Start `llama-server` and verify that `/v1/models`, `/health`, `/v1/chat/completions`, and `/v1/responses` work as expected.

## Compatibility Issues

Note that the current Codex CLI sends a `/v1/responses` tool payload that `llama.cpp` rejects unless a compatibility layer is added. To resolve this issue, you can use a local proxy to filter out non-`function` tools.

## Running Codex

To run Codex with the local `llama.cpp` endpoint, follow these steps:

1. Start `llama-server` with the desired model and quantization.
2. Create a local proxy to filter out non-`function` tools from the Codex request.
3. Run `codex exec` with the local `llama.cpp` endpoint.

## Troubleshooting

If you encounter an error with a message indicating that the request exceeds the available context size, try increasing the context size when starting `llama-server`.

## Example

Here's an example of how to start `llama-server` with a increased context size: