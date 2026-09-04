# Unsloth Studio MCP server

Unsloth can expose a local MCP server so an MCP client can inspect models and
GPU state, validate recipes, start or stop training, inspect recipe output, and
export a loaded model.

The server is disabled by default. Enable it for a local Unsloth process with:

```bash
UNSLOTH_STUDIO_ENABLE_MCP=1 \
UNSLOTH_STUDIO_MCP_TOKEN='use-a-local-secret' \
unsloth studio
```

The endpoint is `http://127.0.0.1:8888/mcp/` when Unsloth uses its default port
(a request to `/mcp` redirects to the canonical `/mcp/`). Use the actual Unsloth
port when it is configured differently.

The high-impact tools are:

- `studio_status` and `list_local_models` for discovery
- `get_training_status`, `start_training`, `stop_training`, and `list_training_runs`
- `validate_recipe`, `get_recipe_job_status`, and `get_recipe_job_dataset`
- `load_checkpoint` and `export_gguf`

`start_training` accepts the same fields as the Unsloth `TrainingStartRequest`.
The request is validated by the existing Pydantic model before a subprocess is
started. Export paths use the existing Unsloth validation as well.

The endpoint always requires `UNSLOTH_STUDIO_MCP_TOKEN` and checks an exact
Bearer token for both HTTP and WebSocket connections. Keep it on localhost
unless the deployment has an authenticated reverse proxy. The MCP endpoint is
intentionally opt-in because tools can consume GPU memory, write model
artifacts, and stop active work.
## Web search provider: Parallel Search MCP (opt-in)

Studio's built-in `web_search` uses DuckDuckGo by default. You can switch it to
Parallel's free Search MCP instead: Settings > Chat > Web search > Search
provider > Parallel.

- Endpoint: `https://search.parallel.ai/mcp` (Streamable HTTP, no account or key needed).
- Optional Bearer key: add a Parallel API key from `https://platform.parallel.ai`
  in the same settings section for higher rate limits. It is stored trimmed,
  never logged, and only sent as an `Authorization` header to Parallel.
- On any Parallel failure the search falls back to DuckDuckGo, so a chat never
  loses web search because of the provider switch.

Manual alternative: add it as a regular MCP server (address above, no headers)
and call its `web_search` / `web_fetch` tools directly.
