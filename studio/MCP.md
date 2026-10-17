# MCP in Unsloth Studio

## Connect Blender MCP

Blender MCP is **disabled by default**. Unsloth Studio downloads a pinned, checksum-verified
runtime on first enable/test and caches it on the backend machine. No commands,
Git or pip installs are needed. Subsequent starts use the cache without internet.
The Blender add-on is installed separately. No MCP archive or source is shipped
in Unsloth Studio's Python package or desktop build.

1. Open **Manage MCP servers → Blender**, approve the execution warning and choose
   **Enable Blender MCP**. Use a tool-capable model with MCP enabled for the chat.
2. Open **Setup help → Download Blender add-on** for
   [Blender's official page](https://www.blender.org/lab/mcp-server/).
3. In Blender 5.1+, enable **Preferences → System → Network → Allow Online Access**.
   Drag the website's install button into Blender twice: first to add the Blender
   Lab repository, then to install MCP. Alternatively, search **MCP** in
   **Get Extensions** after adding the repository.
4. Enable and start the add-on bridge, keep Blender open, then **Test connection**.

A green dot means Blender is connected; amber means only the MCP server is connected.
Setup help stays in the same Blender entry. Advanced settings configure the bridge
port (default `9876`) and optional Blender executable. This port is not an HTTP URL.

The bridge uses loopback on the **Unsloth Studio backend machine**, not a remote browser.
Unsloth Studio does not install or launch Blender during setup. Approved tools can run
Python, write files and launch background Blender. Existing tool permissions apply;
external model providers receive tool results. Keep the unauthenticated bridge local.

The downloaded runtime excludes the large API/manual reference corpus and its three
offline documentation tools. The official source is
https://projects.blender.org/lab/blender_mcp (GPL-3.0-or-later).
The pinned revision and SHA-256 are in `backend/integrations/blender/runtime.py`.
Downloads are staged and verified before activation; failures leave the server
disabled and can be retried with **Enable Blender MCP**. Merely opening the dialog
or launching Unsloth Studio does not download anything.

## Large tool catalogs

A server can expose dozens of tools with long descriptions and deeply nested
parameter schemas: Notion's catalog alone is about 65,000 tokens. Every tool is
listed in full while the catalog takes at most a tenth of the loaded local model's
context window. External providers always get the full listing.

Past that, Studio searches tools the way Claude Code and Codex do. The model gets
`mcp_tool_search`, whose description names every tool with a one-line summary,
instead of the schemas. A search ranks the catalog with the SQLite FTS5 BM25 index
and rank fusion the knowledge base uses (tool names and descriptions searched
separately) and loads the best matches, exact names always; the model then calls
them with their full schemas. A tool the model calls by name loads the same way.

Up front, the listing also carries a tool a `tool_choice` forces, the tools mapped
to take an attached image, and, within a third of the window, the tools the chat
ran on earlier turns (so the prompt prefix the model server caches holds) and up
to five tools ranked against the user's message. Tools a turn loads stay within
half of the window, the oldest leaving first; a tool too large for that is refused
with a message naming it.

## Unsloth Decisions MCP

When the Decision API is on (**Settings → API**), the chat's MCP menu lists
**Unsloth Decisions**. Enable it and a tool-capable chat model can call `decide`,
which asks the local Laya model the same typed questions `POST /v1/systemone`
answers (`noul`, `choice` and `score`), with the model chosen in Settings.

Other MCP clients reach the same tool at `http://127.0.0.1:8888/mcp/decisions/`
(use the actual Unsloth port). It takes the credentials `/v1/systemone` takes, so
send an Unsloth API key as `Authorization: Bearer sk-unsloth-...`.

<a id="studios-own-mcp-server"></a>

## Unsloth Studio's own MCP server

Unsloth Studio has its own MCP server at `/mcp/`. Coding agents such as Claude Code and
Codex can use it to load models, chat, generate images, audio and video,
transcribe, train and export. Every tool goes through Unsloth Studio's own routes as
your API key, so it sees what that key may see and nothing more.

### Turn it on

The server is off by default. While it is off, `/mcp/` answers 404. Start Unsloth
Studio with `UNSLOTH_STUDIO_ENABLE_MCP=1` to turn it on.

The endpoint is `http://127.0.0.1:8888/mcp/` on the default port. Always use
the trailing slash. Use your own address and port when they differ, for example
the LAN address or the Cloudflare tunnel URL.

### Authentication

Every request needs an Unsloth Studio API key: `Authorization: Bearer sk-unsloth-…`.
Create one in **Settings → API**. Unsloth Studio refuses everything else with 401:

- no key, an empty key, or a placeholder key such as `not-needed`, even when
  keyless API access is on;
- a signed-in session token;
- a revoked or expired key;
- the keys Unsloth Studio mints for its own recipe and Deep Research workflows.

A managed account's key works too. Its tools act for that account only.

Unsloth Studio does not offer OAuth. The OAuth discovery paths answer 404, so clients
that probe them fall back to the key.

A browser page from another site gets 403 `Origin not allowed for Unsloth Studio MCP`.
This also blocks browser-based MCP inspectors. Command-line agents send no
Origin and are not affected.

### Breaking changes

The static `UNSLOTH_STUDIO_MCP_TOKEN` is retired. A request that sends it gets
401 with this detail:

```
The MCP static token is no longer supported; use an Unsloth Studio API key (sk-unsloth-…)
```

If the variable is still set, Unsloth Studio logs one warning at startup. Unsloth Studio no
longer refuses to start when the token is missing.

The tools changed:

| Old tool | New tool |
|---|---|
| `list_local_models` | `list_models` |
| `get_training_status` | `studio_status` |
| `stop_training` | `cancel` with `kind="training"` |
| `validate_recipe` | `run_recipe` with `mode="validate"` |
| `get_recipe_job_status`, `get_recipe_job_dataset` | `get_job` with `kind="recipe"` |
| `load_checkpoint` and `export_gguf` | `export_model` |

`studio_status` and `start_training` keep their names, but now run as your key.

### Set up an agent

Each one reads the key from the `UNSLOTH_API_KEY` environment variable, so set
it before you start the agent. Replace `http://127.0.0.1:8888` with your own
address.

**Claude Code** (macOS or Linux):

```bash
claude mcp add --transport http unsloth-studio http://127.0.0.1:8888/mcp/ --header "Authorization: Bearer $UNSLOTH_API_KEY"
```

**Claude Code** (Windows PowerShell):

```powershell
claude mcp add --transport http unsloth-studio http://127.0.0.1:8888/mcp/ --header "Authorization: Bearer $env:UNSLOTH_API_KEY"
```

**OpenAI Codex**:

```bash
codex mcp add unsloth-studio --url http://127.0.0.1:8888/mcp/ --bearer-token-env-var UNSLOTH_API_KEY
# Optional, for long jobs, in ~/.codex/config.toml under [mcp_servers.unsloth-studio]:
# tool_timeout_sec = 300
```

**OpenCode**, in `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "unsloth-studio": {
      "type": "remote",
      "url": "http://127.0.0.1:8888/mcp/",
      "oauth": false,
      "headers": {
        "Authorization": "Bearer {env:UNSLOTH_API_KEY}"
      }
    }
  }
}
```

**Hermes Agent**, in `~/.hermes/config.yaml`, then run `/reload-mcp` in Hermes:

```yaml
mcp_servers:
  unsloth_studio:
    url: "http://127.0.0.1:8888/mcp/"
    headers:
      Authorization: "Bearer ${UNSLOTH_API_KEY}"
```

**OpenClaw**, in `~/.openclaw/openclaw.json`:

```json
{
  "mcp": {
    "servers": {
      "unsloth-studio": {
        "url": "http://127.0.0.1:8888/mcp/",
        "transport": "streamable-http",
        "headers": {
          "Authorization": "Bearer ${UNSLOTH_API_KEY}"
        }
      }
    }
  }
}
```

**Mistral Vibe**, in `~/.vibe/config.toml`:

```toml
[[mcp_servers]]
name = "unsloth_studio"
transport = "streamable-http"
url = "http://127.0.0.1:8888/mcp/"

[mcp_servers.auth]
type = "static"
api_key_env = "UNSLOTH_API_KEY"
api_key_header = "Authorization"
api_key_format = "Bearer {token}"
```

**DeepSeek Harness**, appended to `~/.dsh/cordis.patch.yml`:

```yaml
- insert:
    - id: mcp-unsloth-studio
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: unsloth-studio
        transport: streamable-http
        url: "http://127.0.0.1:8888/mcp/"
        headers:
          Authorization: !!js '`Bearer ${process.env.UNSLOTH_API_KEY}`'
```

### Tools

| Tool | What it does |
|---|---|
| `studio_status` | What is loaded, loading and running: chat, image, video, speech-to-text and embedding models, training, export and GPUs. |
| `list_models` | Models Unsloth Studio can serve, by kind. With `model`, its training defaults. |
| `load_model` | Load a chat, image, video or speech-to-text model, downloading it first if needed. Reports progress, waits up to `wait_s` (240 s by default) and lists what it unloaded. |
| `unload_model` | Unload a model, or every chat, speech, image, video and speech-to-text model with `kind="all"`. The embedding model stays. |
| `plan_load` | Preview a load without loading anything: GGUF variants and sizes, bytes to download, and the models it would unload. |
| `download_model` | Download a model ahead of time without loading it. Follow it with `get_job`; `load_model` waits for it. |
| `chat` | Ask the loaded model, with optional images. Returns a `cancel_id`. |
| `embed` | Embed up to 2048 texts. |
| `system_one` | Ask the Decision API typed questions. The Decision API must be on. |
| `generate_image` | Text to image, image to image, inpaint, edit, outpaint and upscale. |
| `generate_audio` | Clone, speak, edit, convert, music and separate. |
| `transcribe` | Speech to text, or translation to English. |
| `generate_video` | Start a video. Poll it with `get_job`. |
| `get_job` | The state of a video, recipe, export, training or load job. With `wait_s` it waits for the job to finish or move on. |
| `run_recipe` | Validate, preview or run a Data Recipe. |
| `datasets` | List, check, upload, download and follow training datasets, and upload image datasets for image LoRA training. |
| `start_training` | Start LLM, text-to-speech or image LoRA training. Follow it with `get_job`. |
| `list_training_runs` | Past runs and their checkpoints, by name. |
| `export_model` | Export a checkpoint to GGUF, merged weights, a LoRA adapter or the base model. |
| `cancel` | Stop training, a start request, image training, an export, a recipe, an image or video generation, a chat reply or a dataset download. |

Long work keeps running in Unsloth Studio when a tool call times out.
Follow every long job with `get_job(kind=…, id=…, wait_s=…)`: it checks every 2 s and returns once the job finishes, its status or phase changes, or `wait_s` runs out.
`wait_s` is at most 240; 50 stays under Codex's 60 s default tool timeout, and longer waits need a longer client timeout.
`load_model` waits up to `wait_s` too (240 s when unset), then returns `status: "loading"` and a `job_id` for `get_job(kind="load")`.
Stop work with `cancel`.
Unsloth Studio runs one export at a time, so `export_model` is refused while another export runs, and cancelling an export job stops only that job's own step.

### Media and files

Generated images, audio and videos are saved to Unsloth Studio's galleries, the same as
in the UI. Tools return each item's id and URL. Small items also come back
inline. Large images come back as a preview plus a link. A video comes back as a
thumbnail plus a link, and the MP4 is never sent inline. Links point at
Unsloth Studio's API, so fetch them with the same `Authorization: Bearer` key.

Media inputs can be an Unsloth Studio id, inline data, or a file path. **A file path
works only when the agent runs on the Unsloth Studio computer** and connects over
loopback. From any other computer, send the data or an Unsloth Studio id instead.
An MCP request is limited to 4 MiB, so inline data has to fit in that. Send
larger media as an Unsloth Studio id, or as a path from the Unsloth Studio computer. A remote
agent can upload a large audio file, up to 200 MiB, to `POST /v1/audio/inputs` with its API key
and pass the returned `id` as `input_id`. The route takes the file's raw bytes as
the request body with a `Content-Length`, not a multipart form or JSON, and an
optional `name` query parameter:

```bash
curl -X POST "http://127.0.0.1:8888/v1/audio/inputs?name=meeting.mp3" \
  -H "Authorization: Bearer $UNSLOTH_API_KEY" \
  --data-binary @meeting.mp3
```

`transcribe` sends files up to 25 MB directly. Larger files, and audio given by
id, are uploaded to Unsloth Studio first, and their transcript is saved to Audio
history. Translation needs a file under 25 MB. Transcribing from a URL or a
YouTube link is not supported. `timestamps` needs a speech-to-text model that
times segments, such as Qwen3-ASR (audio.cpp) or Parakeet-TDT; other models
refuse it, and translation never has timestamps.

`studio_status` lists the loaded audio model's `audio_options`, such as its
built-in voices. Pass them to `generate_audio` in `options`. A voice or other
listed value is matched without regard to case, and a value the model does not
list is refused.

Tool results never contain paths on the Unsloth Studio computer. Checkpoints are named
`<run folder>` or `<run folder>/checkpoint-N`. Exports are named relative to
Unsloth Studio's exports folder, and `export_model` takes only a relative
`save_directory`.

### Known limits

- An API key cannot unload a chat model that was loaded from a local folder.
  Unsloth Studio shows such a model to API keys under an opaque reference, and the
  unload route does not resolve it. Unload it in the Unsloth Studio UI.
- Unsloth Studio runs one Data Recipe job at a time, and a recipe job id is not tied to
  an account. Anyone with a key who knows the id can read that job's status.
- Image LoRA training takes a dataset that is already in Unsloth Studio. There is no
  upload tool.
- Models can be unloaded between calls when an idle timeout is set. Tools then
  say to load the model again.
