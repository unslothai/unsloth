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
listed in full whenever the catalog fits the loaded local model's context window,
so a model that can hold the full listing always gets it.

When the full listing would take more than three quarters of the window, which
would otherwise get even a short prompt refused, the largest tools (only those
whose description and schema together exceed about 1,500 characters) are listed
in a compact form, largest first, until the listing fits: a compact tool shows its
first sentence plus its top-level parameters with their types, required flags and
short enums. Every other tool keeps its full schema. The model then also gets `mcp_tool_schema`, which returns a tool's
full description and JSON Schema on demand, in pages when it is longer than the
room left for a tool result. A compact tool called without one of its required
arguments, or whose call the server rejects, answers with that schema so the model
can correct the call. Arguments to a compact tool are still typed against its full
schema. External providers always get the full listing.

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

Studio has its own MCP server at `/mcp/`. Coding agents such as Claude Code and
Codex can use it to load models, chat, generate images, audio and video,
transcribe, train and export. Every tool goes through Studio's own routes as
your API key, so it sees what that key may see and nothing more.

### Turn it on

The server is off by default. While it is off, `/mcp/` answers 404.

- In Studio, open **Settings → API** and turn on **Agent access (MCP)**. Only
  the owner can change it, and only from a signed-in Studio session.
- Or start Studio with `UNSLOTH_STUDIO_ENABLE_MCP=1`. The switch then shows as
  on and cannot be turned off in Settings.

The endpoint is `http://127.0.0.1:8888/mcp/` on the default port. Always use
the trailing slash. Use your own address and port when they differ, for example
the LAN address or the Cloudflare tunnel URL.

### Authentication

Every request needs a Studio API key: `Authorization: Bearer sk-unsloth-…`.
Create one in **Settings → API**. Studio refuses everything else with 401:

- no key, an empty key, or a placeholder key such as `not-needed`, even when
  keyless API access is on;
- a signed-in session token;
- a revoked or expired key;
- the keys Studio mints for its own recipe and Deep Research workflows.

A managed account's key works too. Its tools act for that account only.

Studio does not offer OAuth. The OAuth discovery paths answer 404, so clients
that probe them fall back to the key.

A browser page from another site gets 403 `Origin not allowed for Studio MCP`.
This also blocks browser-based MCP inspectors. Command-line agents send no
Origin and are not affected.

### Breaking changes

The static `UNSLOTH_STUDIO_MCP_TOKEN` is retired. A request that sends it gets
401 with this detail:

```
The MCP static token is no longer supported; use a Studio API key (sk-unsloth-…)
```

If the variable is still set, Studio logs one warning at startup. Studio no
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

Studio's **Settings → API** page builds these for you, with your real address.
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
| `list_models` | Models Studio can serve, by kind. With `model`, its training defaults. |
| `load_model` | Load a chat, image, video or speech-to-text model, downloading it first if needed. Reports progress. |
| `unload_model` | Unload a model. |
| `chat` | Ask the loaded model, with optional images. Returns a `cancel_id`. |
| `embed` | Embed up to 2048 texts. |
| `system_one` | Ask the Decision API typed questions. The Decision API must be on. |
| `generate_image` | Text to image, image to image, inpaint, edit, outpaint and upscale. |
| `generate_audio` | Clone, speak, edit, convert, music and separate. |
| `transcribe` | Speech to text, or translation to English. |
| `generate_video` | Start a video. Poll it with `get_job`. |
| `get_job` | The state of a video, recipe or export job. |
| `run_recipe` | Validate, preview or run a Data Recipe. |
| `datasets` | List, check, download and follow training datasets. |
| `start_training` | Start LLM or image LoRA training. Follow it with `studio_status`. |
| `list_training_runs` | Past runs and their checkpoints, by name. |
| `export_model` | Export a checkpoint to GGUF, merged weights, a LoRA adapter or the base model. |
| `cancel` | Stop training, a start request, image training, an export, a recipe, an image or video generation, a chat reply or a dataset download. |

Long work returns at once. Start it, then poll `get_job` or `studio_status`,
and stop it with `cancel`. Audio runs cannot be cancelled.

### Media and files

Generated images, audio and videos are saved to Studio's galleries, the same as
in the UI. Tools return each item's id and URL. Small items also come back
inline. Large images come back as a preview plus a link. A video comes back as a
thumbnail plus a link, and the MP4 is never sent inline.

Media inputs can be a Studio id, inline data, or a file path. **A file path
works only when the agent runs on the Studio computer** and connects over
loopback. From any other computer, send the data or a Studio id instead.

`transcribe` sends files up to 25 MB directly. Larger files, and audio given by
id, are uploaded to Studio first, and their transcript is saved to Audio
history. Translation needs a file under 25 MB. Transcribing from a URL or a
YouTube link is not supported.

Tool results never contain paths on the Studio computer. Checkpoints are named
`<run folder>` or `<run folder>/checkpoint-N`. Exports are named relative to
Studio's exports folder, and `export_model` takes only a relative
`save_directory`.

### Known limits

- An API key cannot unload a chat model that was loaded from a local folder.
  Studio shows such a model to API keys under an opaque reference, and the
  unload route does not resolve it. Unload it in the Studio UI.
- Studio runs one Data Recipe job at a time, and a recipe job id is not tied to
  an account. Anyone with a key who knows the id can read that job's status.
- Image LoRA training takes a dataset that is already in Studio. There is no
  upload tool.
- Models can be unloaded between calls when an idle timeout is set. Tools then
  say to load the model again.
