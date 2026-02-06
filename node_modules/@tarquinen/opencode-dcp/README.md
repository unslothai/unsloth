# Dynamic Context Pruning Plugin

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/dansmolsky)
[![npm version](https://img.shields.io/npm/v/@tarquinen/opencode-dcp.svg)](https://www.npmjs.com/package/@tarquinen/opencode-dcp)

Automatically reduces token usage in OpenCode by removing obsolete content from conversation history.

![DCP in action](assets/images/dcp-demo5.png)

## Installation

Add to your OpenCode config:

```jsonc
// opencode.jsonc
{
    "plugin": ["@tarquinen/opencode-dcp@latest"],
}
```

Using `@latest` ensures you always get the newest version automatically when OpenCode starts.

Restart OpenCode. The plugin will automatically start optimizing your sessions.

## How Pruning Works

DCP uses multiple tools and strategies to reduce context size:

### Tools

**Distill** — Exposes a `distill` tool that the AI can call to distill valuable context into concise summaries before removing the tool content.

**Compress** — Exposes a `compress` tool that the AI can call to collapse a large section of conversation (messages and tools) into a single summary.

**Prune** — Exposes a `prune` tool that the AI can call to remove completed or noisy tool content from context.

### Strategies

**Deduplication** — Identifies repeated tool calls (e.g., reading the same file multiple times) and keeps only the most recent output. Runs automatically on every request with zero LLM cost.

**Supersede Writes** — Removes write tool calls for files that have subsequently been read. When a file is written and later read, the original write content becomes redundant since the current file state is captured in the read result. Runs automatically on every request with zero LLM cost.

**Purge Errors** — Prunes tool inputs for tools that returned errors after a configurable number of turns (default: 4). Error messages are preserved for context, but the potentially large input content is removed. Runs automatically on every request with zero LLM cost.

Your session history is never modified—DCP replaces pruned content with placeholders before sending requests to your LLM.

## Impact on Prompt Caching

LLM providers like Anthropic and OpenAI cache prompts based on exact prefix matching. When DCP prunes a tool output, it changes the message content, which invalidates cached prefixes from that point forward.

**Trade-off:** You lose some cache read benefits but gain larger token savings from reduced context size and performance improvements through reduced context poisoning. In most cases, the token savings outweigh the cache miss cost—especially in long sessions where context bloat becomes significant.

> **Note:** In testing, cache hit rates were approximately 80% with DCP enabled vs 85% without for most providers.

**Best use case:** Providers that count usage in requests, such as Github Copilot and Google Antigravity, have no negative price impact.

**Best use cases:**

- **Request-based billing** — Providers that count usage in requests, such as Github Copilot and Google Antigravity, have no negative price impact.
- **Uniform token pricing** — Providers that bill cached tokens at the same rate as regular input tokens, such as Cerebras, see pure savings with no cache-miss penalty.

**Claude Subscriptions:** Anthropic subscription users (who receive "free" caching) may experience faster limit depletion than hit-rate ratios suggest due to the higher relative cost of cache misses. See [Claude Cache Limits](https://she-llac.com/claude-limits) for details.

## Configuration

DCP uses its own config file:

- Global: `~/.config/opencode/dcp.jsonc` (or `dcp.json`), created automatically on first run
- Custom config directory: `$OPENCODE_CONFIG_DIR/dcp.jsonc` (or `dcp.json`), if `OPENCODE_CONFIG_DIR` is set
- Project: `.opencode/dcp.jsonc` (or `dcp.json`) in your project's `.opencode` directory

> <details>
> <summary><strong>Default Configuration</strong> (click to expand)</summary>
>
> ```jsonc
> {
>     "$schema": "https://raw.githubusercontent.com/Opencode-DCP/opencode-dynamic-context-pruning/master/dcp.schema.json",
>     // Enable or disable the plugin
>     "enabled": true,
>     // Enable debug logging to ~/.config/opencode/logs/dcp/
>     "debug": false,
>     // Notification display: "off", "minimal", or "detailed"
>     "pruneNotification": "detailed",
>     // Notification type: "chat" (in-conversation) or "toast" (system toast)
>     "pruneNotificationType": "chat",
>     // Slash commands configuration
>     "commands": {
>         "enabled": true,
>         // Additional tools to protect from pruning via commands (e.g., /dcp sweep)
>         "protectedTools": [],
>     },
>     // Protect from pruning for <turns> message turns past tool invocation
>     "turnProtection": {
>         "enabled": false,
>         "turns": 4,
>     },
>     // Protect file operations from pruning via glob patterns
>     // Patterns match tool parameters.filePath (e.g. read/write/edit)
>     "protectedFilePatterns": [],
>     // LLM-driven context pruning tools
>     "tools": {
>         // Shared settings for all prune tools
>         "settings": {
>             // Nudge the LLM to use prune tools (every <nudgeFrequency> tool results)
>             "nudgeEnabled": true,
>             "nudgeFrequency": 10,
>             // Token limit at which the model begins actively
>             // compressing session context. Best kept around 40% of
>             // the model's context window to stay in the "smart zone".
>             // Set to "model" to use the model's full context window.
>             "contextLimit": 100000,
>             // Additional tools to protect from pruning
>             "protectedTools": [],
>         },
>         // Distills key findings into preserved knowledge before removing raw content
>         "distill": {
>             // Permission mode: "allow" (no prompt), "ask" (prompt), "deny" (tool not registered)
>             "permission": "allow",
>             // Show distillation content as an ignored message notification
>             "showDistillation": false,
>         },
>         // Collapses a range of conversation content into a single summary
>         "compress": {
>             // Permission mode: "ask" (prompt), "allow" (no prompt), "deny" (tool not registered)
>             "permission": "ask",
>             // Show summary content as an ignored message notification
>             "showCompression": false,
>         },
>         // Removes tool content from context without preservation (for completed tasks or noise)
>         "prune": {
>             // Permission mode: "allow" (no prompt), "ask" (prompt), "deny" (tool not registered)
>             "permission": "allow",
>         },
>     },
>     // Automatic pruning strategies
>     "strategies": {
>         // Remove duplicate tool calls (same tool with same arguments)
>         "deduplication": {
>             "enabled": true,
>             // Additional tools to protect from pruning
>             "protectedTools": [],
>         },
>         // Prune write tool inputs when the file has been subsequently read
>         "supersedeWrites": {
>             "enabled": true,
>         },
>         // Prune tool inputs for errored tools after X turns
>         "purgeErrors": {
>             "enabled": true,
>             // Number of turns before errored tool inputs are pruned
>             "turns": 4,
>             // Additional tools to protect from pruning
>             "protectedTools": [],
>         },
>     },
> }
> ```
>
> </details>

### Commands

DCP provides a `/dcp` slash command:

- `/dcp` — Shows available DCP commands
- `/dcp context` — Shows a breakdown of your current session's token usage by category (system, user, assistant, tools, etc.) and how much has been saved through pruning.
- `/dcp stats` — Shows cumulative pruning statistics across all sessions.
- `/dcp sweep` — Prunes all tools since the last user message. Accepts an optional count: `/dcp sweep 10` prunes the last 10 tools. Respects `commands.protectedTools`.

### Protected Tools

By default, these tools are always protected from pruning:
`task`, `todowrite`, `todoread`, `distill`, `compress`, `prune`, `batch`, `plan_enter`, `plan_exit`

The `protectedTools` arrays in each section add to this default list.

### Config Precedence

Settings are merged in order:
Defaults → Global (`~/.config/opencode/dcp.jsonc`) → Config Dir (`$OPENCODE_CONFIG_DIR/dcp.jsonc`) → Project (`.opencode/dcp.jsonc`).
Each level overrides the previous, so project settings take priority over config-dir and global, which take priority over defaults.

Restart OpenCode after making config changes.

## Limitations

**Subagents** — DCP is disabled for subagents. Subagents are not designed to be token efficient; what matters is that the final message returned to the main agent is a concise summary of findings. DCP's pruning could interfere with this summarization behavior.

## License

AGPL-3.0-or-later
