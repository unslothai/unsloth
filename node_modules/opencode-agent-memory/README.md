# opencode-agent-memory

[Letta](https://letta.com)-style editable [memory blocks](https://docs.letta.com/guides/agents/memory-blocks/) for [OpenCode](https://opencode.ai).

## Experimental

This plugin is experimental. The core idea - giving the agent persistent, self-editable memory blocks - is adapted from [Letta](https://github.com/letta-ai/letta). Specifially, the plugin follows Letta's [shared memory blocks](https://docs.letta.com/tutorials/shared-memory-blocks) pattern - the markdown files on disk are shared state that every OpenCode session can read and write.

Think of it as AGENTS.md with a harness. OpenCode supports [rules](https://opencode.ai/docs/rules/) via `AGENTS.md` and custom instruction files - this plugin is similar in spirit, but adds structure (scoped blocks with metadata and size limits), dedicated tools for memory operations, and prompting that encourages the agent to actively maintain its own memory. The content is similar; the scaffolding around it is what's different.

For background on the memory concept, see Letta's docs on [memory](https://docs.letta.com/guides/agents/memory/) and [memory blocks](https://docs.letta.com/guides/agents/memory-blocks/).

## Features

- **Persistent memory** - Information survives across sessions and context compaction
- **Shared across sessions** - Global blocks shared across all projects, project blocks shared across sessions in that codebase
- **Self-editing** - The agent can read and modify its own memory with dedicated tools
- **System prompt injection** - Memory blocks appear in the system prompt, always in-context

## Requirements

- [OpenCode](https://opencode.ai/) v1.0.115 or later

## Installation

Add to your OpenCode config (`~/.config/opencode/config.json`):

```json
{
  "plugin": ["opencode-agent-memory@0.1.0"]
}
```

Restart OpenCode and you're ready to go.

Pin to a specific version to ensure updates work correctly - OpenCode's lockfile won't re-resolve unpinned versions. To upgrade, change the version and restart.

### Local Development

If you want to customize or contribute:

```bash
git clone https://github.com/joshuadavidthomas/opencode-agent-memory ~/.config/opencode/opencode-agent-memory
mkdir -p ~/.config/opencode/plugin
ln -sf ~/.config/opencode/opencode-agent-memory/src/plugin.ts ~/.config/opencode/plugin/memory.ts
```

## Usage

The plugin gives the agent 3 tools for managing memory:

| Tool | Description |
|------|-------------|
| `memory_list` | List available memory blocks (labels, descriptions, sizes) |
| `memory_set` | Create or update a memory block (full overwrite) |
| `memory_replace` | Replace a substring within a memory block |

You interact with memory by editing the markdown files directly or asking the agent to update its memory.

### Default Blocks

Three blocks are seeded on first run:

| Block | Scope | Purpose |
|-------|-------|---------|
| `persona` | global | How the agent should behave and respond |
| `human` | global | Details about you (preferences, habits, constraints) |
| `project` | project | Codebase-specific knowledge (commands, architecture, conventions) |

These are just starting points. Create whatever blocks make sense for your workflow - `debugging-notes`, `api-preferences`, `learned-patterns`, etc.

### Memory Locations

- **Global blocks**: `~/.config/opencode/memory/*.md`
- **Project blocks**: `.opencode/memory/*.md` (auto-gitignored)

### Block Format

Each block is a markdown file with YAML frontmatter:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `label` | string | filename | Unique identifier for the block |
| `description` | string | generic | Tells the agent how to use this block |
| `limit` | integer | 5000 | Maximum characters allowed |
| `read_only` | boolean | false | Prevent agent from modifying |

All fields have defaults for graceful degradation, but `description` is essential - without it, the agent gets a generic fallback and won't know how to use the block effectively. See Letta's docs on [the importance of the description field](https://docs.letta.com/guides/agents/memory-blocks/#the-importance-of-the-description-field).

## Inspiration

The memory architecture and philosophical framing are adapted from [Letta](https://github.com/letta-ai/letta) (formerly MemGPT), a framework for building LLM agents with editable long-term memory.

Also worth exploring: [private-journal-mcp](https://github.com/obra/private-journal-mcp) by Jesse Vincent, which gives Claude a private journaling capability to process feelings and thoughts. His [blog post](https://blog.fsck.com/2025/05/28/dear-diary-the-user-asked-me-if-im-alive/) about it explores similar territory around AI self-reflection and persistent inner experience.

## Contributing

Contributions are welcome! Here's how to set up for development:

```bash
git clone https://github.com/joshuadavidthomas/opencode-agent-memory
cd opencode-agent-memory
bun install
```

Then symlink the plugin to your OpenCode config:

```bash
mkdir -p ~/.config/opencode/plugin
ln -sf "$(pwd)/src/plugin.ts" ~/.config/opencode/plugin/memory.ts
```

## License

opencode-agent-memory is licensed under the MIT license. See the [`LICENSE`](LICENSE) file for more information.

---

opencode-agent-memory is not built by, or affiliated with, the OpenCode team.

OpenCode is ©2025 Anomaly.
