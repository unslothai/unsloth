# micode

[![CI](https://github.com/vtemian/micode/actions/workflows/ci.yml/badge.svg)](https://github.com/vtemian/micode/actions/workflows/ci.yml)
[![npm version](https://badge.fury.io/js/micode.svg)](https://www.npmjs.com/package/micode)

OpenCode plugin with structured Brainstorm → Plan → Implement workflow and session continuity.

https://github.com/user-attachments/assets/85236ad3-e78a-4ff7-a840-620f6ea2f512

## Quick Start

Add to `~/.config/opencode/opencode.json`:

```json
{ "plugin": ["micode"] }
```

Then run `/init` to generate `ARCHITECTURE.md` and `CODE_STYLE.md`.

## Workflow

```
Brainstorm → Plan → Implement
     ↓         ↓        ↓
  research  research  executor
```

### Brainstorm
Refine ideas into designs through collaborative questioning. Fires research subagents in parallel. Output: `thoughts/shared/designs/YYYY-MM-DD-{topic}-design.md`

### Plan  
Transform designs into implementation plans with bite-sized tasks (2-5 min each), exact file paths, and TDD workflow. Output: `thoughts/shared/plans/YYYY-MM-DD-{topic}.md`

### Implement
Execute in git worktree for isolation. The **Executor** orchestrates implementer→reviewer cycles with parallel execution via fire-and-check pattern.

### Session Continuity
Maintain context across sessions with structured compaction. Run `/ledger` to create/update `thoughts/ledgers/CONTINUITY_{session}.md`.

## Commands

| Command | Description |
|---------|-------------|
| `/init` | Initialize project docs |
| `/ledger` | Create/update continuity ledger |
| `/search` | Search past plans and ledgers |

## Agents

| Agent | Purpose |
|-------|---------|
| commander | Orchestrator |
| brainstormer | Design exploration |
| planner | Implementation plans |
| executor | Orchestrate implement→review |
| implementer | Execute tasks |
| reviewer | Check correctness |
| codebase-locator | Find file locations |
| codebase-analyzer | Deep code analysis |
| pattern-finder | Find existing patterns |
| project-initializer | Generate project docs |
| ledger-creator | Continuity ledgers |
| artifact-searcher | Search past work |

## Tools

| Tool | Description |
|------|-------------|
| `ast_grep_search` | AST-aware code pattern search |
| `ast_grep_replace` | AST-aware code pattern replacement |
| `look_at` | Extract file structure |
| `artifact_search` | Search past plans/ledgers |
| `btca_ask` | Query library source code |
| `pty_spawn` | Start background terminal session |
| `pty_write` | Send input to PTY |
| `pty_read` | Read PTY output |
| `pty_list` | List PTY sessions |
| `pty_kill` | Terminate PTY |

## Hooks

- **Think Mode** - Keywords like "think hard" enable 32k token thinking budget
- **Ledger Loader** - Injects continuity ledger into system prompt
- **Auto-Compact** - At 50% context usage, automatically summarizes session to reduce context
- **File Ops Tracker** - Tracks read/write/edit for deterministic logging
- **Artifact Auto-Index** - Indexes artifacts in thoughts/ directories
- **Context Injector** - Injects ARCHITECTURE.md, CODE_STYLE.md
- **Token-Aware Truncation** - Truncates large tool outputs

## Configuration

### Model Configuration

micode reads your default model from `opencode.json`:

```json
{
  "model": "github-copilot/gpt-5-mini",
  "plugin": ["micode"]
}
```

All micode agents will use this model automatically.

### micode.json

Create `~/.config/opencode/micode.json` for micode-specific settings:

```json
{
  "agents": {
    "brainstormer": { "model": "openai/gpt-4o", "temperature": 0.8 },
    "commander": { "maxTokens": 8192 }
  },
  "features": {
    "mindmodelInjection": true
  },
  "compactionThreshold": 0.5,
  "fragments": {
    "commander": ["custom-instructions.md"]
  }
}
```

#### Options

| Option | Type | Description |
|--------|------|-------------|
| `agents` | object | Per-agent overrides (model, temperature, maxTokens) |
| `features.mindmodelInjection` | boolean | Enable mindmodel context injection |
| `compactionThreshold` | number | Context usage threshold (0-1) for auto-compaction. Default: 0.5 |
| `fragments` | object | Additional prompt fragments per agent |

#### Model Resolution Priority

1. Per-agent override in `micode.json` (highest)
2. Default model from `opencode.json` `"model"` field
3. Plugin default (fallback)

#### Model Syntax

Models use `provider/model` format. The provider must match exactly what's in your `opencode.json`:

```json
{
  "provider": {
    "github-copilot": {
      "models": { "gpt-5-mini": {} }
    }
  }
}
```

Use `"model": "github-copilot/gpt-5-mini"` (not `github/copilot:gpt-5-mini`).

## Development

```bash
git clone git@github.com:vtemian/micode.git ~/.micode
cd ~/.micode && bun install && bun run build
```

```json
// Use local path
{ "plugin": ["~/.micode"] }
```

### Release

```bash
npm version patch  # or minor, major
git push --follow-tags
```

## Philosophy

1. **Brainstorm first** - Refine ideas before coding
2. **Research before implementing** - Understand the codebase
3. **Plan with human buy-in** - Get approval before coding
4. **Parallel investigation** - Spawn multiple subagents
5. **Isolated implementation** - Use git worktrees
6. **Continuous verification** - Implementer + Reviewer per task
7. **Session continuity** - Never lose context

## Inspiration

- [oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode) - Plugin architecture
- [HumanLayer ACE-FCA](https://github.com/humanlayer/12-factor-agents) - Structured workflows
- [Factory.ai](https://factory.ai/blog/context-compression) - Structured compaction research
