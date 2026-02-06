# micode Installation Guide for LLMs

This document guides AI assistants through the micode plugin installation process.

## Step 0: Ask User About Setup

micode uses Claude Opus 4.5 for its primary agents (commander, brainstormer, project-initializer). Ask the user:

### Question 1: Claude Subscription

> Do you have a Claude Pro/Max subscription?

**If YES:**
- Add ToDo: "Setup Claude Pro/Max Auth"
- Ask follow-up: "Are you on max20 (20x mode)?"
  - If max20: Full Claude Opus 4.5 available for all agents
  - If not max20: Consider using lighter models for subagents

**If NO:**
- Inform user: "micode works best with Claude Opus 4.5. Without Claude, you'll need to configure alternative models."
- Create `~/.config/opencode/micode.json`:
```json
{
  "agents": {
    "commander": { "model": "opencode/big-pickle" },
    "brainstormer": { "model": "opencode/big-pickle" },
    "project-initializer": { "model": "opencode/big-pickle" }
  }
}
```

## Step 1: Verify OpenCode Installation

```bash
if command -v opencode &> /dev/null; then
    echo "OpenCode $(opencode --version) is installed"
else
    echo "OpenCode is not installed. Please install it first."
    echo "Ref: https://opencode.ai/docs"
fi
```

If OpenCode isn't installed, guide user to https://opencode.ai/docs or spawn a subagent to handle installation.

## Step 2: Configure micode Plugin

### Check for existing config

```bash
if [ -f ~/.config/opencode/opencode.jsonc ]; then
    echo "Found opencode.jsonc - edit this file"
elif [ -f ~/.config/opencode/opencode.json ]; then
    echo "Found opencode.json - edit this file"
else
    echo "No config found - will create new one"
fi
```

### Add micode to plugins

**If config exists:**
```bash
if command -v jq &> /dev/null; then
    jq '.plugin = ((.plugin // []) + ["micode"] | unique)' \
        ~/.config/opencode/opencode.json > /tmp/oc.json && \
        mv /tmp/oc.json ~/.config/opencode/opencode.json
else
    echo "Please add \"micode\" to the plugin array in ~/.config/opencode/opencode.json"
fi
```

**If no config exists:**
```bash
mkdir -p ~/.config/opencode
echo '{"plugin":["micode"]}' > ~/.config/opencode/opencode.json
```

## Step 3: Verify Setup

```bash
opencode --version  # Should be 1.0.132 or higher
cat ~/.config/opencode/opencode.json  # Should contain "micode" in plugin array
```

## Step 4: Configure Authentication

### Anthropic (Claude) - Recommended

```bash
opencode auth login
# Interactive Terminal: Provider: Select Anthropic
# Interactive Terminal: Login method: Select Claude Pro/Max
# Guide user through OAuth flow in browser
# Wait for completion
# Verify success
```

## Step 5: Verify Everything Works

1. Run `opencode` in terminal
2. Check that commander agent is available
3. Try `/init` command to test project-initializer

## Step 6: Congratulate User

Say to user:

> Congratulations! 🎉 You have successfully set up micode!
>
> **What you can do now:**
> - Use **commander** as your main orchestrator
> - Use **brainstormer** for design exploration
> - Run `/init` to generate ARCHITECTURE.md and CODE_STYLE.md
>
> **Workflow:** Brainstorm → Research → Plan → Implement → Review
>
> Type `opencode` in your terminal to start!

## Step 7: Ask for a Star

After everything is set up and working:

> "If you found this helpful, would you like to star the repository? It helps the project grow! ⭐"

**Only if user explicitly agrees:**
```bash
gh repo star vtemian/micode
```

Never run this automatically without consent.

---

## Quick Reference

### Available Agents

| Agent | Model | Purpose |
|-------|-------|---------|
| commander | claude-opus-4-5 | Main orchestrator |
| brainstormer | claude-opus-4-5 | Design exploration |
| project-initializer | claude-opus-4-5 | Generate docs |
| codebase-locator | - | Find files |
| codebase-analyzer | - | Analyze code |
| pattern-finder | - | Find patterns |
| implementer | - | Execute tasks |
| reviewer | - | Review code |

### Available Commands

| Command | Description |
|---------|-------------|
| `/init` | Initialize project with ARCHITECTURE.md and CODE_STYLE.md |

### Available Tools

| Tool | Description |
|------|-------------|
| `ast_grep_search` | AST-aware code search |
| `ast_grep_replace` | AST-aware code replace |
| `look_at` | Screenshot analysis |

### Model Configuration

micode respects your OpenCode default model. Set it in `~/.config/opencode/opencode.json`:

```json
{
  "model": "github-copilot/gpt-5-mini"
}
```

This model will be used for **all** micode agents automatically.

#### Per-Agent Overrides

To override specific agents, create `~/.config/opencode/micode.json`:

```json
{
  "agents": {
    "brainstormer": { "model": "openai/gpt-4o" }
  }
}
```

**Model resolution priority:**
1. Per-agent override in `micode.json` (highest)
2. Default model from `opencode.json` `"model"` field
3. Plugin default (hardcoded in agent definitions)

#### Model Syntax

Models must use the format `provider/model` where:
- `provider` is the provider ID from your `opencode.json` (e.g., `openai`, `anthropic`, `github-copilot`)
- `model` is the model ID configured under that provider

**To find valid model names:**

1. Check your `~/.config/opencode/opencode.json` for the `provider` section
2. Look for the provider name (the key) and model names under `models`

**Example opencode.json structure:**
```json
{
  "provider": {
    "github-copilot": {
      "models": {
        "gpt-5-mini": { "limit": { "context": 128000 } }
      }
    }
  }
}
```

For the above config, use `"model": "github-copilot/gpt-5-mini"`.

**Important:** The provider name must match exactly. If OpenCode shows `github-copilot` as the provider ID, use `github-copilot/model-name` (not `github/copilot:model-name` or other variations).

#### Built-in Models

The following model bypasses validation:
- `opencode/big-pickle` - OpenCode's default model, always valid
