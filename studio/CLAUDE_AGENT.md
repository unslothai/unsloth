# Claude Agent in Unsloth Studio

Claude Agent is an **experimental** connection that chats with Claude through the
`claude` CLI (Claude Code) installed and signed in on the **Unsloth Studio backend
machine**. Unsloth Studio runs that CLI through Anthropic's public Claude Agent SDK,
so the CLI's own login is used: a Claude Pro, Max, Team or Enterprise plan, an
Anthropic API key, or a cloud account. Unsloth Studio never reads, stores or sends
that login, and never changes the CLI's requests.

Anthropic's terms for third-party apps apply to how a Claude plan may be used.

## Turn it on

Open **Settings → Connections → Add connection** and choose **Claude Agent**
(marked Experimental).

1. If it is off, choose **Turn on**. This is the same switch as **Settings → System →
   Developer mode → Claude Agent (experimental)**, which is also where you turn it
   off. `UNSLOTH_CLAUDE_AGENT=1` or `0` overrides the setting.
2. If Claude Code is not installed or not signed in on the backend machine, the view
   shows the install command and `claude` to sign in, each with a copy button. It
   checks again when you come back to the window.
3. Once it is ready, the main models (Opus, Fable, Sonnet, Haiku) are ticked. Choose
   **Connect**. Older models are under **More models**, and extra model IDs and Max
   Tokens under **Advanced**.

Unsloth Studio installs the Claude Agent SDK itself, always from its source package:
its wheels bundle a second copy of Claude Code that Unsloth Studio never runs. On an
install that predates this, the card offers **Install**; no restart is needed.

Only the installation owner can use it, from the Studio UI or the owner's own
scheduled tasks. It cannot be shared, is hidden from managed accounts, and API-key
clients are refused. Turning the setting off hides saved Claude Agent connections
until it is turned back on.

## What works

- Streaming replies with thinking, effort levels (each model's supported levels;
  higher levels are lowered to the closest one the model accepts) and Max Tokens.
- Studio tools, MCP, project knowledge and skills. Tools run through Studio's own
  approvals and sandbox; Claude Code's built-in tools stay off.
- Images in the newest message. Images in earlier messages are shown to Claude as
  `[image]`.
- Long chats: Studio trims older messages to each model's context window, as
  reported by the CLI.

Temperature, Top P and Top K are not used. Chat titles are made locally, without a
model call. Read aloud, dictation and image generation cannot use this connection.
Each reply starts a fresh `claude` process, so the first token takes a few seconds;
the reply shows "Starting Claude..." until then.
