# Claude Agent in Unsloth Studio

Claude Agent is an **experimental** connection that chats with Claude through the
`claude` CLI (Claude Code) installed and signed in on the **Unsloth Studio backend
machine**. Unsloth Studio runs that CLI through Anthropic's public Claude Agent SDK,
so the CLI's own login is used: a Claude Pro, Max, Team or Enterprise plan, an
Anthropic API key, or a cloud account. Unsloth Studio never reads, stores or sends
that login, and never changes the CLI's requests.

Anthropic's terms for third-party apps apply to how a Claude plan may be used.

## Turn it on

1. Install Claude Code on the backend machine and run `claude` once in a terminal
   to sign in.
2. Open **Settings → System → Developer mode** and turn on **Claude Agent
   (experimental)**. `UNSLOTH_CLAUDE_AGENT=1` or `0` overrides the setting.
3. Open **Settings → Connections → Add connection → Claude Agent**. The card shows
   whether the CLI is installed and signed in. If the Claude Agent SDK is missing,
   choose **Install**; no restart is needed. Pick the models and save.

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
Each reply starts a fresh `claude` process, so the first token takes a few seconds.
