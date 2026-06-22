#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Drive one coding agent against the running `unsloth run` server for the
# Local Agent Guides CI. All failures from here are failure class (c)
# "guide drift": the server preflight already passed and the agent CLI
# already installed, so a failure here means the documented recipe in
# unsloth_cli/commands/connect.py no longer produces a working flow.
#
# Self-updating: for the 5 agents with a connect.py recipe we obtain the
# exact env + command from `unsloth connect <agent> --no-launch` and run
# THAT, so a recipe change is exercised automatically. Pi (no connect.py
# command at HEAD) is driven by a hand-written recipe.
#
# Every agent invocation is wrapped in `timeout` so a headless-TTY prompt
# can never hang the runner -- a timeout is reported as guide drift with a
# distinct message.
#
# Usage:
#   agent-guides-drive.sh connection     <agent>
#   agent-guides-drive.sh file-edit      <agent>
#   agent-guides-drive.sh attribution-ab claude
#
# Required env (exported by serve-unsloth-run.sh):
#   UNSLOTH_BASE_URL UNSLOTH_API_KEY UNSLOTH_MODEL_ID
#   UNSLOTH_LLAMA_LOG_DIR  AGENT_INVOKE_TIMEOUT  UNSLOTH_SEED
set -uo pipefail

MODE="${1:?usage: agent-guides-drive.sh <mode> <agent>}"
AGENT="${2:?usage: agent-guides-drive.sh <mode> <agent>}"

: "${UNSLOTH_BASE_URL:?serve step did not export UNSLOTH_BASE_URL}"
: "${UNSLOTH_API_KEY:?serve step did not export UNSLOTH_API_KEY}"
: "${UNSLOTH_MODEL_ID:?serve step did not export UNSLOTH_MODEL_ID}"
# Determinism (seed/temp) is applied at the server level by
# serve-unsloth-run.sh --extra; agents inherit it through the API.
TIMEOUT="${AGENT_INVOKE_TIMEOUT:-180}"

# Claude refuses --dangerously-skip-permissions outside a sandbox; the CI runner
# IS the sandbox, so declare it (mirrors unslothai/scripts launcher.sh). Harmless
# to the other agents, which ignore it.
export IS_SANDBOX=1

# Absolute paths anchored at the repo root (this script lives in
# .github/scripts/). Everything writes here regardless of the current working
# directory, so the file-edit mode can `cd` into a scratch work dir without
# breaking log/redaction writes.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOGS_DIR="$REPO_ROOT/logs"
REDACTED_DIR="$REPO_ROOT/redacted-configs"
WORKDIR_BASE="$REPO_ROOT/agent-workdir"
CACHE_HELPER="$SCRIPT_DIR/assert-prompt-cache.sh"
mkdir -p "$LOGS_DIR" "$REDACTED_DIR"
CONNECT_REF="unsloth_cli/commands/connect.py"

# Prefill-shrinking flags for Claude Code. The heavyweight agents send
# multi-thousand-token system prompts + full tool schemas, which on a CPU-only
# runner is minutes of prefill per model round-trip (~16 tok/s for a 4B model).
# Replacing the ~5.7k default system prompt with a tiny one (--system-prompt-file)
# and restricting tools cuts the prefill to a few hundred tokens so it completes
# quickly on CPU. These only shape the request size; the connect.py recipe
# (endpoint, auth, model) is still exercised end to end.
#
# The bulk of Claude Code's prompt is the built-in tool JSON schemas: measured
# via `claude -p /context`, the default prompt is ~28k tokens of which ~18k is
# "System tools" alone. --allowedTools/--disallowedTools only gate PERMISSION to
# call a tool; they do NOT remove its schema from what is sent to the model, so
# the earlier whitelist left the full ~18k in the prompt and CPU prefill
# (~16 tok/s) overran claude's own request timeout into a retry loop. --tools is
# the flag that restricts which schemas are sent. (The ~8k "Memory files" chunk
# is auto-loaded CLAUDE.md; the unsloth repo ships none, so it is 0 in CI.)
#
# Connection probe: --tools "" sends ZERO tool schemas, leaving ~20 tokens total
# (a one-line --system-prompt-file + the user turn), which prefills instantly.
CLAUDE_CONNECT_FLAGS=(
  --system-prompt-file "$SCRIPT_DIR/ci-connect-prompt.txt"
  --tools ""
)
# File-edit: the task needs the file/shell tools, so send only those schemas
# (~2.3k tokens vs ~18k for the full set).
CLAUDE_EDIT_FLAGS=(
  --system-prompt-file "$SCRIPT_DIR/ci-min-system-prompt.txt"
  --tools "Bash,Edit,Write,Read"
)

guide_fail() {
  echo "::error::[guide drift] agent=${AGENT}: $* (preflight passed + install OK, so the documented flow in ${CONNECT_REF} drifted)." >&2
  exit 1
}

# Redact the API key from any file we are about to keep as an artifact.
# Portable across GNU sed (Linux runners) and BSD sed (macOS), so the
# redaction is never silently skipped.
redact() {
  local f
  for f in "$@"; do
    [ -f "$f" ] || continue
    if sed --version >/dev/null 2>&1; then
      sed -i "s#${UNSLOTH_API_KEY}#<REDACTED>#g" "$f" 2>/dev/null || true
    else
      sed -i '' "s#${UNSLOTH_API_KEY}#<REDACTED>#g" "$f" 2>/dev/null || true
    fi
  done
}

# A reply must be non-empty and free of connection/auth errors.
assert_reply() {
  local out="$1"
  if [ ! -s "$out" ]; then
    guide_fail "agent produced an EMPTY reply"
  fi
  if grep -qiE 'connection refused|connection error|econnrefused|fetch failed|http 4[0-9][0-9]|unauthorized|invalid api key|authentication failed' "$out"; then
    guide_fail "agent reply contained a connection/auth error: $(grep -iE 'connection|unauthorized|auth|http 4' "$out" | head -1)"
  fi
  echo "[$AGENT] reply (first 20 lines):"
  head -20 "$out"
}

# Run a command under a hard timeout; map 124 to a guide-drift hang message.
run_timed() {  # $1=outfile, rest=command
  local out="$1"; shift
  timeout "$TIMEOUT" "$@" > "$out" 2>&1
  local rc=$?
  if [ "$rc" -eq 124 ]; then
    redact "$out"  # guide_fail exits below, so scrub the transcript here too
    echo "[$AGENT] last 40 lines before timeout:"; tail -40 "$out" 2>/dev/null || true
    guide_fail "invoke timed out after ${TIMEOUT}s (headless-TTY hang -- the recipe likely needs a non-interactive/print flag)"
  fi
  return "$rc"
}

# ── Pi: no connect.py command at HEAD -> hand-written recipe ──────────────
write_pi_config() {
  if unsloth connect pi --help >/dev/null 2>&1; then
    # Tripwire: once a real recipe exists, the hand-written config would mask any
    # drift in it, defeating the point of this CI. Fail hard so the cell is
    # migrated to the self-updating `unsloth connect pi --no-launch` path.
    guide_fail "connect.py now ships a 'pi' command -- migrate this CI cell to the 'unsloth connect pi --no-launch' path so the documented recipe is exercised (the hand-written Pi config no longer reflects it)"
  fi
  mkdir -p "$HOME/.pi/agent"
  python3 - "$UNSLOTH_BASE_URL" "$UNSLOTH_API_KEY" "$UNSLOTH_MODEL_ID" <<'PY'
import json, os, sys
base, key, model = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = {"providers": {"unsloth": {
    "api": "openai-completions",
    "baseUrl": f"{base}/v1",
    "apiKey": key,
    "models": [{"id": model}],
}}}
path = os.path.expanduser("~/.pi/agent/models.json")
with open(path, "w") as fh:
    json.dump(cfg, fh, indent=2)
PY
  cp "$HOME/.pi/agent/models.json" "$REDACTED_DIR/pi-models.json" 2>/dev/null || true
  redact "$REDACTED_DIR/pi-models.json"
}

# ── 5-agent connect.py path: parse env + command from --no-launch ─────────
# Populates globals CONNECT_ENV (export/unset lines) and CONNECT_CMD (the
# launch command on the last printed line), and runs connect.py's config
# writers as a side effect (it writes ~/.codex, ~/.claude, etc.).
parse_connect() {
  local raw="$LOGS_DIR/connect-${AGENT}.txt"
  if ! unsloth connect "$AGENT" --no-launch --api-key "$UNSLOTH_API_KEY" > "$raw" 2>&1; then
    cat "$raw"
    guide_fail "'unsloth connect ${AGENT} --no-launch' exited non-zero"
  fi
  echo "[$AGENT] connect --no-launch printed:"; cat "$raw"
  CONNECT_ENV="$(grep -E '^(export |unset )' "$raw" || true)"
  # The launch command is the last non-export, non-status line. connect.py
  # prints "Studio <url> · model <id>" and "Updated ..." status lines first.
  CONNECT_CMD="$(grep -vE '^(export |unset |Studio |Updated |Disabled |Warning|Loading)' "$raw" \
    | grep -E '[^[:space:]]' | tail -1)"
  [ -n "$CONNECT_CMD" ] || guide_fail "could not parse a launch command from connect --no-launch output"
  redact "$raw"
}

# Cross-check the documented contract knobs so silent connect.py changes
# (env-var rename, wire_api flip, attribution setting drop) also fail/flag.
crosscheck_contract() {
  local raw="$LOGS_DIR/connect-${AGENT}.txt"
  case "$AGENT" in
    codex)
      grep -q 'UNSLOTH_STUDIO_AUTH_TOKEN' "$raw" \
        || guide_fail "Codex env key is no longer UNSLOTH_STUDIO_AUTH_TOKEN (connect.py _CODEX_ENV_KEY)"
      if [ -f "$HOME/.codex/config.toml" ]; then
        grep -q 'wire_api = "responses"' "$HOME/.codex/config.toml" \
          || guide_fail "Codex wire_api is no longer \"responses\" in ~/.codex/config.toml"
        cp "$HOME/.codex/config.toml" "$REDACTED_DIR/codex-config.toml"
      fi
      grep -q 'codex --oss --profile unsloth_api' "$raw" \
        || echo "::warning::Codex launch command changed from 'codex --oss --profile unsloth_api'"
      ;;
    claude)
      grep -q 'ANTHROPIC_AUTH_TOKEN' "$raw" \
        || guide_fail "Claude no longer exports ANTHROPIC_AUTH_TOKEN (connect.py claude())"
      if [ -f "$HOME/.claude/settings.json" ]; then
        grep -q '"CLAUDE_CODE_ATTRIBUTION_HEADER"' "$HOME/.claude/settings.json" \
          || echo "::warning::CLAUDE_CODE_ATTRIBUTION_HEADER not written to ~/.claude/settings.json (ensure_claude_attribution_header)"
        cp "$HOME/.claude/settings.json" "$REDACTED_DIR/claude-settings.json"
      fi
      ;;
    hermes)
      grep -q 'UNSLOTH_API_KEY' "$raw" \
        || guide_fail "Hermes env key is no longer UNSLOTH_API_KEY (connect.py _HERMES_ENV_KEY)"
      [ -f "$HOME/.hermes/config.yaml" ] && cp "$HOME/.hermes/config.yaml" "$REDACTED_DIR/hermes-config.yaml"
      ;;
    openclaw)
      if [ -f "$HOME/.openclaw/openclaw.json" ]; then
        grep -q '"openai-completions"' "$HOME/.openclaw/openclaw.json" \
          || echo "::warning::OpenClaw provider api is no longer 'openai-completions' (write_openclaw_config)"
        cp "$HOME/.openclaw/openclaw.json" "$REDACTED_DIR/openclaw.json"
      fi
      ;;
    opencode)
      [ -f "$HOME/.config/opencode/opencode.json" ] && cp "$HOME/.config/opencode/opencode.json" "$REDACTED_DIR/opencode.json"
      ;;
  esac
  redact "$REDACTED_DIR"/* 2>/dev/null || true
}

# Heavyweight agents (hermes, openclaw) bake a large system prompt + tool JSON
# schemas into every request, which a CPU runner cannot prefill before the invoke
# timeout. As with claude's --tools, we shrink the request from the agent's own
# config: zero tools for the connection probe collapses the prompt to a few
# hundred tokens, since both CLIs gate the bulk of their prompt on having tools.

# Hermes: an explicit empty cli toolset disables all tools (and drops the
# tool-gated guidance blocks), so -z sends ~300 tokens instead of thousands.
# hermes ships a DEFAULT config.yaml that already has a populated
# platform_toolsets, and `unsloth connect` merges into it, so we must override
# cli (not just append). That needs a YAML parser, and the runner's bare
# python3 has no PyYAML -- but the venv that ships `unsloth` does (connect.py
# imports yaml), so run the patch with that interpreter.
# (-z reads platform_toolsets.cli; --ignore-rules is a no-op under -z.)
patch_hermes_tools() {  # $1 = none|default
  # Find a python that can import yaml. The runner's bare python3 cannot, but the
  # interpreter in the `unsloth` console-script shebang provably can (it runs
  # connect.py's write_hermes_config, which imports yaml). Try that first, then
  # any python on PATH, then the venv sibling, picking the first with PyYAML.
  local cand py="" shebang
  shebang="$(head -1 "$(command -v unsloth)" 2>/dev/null | sed -n 's/^#![[:space:]]*//p' | awk '{print $1}')"
  for cand in "$shebang" python3 python "$(dirname "$(command -v unsloth)")/python"; do
    [ -n "$cand" ] || continue
    { [ -x "$cand" ] || command -v "$cand" >/dev/null 2>&1; } || continue
    if "$cand" -c 'import yaml' 2>/dev/null; then py="$cand"; break; fi
  done
  [ -n "$py" ] || guide_fail "could not find a python with PyYAML to patch ~/.hermes/config.yaml"
  echo "[hermes] patching config with $py"
  "$py" - "$1" <<'PY'
import os, sys
import yaml
mode = sys.argv[1]
p = os.path.expanduser("~/.hermes/config.yaml")
cfg = (yaml.safe_load(open(p)) or {}) if os.path.exists(p) else {}
ts = cfg.get("platform_toolsets")
if not isinstance(ts, dict):
    ts = cfg["platform_toolsets"] = {}
if mode == "none":
    ts["cli"] = []          # explicit empty list -> zero tools (not "defaults")
else:
    ts.pop("cli", None)     # file-edit needs real tools -> restore defaults
with open(p, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
print(f"[hermes] platform_toolsets.cli = {ts.get('cli', 'default')}")
PY
}

# OpenClaw: 'openclaw agent' has no tool/prompt flags, so we define a 'ci' agent
# in openclaw.json. tools.deny ["*"] sends zero tool schemas (deny always wins)
# for the connection probe; contextInjection "never" + defaults.skipBootstrap
# drop the auto-injected AGENTS.md/SOUL.md bootstrap (the bulk of the prompt) for
# both modes. --agent must reference a defined agent, so write it before invoking.
patch_openclaw_agent() {  # $1 = notools|tools
  python3 - "$1" <<'PY'
import os, sys, json
mode = sys.argv[1]
p = os.path.expanduser("~/.openclaw/openclaw.json")
cfg = json.load(open(p)) if os.path.exists(p) else {}
agents = cfg.setdefault("agents", {})
agents.setdefault("defaults", {})["skipBootstrap"] = True
lst = [a for a in agents.get("list", []) if a.get("id") != "ci"]
agent = {"id": "ci", "contextInjection": "never"}
if mode == "notools":
    agent["tools"] = {"deny": ["*"]}
lst.append(agent)
agents["list"] = lst
with open(p, "w") as fh:
    json.dump(cfg, fh, indent=2)
print(f"[openclaw] agent ci tools = {agent.get('tools', 'default')}")
PY
}

# Build an invoke script that applies connect.py's env then runs the launch
# command (with extra args appended) under bash. We do NOT eval connect's env
# into this shell; we write it into a one-shot script so the export/unset
# semantics are exactly what connect.py printed. The script path is absolute
# so it is valid even when the caller has cd'd into a scratch work dir.
invoke_via_connect() {  # $1=outfile, rest=extra args appended to the command
  local out="$1"; shift
  local script="$LOGS_DIR/invoke-${AGENT}.sh"
  local real; real="$(mktemp)"
  {
    echo "set -uo pipefail"
    echo "$CONNECT_ENV"
    # Append extra args (the prompt / flags) to the launch command verbatim.
    printf '%s' "$CONNECT_CMD"
    local a
    for a in "$@"; do printf ' %q' "$a"; done
    printf '\n'
  } > "$real"
  # Upload a REDACTED copy of the script, but EXECUTE the un-redacted one from a
  # temp path outside the artifact dir. Redacting the script we run would turn
  # the real `export TOKEN=sk-...` line into `export TOKEN=<REDACTED>`, which is
  # invalid bash (the `<`/`>` are redirections) and silently breaks every agent.
  # Writing the redacted copy up front keeps the key out of the artifact even if
  # the run times out (run_timed exits before returning here).
  cp "$real" "$script"; redact "$script"
  echo "[$AGENT] invoking (timeout ${TIMEOUT}s): $CONNECT_CMD $*"
  run_timed "$out" bash "$real"
  local rc=$?
  rm -f "$real"
  redact "$out"  # the transcript can echo the token; scrub before upload
  return "$rc"
}

# ═════════════════════════════════════════════════════════════════════════
case "$MODE" in
  # ── connection: trivial prompt, assert a non-empty, error-free reply ────
  connection)
    PROMPT='Reply with exactly the single word: pong'
    OUT="$LOGS_DIR/${AGENT}-connection.txt"
    if [ "$AGENT" = "pi" ]; then
      write_pi_config
      run_timed "$OUT" pi -p --provider unsloth --model "$UNSLOTH_MODEL_ID" "$PROMPT"
    else
      parse_connect
      crosscheck_contract
      # claude/codex run in print mode via the flags connect.py emits
      # (claude -p / codex exec). For agents whose default subcommand prints
      # to stdout we pass the prompt through ctx.args.
      case "$AGENT" in
        claude)   invoke_via_connect "$OUT" "${CLAUDE_CONNECT_FLAGS[@]}" -p "$PROMPT" ;;
        codex)    invoke_via_connect "$OUT" exec --dangerously-bypass-approvals-and-sandbox "$PROMPT" ;;
        opencode) invoke_via_connect "$OUT" run "$PROMPT" ;;
        hermes)   patch_hermes_tools none
                  invoke_via_connect "$OUT" -z "$PROMPT" ;;
        openclaw) patch_openclaw_agent notools
                  invoke_via_connect "$OUT" agent --local --agent ci \
                    --model "unsloth/${UNSLOTH_MODEL_ID}" --message "$PROMPT" ;;
        *)        invoke_via_connect "$OUT" "$PROMPT" ;;
      esac
    fi
    # A non-zero exit from the documented launch command is drift even if it
    # printed something: a benign-looking "command not found" / usage dump would
    # otherwise slip past assert_reply (which only flags empty/error-keyword text).
    rc=$?
    [ "$rc" -eq 0 ] || guide_fail "the documented launch command exited non-zero (rc=$rc) -- see the transcript above"
    assert_reply "$OUT"
    echo "[$AGENT] connection OK"
    ;;

  # ── file-edit: deterministic 2-turn hello.py test (Qwen3.5-2B) ──────────
  file-edit)
    WORK="$WORKDIR_BASE/${AGENT}"
    rm -rf "$WORK"; mkdir -p "$WORK"
    OUT1="$LOGS_DIR/${AGENT}-fileedit-turn1.txt"
    OUT2="$LOGS_DIR/${AGENT}-fileedit-turn2.txt"
    T1='Create a file named hello.py in the current directory whose entire contents are a single line: print("Hello"). Do not run it.'
    T2='Run hello.py with python and show me the exact output.'

    # The connect.py recipe writers + crosscheck must see the repo; run them
    # from the repo root BEFORE cd-ing into the scratch work dir.
    if [ "$AGENT" != "pi" ]; then
      parse_connect
      crosscheck_contract
      # File-edit needs real tools, so we cannot zero them as in connection.
      # hermes keeps default tools; openclaw still strips its AGENTS.md/SOUL.md
      # bootstrap (the largest prompt chunk) via the 'ci' agent. The scratch work
      # dir is empty, so no project context files are auto-loaded either.
      case "$AGENT" in
        hermes)   patch_hermes_tools default ;;
        openclaw) patch_openclaw_agent tools ;;
      esac
    else
      write_pi_config
    fi

    # Drive from inside the work dir so the agent edits files there. All log
    # writes use absolute $LOGS_DIR, so cwd does not matter for them.
    cd "$WORK" || guide_fail "could not enter work dir $WORK"

    invoke_turn() {  # $1=outfile $2=continue? $3=prompt
      local out="$1" cont="$2" prompt="$3"
      case "$AGENT" in
        pi)       run_timed "$out" pi -p --provider unsloth --model "$UNSLOTH_MODEL_ID" "$prompt" ;;
        claude)
          # --dangerously-skip-permissions lets headless claude actually use the
          # Write/Bash tools (otherwise it blocks on an approval prompt and emits
          # nothing). IS_SANDBOX=1 (exported above) authorizes it.
          if [ "$cont" = "continue" ]; then
            invoke_via_connect "$out" "${CLAUDE_EDIT_FLAGS[@]}" --dangerously-skip-permissions -p --continue "$prompt"
          else
            invoke_via_connect "$out" "${CLAUDE_EDIT_FLAGS[@]}" --dangerously-skip-permissions -p "$prompt"
          fi ;;
        codex)
          # --dangerously-bypass-approvals-and-sandbox gives codex exec
          # workspace-write (default is read-only -> cannot create hello.py) and
          # skips the bubblewrap sandbox that the runner lacks.
          if [ "$cont" = "continue" ]; then
            invoke_via_connect "$out" exec --dangerously-bypass-approvals-and-sandbox resume --last "$prompt"
          else
            invoke_via_connect "$out" exec --dangerously-bypass-approvals-and-sandbox "$prompt"
          fi ;;
        opencode) invoke_via_connect "$out" run "$prompt" ;;
        hermes)   invoke_via_connect "$out" -z "$prompt" ;;
        openclaw) invoke_via_connect "$out" agent --local --agent ci \
                    --model "unsloth/${UNSLOTH_MODEL_ID}" --message "$prompt" ;;
        *)        invoke_via_connect "$out" "$prompt" ;;
      esac
    }

    # Turn 1: create hello.py.
    invoke_turn "$OUT1" fresh "$T1"
    # Fail on a non-zero agent exit before trusting side effects: an agent can
    # error out (API/tool failure) yet leave a plausible file/transcript behind,
    # which would otherwise slip past the assertions below (mirrors connection).
    rc=$?
    [ "$rc" -eq 0 ] || { echo "[$AGENT] turn-1 transcript:"; tail -40 "$OUT1" 2>/dev/null || true; \
      guide_fail "turn 1 (create hello.py) exited non-zero (rc=$rc)"; }

    # Hard assertions on the side effect (the real test): file + content + run.
    if [ ! -f hello.py ]; then
      echo "[$AGENT] turn-1 transcript:"; tail -40 "$OUT1" 2>/dev/null || true
      guide_fail "turn 1 did not create hello.py"
    fi
    grep -q 'Hello' hello.py || guide_fail "hello.py does not contain 'Hello'"
    RUN_OUT="$(python3 hello.py 2>&1 || true)"
    [ "$RUN_OUT" = "Hello" ] || guide_fail "python3 hello.py printed '$RUN_OUT', expected exactly 'Hello'"
    echo "[$AGENT] turn 1 OK (file created, prints 'Hello')"

    # Turn 2: same cwd + session continuation; assert the agent's run output
    # contains Hello. Narration drift is WARN-only, missing output is a hard fail.
    invoke_turn "$OUT2" continue "$T2"
    rc=$?
    [ "$rc" -eq 0 ] || { echo "[$AGENT] turn-2 transcript:"; tail -60 "$OUT2" 2>/dev/null || true; \
      guide_fail "turn 2 (run hello.py) exited non-zero (rc=$rc)"; }
    if grep -q 'Hello' "$OUT2"; then
      echo "[$AGENT] turn 2 OK (run output contains 'Hello')"
    else
      echo "[$AGENT] turn-2 transcript:"; tail -60 "$OUT2" 2>/dev/null || true
      guide_fail "turn 2 run/bash output did not contain 'Hello'"
    fi
    cd "$REPO_ROOT" || true
    echo "[$AGENT] file-edit OK"
    ;;

  # ── attribution-ab: Claude Code KV-cache HIT vs MISS ────────────────────
  attribution-ab)
    [ "$AGENT" = "claude" ] || guide_fail "attribution-ab only applies to claude"
    # The llama-server log filename uses the INTERNAL random llama.cpp port,
    # not STUDIO_PORT, so we never glob by port: assert-prompt-cache.sh picks
    # the newest llama-*.log and we slice it by a byte offset (`mark`) captured
    # right before the measured turn, so an earlier turn's reuse can't leak in.
    LLAMA_LOG_DIR="${UNSLOTH_LLAMA_LOG_DIR:-$HOME/.unsloth/studio/logs/llama-server}"
    export LLAMA_LOG_DIR
    parse_connect          # writes ~/.claude/settings.json (header=0) + env
    crosscheck_contract
    PROMPT='Reply with exactly the single word: pong'

    # Phase A: header DISABLED (=0, the documented setting) -> expect a HIT on
    # the continued turn. connect.py's ensure_claude_attribution_header() set 0.
    invoke_via_connect "$LOGS_DIR/claude-ab-hit-1.txt" -p "$PROMPT"        # turn 1 primes
    FROM_HIT="$(bash "$CACHE_HELPER" mark)"                                # offset before turn 2
    invoke_via_connect "$LOGS_DIR/claude-ab-hit-2.txt" -p --continue "$PROMPT again"
    CACHE_LOG_FROM="$FROM_HIT" bash "$CACHE_HELPER" log HIT

    # Phase B: header ENABLED -> expect a MISS. The header prepends a
    # per-request-changing attribution line to the system prompt, so the shared
    # prefix changes every turn and the KV cache is invalidated (~90% slower);
    # this is exactly what the guide flag prevents.
    python3 - <<'PY'
import json, os
p = os.path.expanduser("~/.claude/settings.json")
s = json.load(open(p)) if os.path.exists(p) else {}
s.setdefault("env", {})["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "1"
json.dump(s, open(p, "w"), indent=2)
PY
    invoke_via_connect "$LOGS_DIR/claude-ab-miss-1.txt" -p "$PROMPT"
    FROM_MISS="$(bash "$CACHE_HELPER" mark)"
    invoke_via_connect "$LOGS_DIR/claude-ab-miss-2.txt" -p --continue "$PROMPT again"
    CACHE_LOG_FROM="$FROM_MISS" bash "$CACHE_HELPER" log MISS
    echo "[claude] attribution A/B OK (header=0 HIT, header=1 MISS)"
    ;;

  *)
    echo "agent-guides-drive.sh: unknown mode '$MODE'" >&2
    exit 2
    ;;
esac
