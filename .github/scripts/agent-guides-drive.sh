#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Drive one coding agent against the running `unsloth run` server. Failures here are guide drift:
# the recipe comes from `unsloth start <agent> --no-launch`.
# Usage: agent-guides-drive.sh connection|file-edit <agent> | attribution-ab claude
set -uo pipefail

MODE="${1:?usage: agent-guides-drive.sh <mode> <agent>}"
AGENT="${2:?usage: agent-guides-drive.sh <mode> <agent>}"

: "${UNSLOTH_BASE_URL:?serve step did not export UNSLOTH_BASE_URL}"
: "${UNSLOTH_API_KEY:?serve step did not export UNSLOTH_API_KEY}"
: "${UNSLOTH_MODEL_ID:?serve step did not export UNSLOTH_MODEL_ID}"
TIMEOUT="${AGENT_INVOKE_TIMEOUT:-180}"
# opencode makes an extra session-naming call, so it needs a longer timeout than the others.
case "$AGENT" in
  opencode)
    # Double only a bare-integer value; timeout(1) suffixes are left unchanged.
    case "$TIMEOUT" in
      *[!0-9]*) ;;
      *) TIMEOUT=$(( TIMEOUT * 2 )) ;;
    esac
    ;;
esac

# Claude refuses --dangerously-skip-permissions outside a sandbox; the CI runner is one.
export IS_SANDBOX=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOGS_DIR="$REPO_ROOT/logs"
REDACTED_DIR="$REPO_ROOT/redacted-configs"
WORKDIR_BASE="$REPO_ROOT/agent-workdir"
CACHE_HELPER="$SCRIPT_DIR/assert-prompt-cache.sh"
mkdir -p "$LOGS_DIR" "$REDACTED_DIR"
CONNECT_REF="unsloth_cli/commands/start.py"

# Shrink Claude's prefill for CPU serving. --tools limits which schemas are sent;
# --allowedTools only gates permission.
CLAUDE_CONNECT_FLAGS=(
  --system-prompt-file "$SCRIPT_DIR/ci-connect-prompt.txt"
  --tools ""
)
CLAUDE_EDIT_FLAGS=(
  --system-prompt-file "$SCRIPT_DIR/ci-min-system-prompt.txt"
  --tools "Bash,Edit,Write,Read"
)

guide_fail() {
  echo "::error::[guide drift] agent=${AGENT}: $* (preflight passed + install OK, so the documented flow in ${CONNECT_REF} drifted)." >&2
  exit 1
}

# Portable across GNU and BSD sed so redaction is never skipped.
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

# Print with the key scrubbed, leaving the file intact for env parsing.
cat_redacted() {
  sed "s#${UNSLOTH_API_KEY}#<REDACTED>#g" "$1"
}

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

# Run under a hard timeout. Sets TIMED_OUT when the cap was hit; a 137 counts only at the
# deadline, since an unrelated SIGKILL (OOM) also returns 137.
run_timed() {
  local out="$1"; shift
  TIMED_OUT=0
  TURN_DONE=0
  local t0=$SECONDS
  local rc
  if [ -z "${TURN_DONE_RE:-}" ]; then
    timeout --kill-after=30 "$TIMEOUT" "$@" > "$out" 2>&1
    rc=$?
  else
    # Agents with an end-of-run marker (TURN_DONE_RE) get EXIT_GRACE seconds to exit, then are killed.
    : > "$out"
    timeout --kill-after=30 "$TIMEOUT" "$@" > "$out" 2>&1 &
    local tpid=$! seen=""
    while kill -0 "$tpid" 2>/dev/null; do
      if [ -z "$seen" ]; then
        grep -qF -- "$TURN_DONE_RE" "$out" 2>/dev/null && seen=$SECONDS
      elif [ $(( SECONDS - seen )) -ge "${EXIT_GRACE:-30}" ]; then
        TURN_DONE=1
        # Kill the whole process group of timeout's child, never our own group.
        local kid pg
        kid="$(pgrep -P "$tpid" 2>/dev/null | head -1)"
        pg="$(ps -o pgid= -p "${kid:-0}" 2>/dev/null | tr -d ' ')"
        if [ -n "$pg" ] && [ "$pg" != "$(ps -o pgid= -p $$ | tr -d ' ')" ]; then
          kill -TERM "-$pg" 2>/dev/null || true
          sleep 10
          kill -KILL "-$pg" 2>/dev/null || true
        else
          pkill -TERM -P "$tpid" 2>/dev/null || true
          sleep 10
          pkill -KILL -P "$tpid" 2>/dev/null || true
        fi
        break
      fi
      sleep 2
    done
    wait "$tpid"
    rc=$?
  fi
  local elapsed=$(( SECONDS - t0 ))
  # Both 124 and 137 must agree with the clock; allow one second of slack for SECONDS truncation.
  local expired=0
  case "$TIMEOUT" in
    *[!0-9]*) [ "$rc" -eq 124 ] && expired=1 ;;
    *)
      if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
        [ "$elapsed" -ge $(( TIMEOUT - 1 )) ] && expired=1
      fi
      ;;
  esac
  if [ "$expired" -eq 1 ]; then
    redact "$out"  # guide_fail may exit below, so scrub the transcript here too
    echo "[$AGENT] last 40 lines before timeout:"; tail -40 "$out" 2>/dev/null || true
    [ -s "$out" ] || guide_fail "invoke timed out after ${TIMEOUT}s having printed nothing (headless-TTY hang -- the recipe likely needs a non-interactive/print flag)"
    TIMED_OUT=1
    echo "::warning::[$AGENT] the CLI printed a transcript but did not exit within ${TIMEOUT}s; whether that is fatal is the caller's call."
    if [ -n "${TURN_DONE_RE:-}" ] && grep -qF -- "$TURN_DONE_RE" "$out" 2>/dev/null; then
      TURN_DONE=1
    fi
  fi
  if [ "${TURN_DONE:-0}" = 1 ]; then
    echo "::warning::[$AGENT] the CLI logged the end of its run (${TURN_DONE_RE}) and then would not exit."
  fi
  return "$rc"
}

# Read a value from an `export VAR=...` line in the --no-launch output.
raw_env() {  # $1 = var name -> value (one shlex-quote layer stripped)
  local raw="$LOGS_DIR/connect-${AGENT}.txt"
  local v; v="$(sed -n "s/^export $1=//p" "$raw" | tail -1)"
  v="${v#\'}"; v="${v%\'}"; printf '%s' "$v"
}

# Sets CONNECT_ENV and CONNECT_CMD; also runs start.py's config writers as a side effect.
parse_connect() {
  local raw="$LOGS_DIR/connect-${AGENT}.txt"
  local yolo=()
  [ -n "${CONNECT_YOLO:-}" ] && yolo=(--yolo)
  # dsh's `--profile headless`: its default recipe opens the browser UI instead.
  # shellcheck disable=SC2206
  local passthrough=(${CONNECT_START_ARGS:-})
  if ! unsloth start "$AGENT" --no-launch "${yolo[@]}" --api-key "$UNSLOTH_API_KEY" \
      "${passthrough[@]}" > "$raw" 2>&1; then
    cat_redacted "$raw"
    guide_fail "'unsloth start ${AGENT} --no-launch' exited non-zero"
  fi
  echo "[$AGENT] connect --no-launch printed:"; cat_redacted "$raw"
  CONNECT_ENV="$(grep -E '^(export |unset )' "$raw" || true)"
  # The launch command is the last line that is not an export or status line.
  CONNECT_CMD="$(grep -vE '^(export |unset |Unsloth |Updated |Disabled |Warning|Loading)' "$raw" \
    | grep -E '[^[:space:]]' | tail -1)"
  [ -n "$CONNECT_CMD" ] || guide_fail "could not parse a launch command from connect --no-launch output"
  redact "$raw"
}

crosscheck_contract() {
  local raw="$LOGS_DIR/connect-${AGENT}.txt"
  local cfg home
  case "$AGENT" in
    codex)
      grep -q 'UNSLOTH_STUDIO_AUTH_TOKEN' "$raw" \
        || guide_fail "Codex env key is no longer UNSLOTH_STUDIO_AUTH_TOKEN (start.py _CODEX_ENV_KEY)"
      home="$(raw_env CODEX_HOME)"
      [ -n "$home" ] || guide_fail "CODEX_HOME missing from connect output (start.py codex())"
      cfg="$home/config.toml"
      if [ -f "$cfg" ]; then
        grep -q 'wire_api = "responses"' "$cfg" \
          || guide_fail "Codex wire_api is no longer \"responses\" in \$CODEX_HOME/config.toml"
        cp "$cfg" "$REDACTED_DIR/codex-config.toml"
      fi
      grep -q 'codex --oss --profile unsloth_api' "$raw" \
        || echo "::warning::Codex launch command changed from 'codex --oss --profile unsloth_api'"
      ;;
    claude)
      grep -q 'ANTHROPIC_AUTH_TOKEN' "$raw" \
        || guide_fail "Claude no longer exports ANTHROPIC_AUTH_TOKEN (start.py claude())"
      grep -q 'CLAUDE_CODE_ATTRIBUTION_HEADER' "$raw" \
        || echo "::warning::CLAUDE_CODE_ATTRIBUTION_HEADER no longer set for the session (start.py claude())"
      ;;
    hermes)
      grep -q 'UNSLOTH_API_KEY' "$raw" \
        || guide_fail "Hermes env key is no longer UNSLOTH_API_KEY (start.py _HERMES_ENV_KEY)"
      home="$(raw_env HERMES_HOME)"
      [ -n "$home" ] || guide_fail "HERMES_HOME missing from connect output (start.py hermes())"
      cfg="$home/config.yaml"
      [ -f "$cfg" ] && cp "$cfg" "$REDACTED_DIR/hermes-config.yaml"
      ;;
    openclaw)
      cfg="$(raw_env OPENCLAW_CONFIG_PATH)"
      if [ -n "$cfg" ] && [ -f "$cfg" ]; then
        grep -q '"openai-completions"' "$cfg" \
          || echo "::warning::OpenClaw provider api is no longer 'openai-completions' (write_openclaw_config)"
        cp "$cfg" "$REDACTED_DIR/openclaw.json"
      fi
      ;;
    opencode)
      cfg="$(raw_env OPENCODE_CONFIG)"
      [ -n "$cfg" ] && [ -f "$cfg" ] && cp "$cfg" "$REDACTED_DIR/opencode.json"
      ;;
    pi)
      cfg="$(raw_env HOME)/.pi/agent/models.json"
      if [ -f "$cfg" ]; then
        grep -q '"openai-completions"' "$cfg" \
          || echo "::warning::Pi provider api is no longer 'openai-completions' (write_pi_config)"
        cp "$cfg" "$REDACTED_DIR/pi-models.json"
      fi
      ;;
    dsh)
      grep -q 'UNSLOTH_API_KEY' "$raw" \
        || guide_fail "dsh env key is no longer UNSLOTH_API_KEY (start.py _DSH_ENV_KEY)"
      home="$(raw_env DSH_HOME)"
      [ -n "$home" ] || guide_fail "DSH_HOME missing from connect output (start.py dsh())"
      # A --patch overlay, not settings.yaml: dsh 0.1.7 imports that only after the first boot.
      cfg="$home/unsloth.patch.yml"
      [ -f "$cfg" ] || guide_fail "dsh patch $cfg missing (start.py write_dsh_patch)"
      grep -qF -- "--patch $cfg" "$raw" \
        || guide_fail "dsh launch command no longer passes --patch $cfg (start.py _dsh_command)"
      grep -q 'openai-completions' "$cfg" \
        || echo "::warning::dsh provider api is no longer 'openai-completions' (write_dsh_patch)"
      cp "$cfg" "$REDACTED_DIR/dsh-patch.yml"
      ;;
    vibe)
      grep -q 'UNSLOTH_API_KEY' "$raw" \
        || guide_fail "Vibe env key is no longer UNSLOTH_API_KEY (start.py _VIBE_ENV_KEY)"
      cfg="$(raw_env VIBE_PROVIDERS)"
      [ -n "$cfg" ] || guide_fail "VIBE_PROVIDERS missing from connect output (start.py _vibe_env)"
      grep -q '"api_style": "openai"' <<<"$cfg" \
        || echo "::warning::Vibe provider api_style is no longer 'openai' (start.py _vibe_env)"
      printf '%s\n%s\n' "$cfg" "$(raw_env VIBE_MODELS)" > "$REDACTED_DIR/vibe-env.json"
      ;;
  esac
  redact "$REDACTED_DIR"/* 2>/dev/null || true
}

# Heavyweight agents: shrink the request via their own config so CPU prefill fits the timeout.

# platform_toolsets.cli must be set to [] explicitly to get zero tools. Needs PyYAML, so use
# the interpreter from the `unsloth` venv.
patch_hermes_tools() {  # $1 = none|default
  # Check the raw var before appending /config.yaml; the joined path is never empty.
  local home; home="$(raw_env HERMES_HOME)"
  [ -n "$home" ] || guide_fail "Hermes HERMES_HOME missing from connect output (start.py hermes())"
  local cfg; cfg="$home/config.yaml"
  local cand py="" shebang
  shebang="$(head -1 "$(command -v unsloth)" 2>/dev/null | sed -n 's/^#![[:space:]]*//p' | awk '{print $1}')"
  for cand in "$shebang" python3 python "$(dirname "$(command -v unsloth)")/python"; do
    [ -n "$cand" ] || continue
    { [ -x "$cand" ] || command -v "$cand" >/dev/null 2>&1; } || continue
    if "$cand" -c 'import yaml' 2>/dev/null; then py="$cand"; break; fi
  done
  [ -n "$py" ] || guide_fail "could not find a python with PyYAML to patch the hermes session config"
  echo "[hermes] patching $cfg with $py"
  "$py" - "$1" "$cfg" <<'PY'
import os, sys
import yaml
mode = sys.argv[1]
p = sys.argv[2]
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

# openclaw has no tool/prompt flags, so define a 'ci' agent in openclaw.json before invoking.
patch_openclaw_agent() {  # $1 = notools|tools
  local cfg; cfg="$(raw_env OPENCLAW_CONFIG_PATH)"
  [ -n "$cfg" ] || guide_fail "OpenClaw OPENCLAW_CONFIG_PATH missing from connect output (start.py openclaw())"
  python3 - "$1" "$cfg" <<'PY'
import os, sys, json
mode = sys.argv[1]
p = sys.argv[2]
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

# Write start.py's env into a one-shot script instead of eval-ing it here.
invoke_via_connect() {  # $1=outfile, rest=extra args appended to the command
  local out="$1"; shift
  local script="$LOGS_DIR/invoke-${AGENT}.sh"
  local real; real="$(mktemp)"
  local cmd="${CONNECT_CMD_OVERRIDE:-$CONNECT_CMD}"
  # V2 requires --standalone after the run subcommand, not before it.
  if [ "$AGENT" = opencode ] && [[ "$cmd" == *" --standalone" ]] && [ "${1:-}" = run ]; then
    cmd="${cmd% --standalone}"
    set -- run --standalone "${@:2}"
  fi
  {
    echo "set -uo pipefail"
    echo "$CONNECT_ENV"
    [ -n "${CONNECT_ENV_EXTRA:-}" ] && echo "$CONNECT_ENV_EXTRA"
    printf '%s' "$cmd"
    local a
    for a in "$@"; do printf ' %q' "$a"; done
    printf '\n'
  } > "$real"
  # Upload a redacted copy but run the original: `<REDACTED>` is invalid bash.
  cp "$real" "$script"; redact "$script"
  echo "[$AGENT] invoking (timeout ${TIMEOUT}s): ${cmd//${UNSLOTH_API_KEY}/<REDACTED>} $*"
  run_timed "$out" bash "$real"
  local rc=$?
  rm -f "$real"
  redact "$out"  # the transcript can echo the token; scrub before upload
  return "$rc"
}

case "$MODE" in
  connection)
    PROMPT='Reply with exactly the single word: pong'
    OUT="$LOGS_DIR/${AGENT}-connection.txt"
    case "$AGENT" in dsh) CONNECT_START_ARGS='--profile headless' ;; esac
    parse_connect
    crosscheck_contract
    case "$AGENT" in
      claude)   invoke_via_connect "$OUT" "${CLAUDE_CONNECT_FLAGS[@]}" -p "$PROMPT" ;;
      codex)    invoke_via_connect "$OUT" exec --dangerously-bypass-approvals-and-sandbox "$PROMPT" ;;
      opencode) invoke_via_connect "$OUT" run "$PROMPT" ;;
      pi)       invoke_via_connect "$OUT" -p "$PROMPT" ;;
      vibe)     invoke_via_connect "$OUT" --disabled-tools '*' -p "$PROMPT" ;;
      hermes)   patch_hermes_tools none
                invoke_via_connect "$OUT" -z "$PROMPT" ;;
      openclaw) patch_openclaw_agent notools
                # Logged only when an openclaw run is over, so a banner cannot fake it.
                TURN_DONE_RE='ended with stopReason='
                CONNECT_CMD_OVERRIDE=openclaw invoke_via_connect "$OUT" agent --local --agent ci \
                  --model "unsloth/${UNSLOTH_MODEL_ID}" --message "$PROMPT" ;;
      *)        invoke_via_connect "$OUT" "$PROMPT" ;;
    esac
    # Non-zero exit is drift even with output. A timeout is fatal: assert_reply cannot tell a reply from a banner.
    rc=$?
    # A cap is reported separately from drift; it is still fatal.
    if [ "${TIMED_OUT:-0}" = 1 ] && [ "${TURN_DONE:-0}" != 1 ]; then
      echo "::error::[$AGENT] the documented launch command started but never completed a turn within ${TIMEOUT}s. The recipe in ${CONNECT_REF} is not implicated: the transcript above shows what the CLI was doing when the cap hit. A connection or model-server failure looks like this; so does a headless prompt, which prints nothing at all." >&2
      exit 1
    fi
    # TURN_DONE proves the turn finished, so the rc from run_timed's own kill is skipped only then.
    if [ "${TURN_DONE:-0}" != 1 ]; then
      [ "$rc" -eq 0 ] || guide_fail "the documented launch command exited non-zero (rc=$rc) -- see the transcript above"
    fi
    assert_reply "$OUT"
    echo "[$AGENT] connection OK"
    ;;

  file-edit)
    WORK="$WORKDIR_BASE/${AGENT}"
    rm -rf "$WORK"; mkdir -p "$WORK"
    OUT1="$LOGS_DIR/${AGENT}-fileedit-turn1.txt"
    OUT2="$LOGS_DIR/${AGENT}-fileedit-turn2.txt"
    T1='Create a file named hello.py in the current directory whose entire contents are a single line: print("Hello"). Do not run it.'
    # Keep a single instruction; a two-part prompt made opencode narrate instead of acting.
    T2='Run hello.py with python and show me the exact output.'

    # Run the recipe writers from the repo root before entering the scratch work dir.
    case "$AGENT" in
      opencode|openclaw|vibe) CONNECT_YOLO=1 ;;
      dsh) CONNECT_YOLO=1; CONNECT_START_ARGS='--profile headless' ;;
    esac
    parse_connect
    crosscheck_contract
    case "$AGENT" in
      hermes)   patch_hermes_tools default ;;
      openclaw) patch_openclaw_agent tools ;;
    esac

    cd "$WORK" || guide_fail "could not enter work dir $WORK"

    invoke_turn() {  # $1=outfile $2=continue? $3=prompt
      local out="$1" cont="$2" prompt="$3"
      case "$AGENT" in
        pi)
          if [ "$cont" = "continue" ]; then
            invoke_via_connect "$out" -p --continue "$prompt"
          else
            invoke_via_connect "$out" -p "$prompt"
          fi ;;
        claude)
          # IS_SANDBOX=1 (exported above) authorizes --dangerously-skip-permissions.
          if [ "$cont" = "continue" ]; then
            invoke_via_connect "$out" "${CLAUDE_EDIT_FLAGS[@]}" --dangerously-skip-permissions -p --continue "$prompt"
          else
            invoke_via_connect "$out" "${CLAUDE_EDIT_FLAGS[@]}" --dangerously-skip-permissions -p "$prompt"
          fi ;;
        codex)
          # Needed for workspace-write, and the runner lacks bubblewrap.
          if [ "$cont" = "continue" ]; then
            invoke_via_connect "$out" exec --dangerously-bypass-approvals-and-sandbox resume --last "$prompt"
          else
            invoke_via_connect "$out" exec --dangerously-bypass-approvals-and-sandbox "$prompt"
          fi ;;
        opencode) invoke_via_connect "$out" run "$prompt" ;;
        hermes)   invoke_via_connect "$out" -z "$prompt" ;;
        vibe)
          local tools=(--enabled-tools bash --enabled-tools read_file --enabled-tools write_file)
          if [ "$cont" = "continue" ]; then
            invoke_via_connect "$out" "${tools[@]}" -c -p "$prompt"
          else
            invoke_via_connect "$out" "${tools[@]}" -p "$prompt"
          fi ;;
        openclaw) CONNECT_CMD_OVERRIDE=openclaw invoke_via_connect "$out" agent --local --agent ci \
                    --model "unsloth/${UNSLOTH_MODEL_ID}" --message "$prompt" ;;
        *)        invoke_via_connect "$out" "$prompt" ;;
      esac
    }

    invoke_turn "$OUT1" fresh "$T1"
    # Fail on non-zero exit before trusting side effects.
    rc=$?
    [ "$rc" -eq 0 ] || [ "${TIMED_OUT:-0}" = 1 ] \
      || { echo "[$AGENT] turn-1 transcript:"; tail -40 "$OUT1" 2>/dev/null || true; \
      guide_fail "turn 1 (create hello.py) exited non-zero (rc=$rc)"; }

    if [ ! -f hello.py ]; then
      echo "[$AGENT] turn-1 transcript:"; tail -40 "$OUT1" 2>/dev/null || true
      guide_fail "turn 1 did not create hello.py"
    fi
    grep -q 'Hello' hello.py || guide_fail "hello.py does not contain 'Hello'"
    RUN_OUT="$(python3 hello.py 2>&1 || true)"
    [ "$RUN_OUT" = "Hello" ] || guide_fail "python3 hello.py printed '$RUN_OUT', expected exactly 'Hello'"
    echo "[$AGENT] turn 1 OK (file created, prints 'Hello')"

    # A cap is fatal for turn 2: 'Hello' appears in source, output and narration alike.
    invoke_turn "$OUT2" continue "$T2"
    rc=$?
    [ "$rc" -eq 0 ] \
      || { echo "[$AGENT] turn-2 transcript:"; tail -60 "$OUT2" 2>/dev/null || true; \
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

  attribution-ab)
    [ "$AGENT" = "claude" ] || guide_fail "attribution-ab only applies to claude"
    # The log name uses llama.cpp's random internal port, so slice the newest log by byte offset.
    LLAMA_LOG_DIR="${UNSLOTH_LLAMA_LOG_DIR:-$HOME/.unsloth/studio/logs/llama-server}"
    export LLAMA_LOG_DIR
    parse_connect
    crosscheck_contract
    PROMPT='Reply with exactly the single word: pong'

    # A cap is fatal here. Only --tools "": gemma-3-270m rejects tool schemas, and the
    # system prompt must stay because it carries the attribution line.
    ab_invoke() {
      invoke_via_connect "$1" --tools "" "${@:2}"
      [ "${TIMED_OUT:-0}" = 1 ] && guide_fail "attribution-ab invoke timed out after ${TIMEOUT}s; the A/B cannot be judged from a partial turn"
      return 0
    }

    # Phase A: start.py's suppression, expect a cache HIT on the continued turn.
    ab_invoke "$LOGS_DIR/claude-ab-hit-1.txt" -p "$PROMPT"
    FROM_HIT="$(bash "$CACHE_HELPER" mark)"
    ab_invoke "$LOGS_DIR/claude-ab-hit-2.txt" -p --continue "$PROMPT again"
    CACHE_LOG_FROM="$FROM_HIT" bash "$CACHE_HELPER" log HIT

    # Phase B: header enabled and suppression flags stripped, expect a MISS. Session-only.
    CONNECT_ENV_EXTRA='export CLAUDE_CODE_ATTRIBUTION_HEADER=1'
    CONNECT_CMD_OVERRIDE="$(printf '%s' "$CONNECT_CMD" \
      | sed -E "s/ --exclude-dynamic-system-prompt-sections//; s/ --settings '[^']*'//")"
    ab_invoke "$LOGS_DIR/claude-ab-miss-1.txt" -p "$PROMPT"
    FROM_MISS="$(bash "$CACHE_HELPER" mark)"
    ab_invoke "$LOGS_DIR/claude-ab-miss-2.txt" -p --continue "$PROMPT again"
    CACHE_LOG_FROM="$FROM_MISS" bash "$CACHE_HELPER" log MISS
    unset CONNECT_ENV_EXTRA CONNECT_CMD_OVERRIDE
    echo "[claude] attribution A/B OK (suppressed HIT, header=1 MISS)"
    ;;

  # resume: drives the real launch path and checks whether a session persists, with and without --persist.
  resume)
    CODEWORD="PLATYPUS7"
    T1="Remember this codeword for later: ${CODEWORD}. Reply with just the word OK."
    T2="What codeword did I ask you to remember? Reply with just that word."
    WORK="$WORKDIR_BASE/${AGENT}-resume"

    # opencode/claude keep sessions in a fixed user dir, so STABLE_HOME stays empty for them.
    parse_connect
    case "$AGENT" in
      codex)    STABLE_HOME="$(raw_env CODEX_HOME)" ;;
      pi)       STABLE_HOME="$(raw_env HOME)" ;;
      *)        STABLE_HOME="" ;;
    esac

    # A positive file-count delta here means the session persisted.
    resume_tracked_dirs() {
      case "$AGENT" in
        codex)    printf '%s\n' "$HOME/.codex" ;;
        opencode) printf '%s\n' "$HOME/.local/share/opencode" "$HOME/.config/opencode" ;;
        claude)   printf '%s\n' "$HOME/.claude" ;;
        pi)       printf '%s\n' "$HOME/.pi" ;;
        *)        : ;;
      esac
      [ -n "$STABLE_HOME" ] && printf '%s\n' "$STABLE_HOME"
    }
    count_session_files() {
      local total=0 d n
      while IFS= read -r d; do
        [ -n "$d" ] && [ -d "$d" ] || continue
        n="$(find "$d" -type f 2>/dev/null | wc -l)"; total=$((total + n))
      done < <(resume_tracked_dirs)
      echo "$total"
    }

    set_t1_cmd() {
      case "$AGENT" in
        claude)   T1_CMD=("${CLAUDE_CONNECT_FLAGS[@]}" -p "$T1") ;;
        codex)    T1_CMD=(exec "$T1") ;;
        opencode) T1_CMD=(run "$T1") ;;
        pi)       T1_CMD=(-p "$T1") ;;
        *)        guide_fail "resume mode does not cover agent '$AGENT'" ;;
      esac
    }

    launch_turn() {
      local out="$1" rflag="$2"; shift 2
      local flag=(); [ -n "$rflag" ] && flag=("$rflag")
      run_timed "$out" unsloth start "$AGENT" "${flag[@]}" --yolo \
        --api-key "$UNSLOTH_API_KEY" "$@"
      local rc=$?
      redact "$out"
      # A partial turn cannot be judged from a session-store delta, so a hang is fatal.
      [ "${TIMED_OUT:-0}" = 1 ] && guide_fail "invoke timed out after ${TIMEOUT}s; a resume pass cannot be judged from a partial turn"
      return "$rc"
    }

    # Runs in the main shell, not a substitution, so guide_fail actually fails the job.
    RESULT=""
    run_pass() {
      local rflag="$1" label="baseline"
      [ -n "$rflag" ] && label="resume"
      rm -rf "$WORK"; mkdir -p "$WORK"
      set_t1_cmd
      local out="$LOGS_DIR/${AGENT}-resume-${label}.txt"
      local before after rc
      before="$(count_session_files)"
      pushd "$WORK" >/dev/null || guide_fail "could not enter work dir $WORK"
      launch_turn "$out" "$rflag" "${T1_CMD[@]}"; rc=$?
      popd >/dev/null || true
      after="$(count_session_files)"
      echo "[$AGENT] ${label}: session files ${before} -> ${after} (rc=${rc})"
      # The turn must succeed, or a written-then-errored session reads as PERSISTED.
      [ "$rc" -eq 0 ] || { echo "[$AGENT] ${label} transcript (tail):"; tail -30 "$out" 2>/dev/null || true; \
        guide_fail "resume ${label} turn for ${AGENT} exited non-zero (rc=${rc})"; }
      if [ "$after" -gt "$before" ]; then RESULT="PERSISTED"; else RESULT="WIPED"; fi
    }

    run_pass ""; BASELINE="$RESULT"
    case "$AGENT" in
      codex|pi) run_pass "--persist"; RESUME="$RESULT" ;;
      *)        RESUME="n/a (persists either way)" ;;
    esac

    # codex/pi: plain launch is WIPED, only --persist persists. opencode/claude persist either way.
    case "$AGENT" in
      codex|pi)        EXPECT_BASELINE="WIPED" ;;
      opencode|claude) EXPECT_BASELINE="PERSISTED" ;;
    esac

    echo "──────────────────────────────────────────────"
    echo "[$AGENT] RESUME EXPERIMENT"
    echo "  baseline (unsloth start ${AGENT}):                 ${BASELINE}  (expected ${EXPECT_BASELINE})"
    echo "  with --persist (unsloth start ${AGENT} --persist): ${RESUME}"
    echo "──────────────────────────────────────────────"

    [ "$BASELINE" = "$EXPECT_BASELINE" ] \
      || guide_fail "baseline resume behavior for ${AGENT} was ${BASELINE}, expected ${EXPECT_BASELINE}"
    case "$AGENT" in
      codex|pi)
        [ "$RESUME" = "PERSISTED" ] \
          || guide_fail "--persist did not persist ${AGENT}'s session (got ${RESUME}); the session dir is still not stable" ;;
    esac

    # codex recall check is WARN-only; the mechanism gate above is the real assertion.
    if [ "$AGENT" = "codex" ]; then
      rm -rf "$WORK"; mkdir -p "$WORK"
      ( cd "$WORK" && launch_turn "$LOGS_DIR/codex-resume-plant.txt" "--persist" exec "$T1" ) || true
      ( cd "$WORK" && launch_turn "$LOGS_DIR/codex-resume-recall.txt" "--persist" exec resume --last "$T2" ) || true
      if grep -q "$CODEWORD" "$LOGS_DIR/codex-resume-recall.txt" 2>/dev/null; then
        echo "[codex] behavioral recall HIT: resumed session remembered ${CODEWORD}"
      else
        echo "::warning::[codex] behavioral recall MISS (small CI model); mechanism gate still passed"
      fi
    fi
    echo "[$AGENT] resume OK"
    ;;

  *)
    echo "agent-guides-drive.sh: unknown mode '$MODE'" >&2
    exit 2
    ;;
esac
