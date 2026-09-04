// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Turn what the user types into the argv tokens the API takes: `LoadRequest.llama_extra_args` is
 *  one token per entry, so the single string lives only in the control. Deliberately not a
 *  shell, since nothing runs `sh -c` on this path; quotes and backslashes are handled only
 *  because a chat template or grammar needs a space inside one token. */

// Hoisted: these run per token, and biome flags a literal in a hot path.
const NEEDS_QUOTING = /[\s"'\\]/;
const DOUBLE_QUOTE_ESCAPES = /(["\\$`])/g;
const DIGIT = /[0-9]/;
const UNDERSCORE = /_/g;
// biome-ignore lint/suspicious/noControlCharactersInRegex: mirroring the backend's own check
// The same rule as CONTROL_IN_ARGV below, and as the backend's own check.
const CONTROL_CHARACTERS = /[\u0000-\u0008\u000b-\u001f]/;
const INTEGER = /^-?[0-9]+$/;
/** A NUL, or any other C0 control except tab and newline. The two exceptions are the backend's,
 *  from _has_control_characters: a grammar or chat template is routinely multi-line. */
// biome-ignore lint/suspicious/noControlCharactersInRegex: that is exactly what this finds
const CONTROL_IN_ARGV = /[\u0000-\u0008\u000B-\u001F]/;

/** Whether this token is something execve could not carry. The surrogate half is found by walking
 *  the string rather than with a character class, since a class matches BOTH units of a
 *  well-formed pair. Only an UNPAIRED surrogate is refused, which is what Popen raises on. */
function isUnusableInArgv(token: string): boolean {
  if (CONTROL_IN_ARGV.test(token)) {
    return true;
  }
  for (let index = 0; index < token.length; index += 1) {
    const unit = token.charCodeAt(index);
    if (unit >= 0xd800 && unit <= 0xdbff) {
      const next = token.charCodeAt(index + 1);
      if (Number.isNaN(next) || next < 0xdc00 || next > 0xdfff) {
        return true;
      }
      index += 1;
    } else if (unit >= 0xdc00 && unit <= 0xdfff) {
      return true;
    }
  }
  return false;
}
// Hoisted for the same reason as the patterns above: this runs on every keystroke.
const TEXT_ENCODER = new TextEncoder();

/** Longest input the editor will parse, mirroring MAX_EXTRA_ARGS_BYTES in llama_server_args.py. */
export const EXTRA_ARGS_MAX_BYTES = 32 * 1024;
/** Mirrors MAX_EXTRA_ARG_TOKENS in llama_server_args.py. */
export const EXTRA_ARGS_MAX_TOKENS = 256;

/** The one option in llama-server's help that takes two values. */
const TWO_VALUE_FLAGS = new Set(["--control-vector-layer-range"]);

/** Options that take a second value on SOME builds. Today's llama.cpp writes the scale into the
 *  value ("--lora-scaled FNAME:SCALE") while older ones took a separate token, and
 *  _sidecar_weight_files reads both, so the second token is allowed but never required. */
const OPTIONAL_SECOND_VALUE_FLAGS = new Set([
  "--lora-scaled",
  "--control-vector-scaled",
]);

export type ExtraArgsParse = {
  tokens: string[];
  /** Set when a quote is left open, so the row can say so instead of silently dropping it. */
  unterminatedQuote: '"' | "'" | null;
  /** Indices of tokens the user QUOTED, which argv cannot record. `--chat-template '- hello'` is one
   *  option and its value, but the quotes are gone by the time it is a token list and "- hello"
   *  is flag-shaped, so the row called the value missing. Only the editor can tell them apart. */
  quotedIndices: ReadonlySet<number>;
};

/** A stored list with the flags this build refuses removed, values and all. The panel sends what
 *  it holds as an explicit list, which /load validates strictly, so an install upgraded across
 *  a denylist change would fail to load a model that worked yesterday. Mirrors
 *  drop_managed_flags: a flag takes its value, or the orphan reads as a model path. */
/** What `subprocess.list2cmdline` would put on a Windows command line for these. A port of
 *  CPython's rule, because that is the serializer Popen uses and it is not a sum of lengths:
 *  an argument needing quotes has each backslash run before a quote doubled. */
export function windowsCommandLength(tokens: readonly string[]): number {
  const result: string[] = [];
  for (const token of tokens) {
    if (result.length > 0) {
      result.push(" ");
    }
    // Quoted only for whitespace or emptiness, as CPython does: a token that merely contains a quote
    // is not quoted, though its backslashes still double.
    const needQuote = token.includes(" ") || token.includes("\t") || token === "";
    if (needQuote) {
      result.push('"');
    }
    let backslashes = 0;
    for (const character of token) {
      if (character === "\\") {
        backslashes += 1;
        continue;
      }
      if (character === '"') {
        result.push("\\".repeat(backslashes * 2));
        backslashes = 0;
        result.push('\\"');
        continue;
      }
      if (backslashes > 0) {
        result.push("\\".repeat(backslashes));
        backslashes = 0;
      }
      result.push(character);
    }
    if (backslashes > 0) {
      result.push("\\".repeat(backslashes));
      // And once more inside the quotes, or the run would escape the closing one.
      if (needQuote) {
        result.push("\\".repeat(backslashes));
      }
    }
    if (needQuote) {
      result.push('"');
    }
  }
  return result.join("").length;
}

/** Whether this token carries its own value instead of expecting the next one. Not "the name
 *  changed": extraArgFlagName also folds llama.cpp's underscore spelling, so comparing the
 *  folded name against the raw token read "--ctx_size 4096" as attached. Only "=" and an
 *  attached short like -np8 are values in the same token. Mirrors _value_is_attached. */
function valueIsAttached(token: string, flag: string): boolean {
  const raw = token.trim();
  if (raw.includes("=")) {
    return true;
  }
  return raw.replace(UNDERSCORE, "-") !== flag;
}

/** Whether this token's value is the NEXT token rather than part of itself. */
function takesNextToken(
  token: string,
  flag: string,
  next: string | undefined,
): boolean {
  if (valueIsAttached(token, flag)) {
    return false;
  }
  return next !== undefined && extraArgFlagName(next) === null;
}

/** A stored list reduced to what THIS build would accept. Mirrors drop_managed_flags: denied flags
 *  go, tokens carrying control characters or unpaired surrogates go, anything past the size
 *  bounds goes, and a flag never outlives its value (an orphan reads as the model path). The
 *  panel needs this because hydrating turns a stored list into an EXPLICIT request. */
export function sanitizeStoredExtraArgs(
  tokens: readonly string[],
  managed: ReadonlySet<string>,
  limits?: { maxBytes?: number; windowsCommandBudget?: number },
): string[] {
  const kept: string[] = [];
  let skipNext = false;
  for (const [index, token] of tokens.entries()) {
    if (skipNext) {
      skipNext = false;
      continue;
    }
    const flag = extraArgFlagName(token);
    const next = tokens[index + 1];
    if (flag !== null && managed.has(flag)) {
      skipNext = takesNextToken(token, flag, next);
      continue;
    }
    if (isUnusableInArgv(token)) {
      if (flag !== null) {
        skipNext = takesNextToken(token, flag, next);
      } else if (
        kept.length > 0 &&
        extraArgFlagName(kept[kept.length - 1]) !== null
      ) {
        // The flag this value belonged to.
        kept.pop();
      }
      continue;
    }
    if (
      flag !== null &&
      takesNextToken(token, flag, next) &&
      next !== undefined &&
      isUnusableInArgv(next)
    ) {
      // Its value is about to be dropped, so the flag goes with it.
      continue;
    }
    kept.push(token);
  }
  // Then the shapes validate_extra_args refuses outright, whatever their size: a token belonging to
  // no flag, and a two-value option left half-written. The backend re-validates after every
  // cut; this mirror trims by size alone, so it has to know the same rules.
  let bounded = dropUnvalidatableTokens(dropUnusableValues(kept));
  // Then the bounds, shed from the tail, never leaving a flag without its value. The host's own
  // limits when the caller has them: a Windows install takes 24 KiB, not 32.
  const maxBytes = limits?.maxBytes || EXTRA_ARGS_MAX_BYTES;
  const commandBudget = limits?.windowsCommandBudget ?? 0;
  const overBounds = (): boolean =>
    bounded.length > EXTRA_ARGS_MAX_TOKENS ||
    TEXT_ENCODER.encode(bounded.join("")).length > maxBytes ||
    (commandBudget > 0 && windowsCommandLength(bounded) > commandBudget);
  while (bounded.length > 0 && overBounds()) {
    bounded.pop();
    const last = bounded[bounded.length - 1];
    const lastFlag = last === undefined ? null : extraArgFlagName(last);
    // Any flag whose value has just gone, not only one spelled exactly as it normalizes:
    // extraArgFlagName folds llama.cpp's underscores, so comparing the normalized name against
    // the raw token left "--grammar_file" standing after its value was trimmed.
    if (last !== undefined && lastFlag !== null && !valueIsAttached(last, lastFlag)) {
      bounded.pop();
    }
    // Re-applied after every cut, exactly as the backend re-validates after its own: shedding one
    // value of a two-value option leaves the other looking ordinary.
    bounded = dropUnvalidatableTokens(dropUnusableValues(bounded));
  }
  return bounded;
}

/** A list with the flags whose VALUE the backend's own parsers refuse removed. parse_ctx_override
 *  and its siblings raise on a missing or unreadable value, which is a 400 rather than a note.
 *  drop_managed_flags repairs these by shedding its tail, which costs whatever followed;
 *  removing just the option and its value keeps the rest of a legacy list working. */
function dropUnusableValues(tokens: readonly string[]): string[] {
  const out: string[] = [];
  let skipNext = false;
  for (const [index, token] of tokens.entries()) {
    if (skipNext) {
      skipNext = false;
      continue;
    }
    const flag = extraArgFlagName(token);
    if (
      flag === null ||
      !(INTEGER_VALUE_FLAGS.has(flag) || VALUE_REQUIRED_FLAGS.has(flag))
    ) {
      out.push(token);
      continue;
    }
    const attached = valueIsAttached(token, flag);
    const next = tokens[index + 1];
    const value = attached ? token.split("=")[1] : next;
    const missing =
      value === undefined ||
      value === "" ||
      (!attached && extraArgFlagName(value) !== null);
    const minimum = INTEGER_VALUE_MINIMUM[flag];
    const unusable =
      missing ||
      (INTEGER_VALUE_FLAGS.has(flag) &&
        (!INTEGER.test(value.trim()) ||
          (minimum !== undefined && Number(value.trim()) < minimum)));
    if (!unusable) {
      out.push(token);
      continue;
    }
    // The value goes with the flag, or it is left as a bare token, which is the one thing llama-server
    // would read as a model path.
    skipNext = !attached && !missing;
  }
  return out;
}

/** A list reduced to what validate_extra_args would accept on shape alone. Two rules, both
 *  enforced by re-validating after each cut: a bare token belonging to no flag is refused
 *  (llama-server reads a positional as the model path), and a two-value option needs both.
 *  Everything else is left alone, since this side does not know an ordinary flag's arity. */
function dropUnvalidatableTokens(tokens: readonly string[]): string[] {
  const out: string[] = [];
  let pending = 0;
  let twoValuePending = 0;
  // Values still owed to a flag that was itself dropped, so they go rather than being kept as
  // tokens belonging to nothing.
  let droppedOwes = 0;
  // Where the incomplete two-value option starts, so the whole of it can go.
  let ownerAt = -1;
  for (const token of tokens) {
    const flag = extraArgFlagName(token);
    if (droppedOwes > 0 && flag === null) {
      droppedOwes -= 1;
      continue;
    }
    droppedOwes = 0;
    if (flag === null) {
      if (pending <= 0) {
        // Ownerless. Dropped rather than kept, which is what drop_managed_flags does with the same token.
        continue;
      }
      pending -= 1;
      if (twoValuePending > 0) {
        twoValuePending -= 1;
        if (twoValuePending === 0) {
          ownerAt = -1;
        }
      }
      out.push(token);
      continue;
    }
    if (twoValuePending > 0 && ownerAt >= 0) {
      // A new flag arrived while the option still owed a value: the whole option goes, along with the value it did get.
      out.length = ownerAt;
    }
    if (token.includes("=")) {
      // llama.cpp looks the whole token up in its option map, so "--top-k=20" is an argument it has
      // never heard of and validate_extra_args refuses it. The value is inside the token.
      pending = 0;
      twoValuePending = 0;
      ownerAt = -1;
      continue;
    }
    if (token !== token.trim()) {
      // Refused for the same reason: the padding is part of the token llama.cpp looks up, so a quoted
      // "--top-k " never arrives as a flag. Its value goes with it, or the orphan is positional.
      droppedOwes = valueIsAttached(token, flag) ? 0 : 1;
      pending = 0;
      twoValuePending = 0;
      ownerAt = -1;
      continue;
    }
    const attached = valueIsAttached(token, flag);
    if (TWO_VALUE_FLAGS.has(flag)) {
      // An attached value is one of the two, so the option still owes its END.
      pending = attached ? 1 : 2;
      twoValuePending = pending;
      ownerAt = out.length;
    } else if (OPTIONAL_SECOND_VALUE_FLAGS.has(flag)) {
      // Allowed, not owed, so nothing here removes the option for want of it.
      pending = attached ? 1 : 2;
      twoValuePending = 0;
      ownerAt = -1;
    } else {
      pending = attached ? 0 : 1;
      twoValuePending = 0;
      ownerAt = -1;
    }
    out.push(token);
  }
  if (twoValuePending > 0 && ownerAt >= 0) {
    out.length = ownerAt;
  }
  return out;
}

/** The denylist half of the sanitizer, for a caller that only has that to apply. */
export function dropManagedExtraArgs(
  tokens: readonly string[],
  managed: ReadonlySet<string>,
): string[] {
  const kept: string[] = [];
  let skipNext = false;
  for (const [index, token] of tokens.entries()) {
    if (skipNext) {
      skipNext = false;
      continue;
    }
    const flag = extraArgFlagName(token);
    if (flag === null || !managed.has(flag)) {
      kept.push(token);
      continue;
    }
    skipNext = takesNextToken(token, flag, tokens[index + 1]);
  }
  return kept;
}

/** Split a command-line fragment into argv tokens. Newlines are separators like spaces, so a
 *  multi-line box reads as one command and a user can put each flag on its own line. */
export function parseExtraArgs(input: string): ExtraArgsParse {
  const tokens: string[] = [];
  const quotedIndices = new Set<number>();
  let current = "";
  let started = false;
  // Whether any part of the token being built came from inside quotes.
  let currentQuoted = false;
  let quote: '"' | "'" | null = null;

  for (let i = 0; i < input.length; i += 1) {
    const ch = input[i];

    if (
      quote === null &&
      (ch === " " || ch === "\t" || ch === "\n" || ch === "\r")
    ) {
      if (started) {
        if (currentQuoted) {
          quotedIndices.add(tokens.length);
        }
        tokens.push(current);
        current = "";
        started = false;
        currentQuoted = false;
      }
      continue;
    }

    // A backslash escapes the next character, but only where a shell would: inside single quotes it
    // is literal, which is what makes '\d' usable in a grammar.
    if (ch === "\\" && quote !== "'" && i + 1 < input.length) {
      const next = input[i + 1];
      // Inside double quotes only these are escapes; elsewhere the backslash stands.
      if (quote === '"' && !['"', "\\", "$", "`", "\n"].includes(next)) {
        current += ch;
        started = true;
        continue;
      }
      // A backslash-newline is a line continuation, so pasting a wrapped command works. It contributes
      // nothing, so `started` is left as it was: setting it would make the next line's indentation
      // close an empty token and send an empty positional argument.
      if (next === "\n") {
        i += 1;
        continue;
      }
      current += next;
      started = true;
      i += 1;
      continue;
    }

    if (quote === null && (ch === '"' || ch === "'")) {
      quote = ch;
      // An empty quoted string is still a token: --grammar '' means something.
      started = true;
      currentQuoted = true;
      continue;
    }

    if (quote !== null && ch === quote) {
      quote = null;
      continue;
    }

    current += ch;
    started = true;
  }

  if (started) {
    if (currentQuoted) {
      quotedIndices.add(tokens.length);
    }
    tokens.push(current);
  }
  return { tokens, unterminatedQuote: quote, quotedIndices };
}

/** Render tokens back into one editable line. Round-tripping matters: the stored value is a token
 *  list, so this is what the box shows when the panel reopens. Quote only what has to be
 *  quoted, or every reopen would add another layer of escaping. */
export function formatExtraArgs(
  tokens: readonly string[] | null | undefined,
): string {
  if (tokens === null || tokens === undefined || tokens.length === 0) {
    return "";
  }
  return tokens
    .map((token) => {
      if (token === "") {
        return "''";
      }
      if (!NEEDS_QUOTING.test(token)) {
        return token;
      }
      // Single quotes unless the token contains one, since they escape nothing and leave a grammar or a
      // template readable.
      if (!token.includes("'")) {
        return `'${token}'`;
      }
      return `"${token.replace(DOUBLE_QUOTE_ESCAPES, "\\$1")}"`;
    })
    .join(" ");
}

/** The flag name a token carries, or null when it is a value. Mirrors `_flag_name`. */
export function extraArgFlagName(token: string): string | null {
  const trimmed = token.trim();
  if (!trimmed.startsWith("-") || trimmed === "-" || trimmed === "--") {
    return null;
  }
  // A negative number is a value, not a flag: shorts always start with a letter.
  if (trimmed.length >= 2 && (DIGIT.test(trimmed[1]) || trimmed[1] === ".")) {
    return null;
  }
  let name = trimmed.split("=", 1)[0];
  if (name.startsWith("--")) {
    name = name.replace(UNDERSCORE, "-");
  }
  // Attached `-np8` normalises to `-np`, or a denied flag slips through glued to its value. Mirrors
  // the same branch in _flag_name.
  if (name.length > 3 && name.startsWith("-np")) {
    const suffix = name.slice(3);
    if (
      DIGIT.test(suffix[0]) ||
      (suffix.length > 1 && "-+".includes(suffix[0]) && DIGIT.test(suffix[1]))
    ) {
      return "-np";
    }
  }
  return name;
}

/** Every flag token in a parsed list, in order, deduplicated. */
export function extraArgFlags(tokens: readonly string[]): string[] {
  const seen = new Set<string>();
  for (const token of tokens) {
    const flag = extraArgFlagName(token);
    if (flag !== null) {
      seen.add(flag);
    }
  }
  return [...seen];
}

// Kept in this file rather than beside it: the node test harness resolves value imports at
// runtime with no bundler, so a tested helper importing a sibling by path cannot load.

import type { LlamaFlagCatalog } from "../api/llama-flags";

/** What to tell the user about what they typed, before the load tries it. Three levels: an `error`
 *  is refused by the backend so the load cannot start, a `warning` is allowed through because
 *  we may be unable to verify it, and a `note` is correct usage worth stating. */
export type ExtraArgsDiagnostic = {
  level: "error" | "warning" | "note";
  message: string;
};

/** Managed flags whose supported replacement is a control in this panel. Available while the async
 *  flag catalogue is loading: the backend denies these unconditionally, so they must never
 *  fall through to the "wins" note. */
const MANAGED_CONTROL_FLAGS: Record<string, string> = {
  "--parallel": "Parallel Slots",
  "--n-parallel": "Parallel Slots",
  "-np": "Parallel Slots",
};

/** Pass-through flags that a control in this panel also emits. The backend appends these last and
 *  reconciles the ones that affect its sizing, so the mapping explains which value wins. */
const CONTROL_OWNED_FLAGS: Record<string, string> = {
  "--ctx-size": "Context Length",
  "-c": "Context Length",
  "--batch-size": "Batch Size",
  "-b": "Batch Size",
  "--ubatch-size": "Micro-batch Size",
  "-ub": "Micro-batch Size",
  "--cache-type-k": "KV Cache Dtype",
  "-ctk": "KV Cache Dtype",
  "--cache-type-v": "KV Cache Dtype",
  "-ctv": "KV Cache Dtype",
  "--gpu-layers": "GPU Layers",
  "--n-gpu-layers": "GPU Layers",
  "-ngl": "GPU Layers",
  "--n-cpu-moe": "MoE Layers on CPU",
  "-ncmoe": "MoE Layers on CPU",
  "--split-mode": "Tensor Parallelism",
  "-sm": "Tensor Parallelism",
  "--spec-type": "Speculative Decoding",
  "--spec-draft-n-max": "Draft Tokens",
  "--chat-template": "Chat Template",
  "--chat-template-file": "Chat Template",
  "--load-mode": "Mmap/Mlock",
  "-lm": "Mmap/Mlock",
  // Both halves, since the control sets one dtype for the pair, and both spellings, since which one
  // a build understands is a version question.
  "--spec-draft-type-k": "Spec Decoding KV Cache Dtype",
  "-ctkd": "Spec Decoding KV Cache Dtype",
  "--cache-type-k-draft": "Spec Decoding KV Cache Dtype",
  "--spec-draft-type-v": "Spec Decoding KV Cache Dtype",
  "-ctvd": "Spec Decoding KV Cache Dtype",
  "--cache-type-v-draft": "Spec Decoding KV Cache Dtype",
  "--ctx-checkpoints": "Checkpoints",
  "-ctxcp": "Checkpoints",
  // upstream's older spelling of the same flag
  "--swa-checkpoints": "Checkpoints",
  "--cache-ram": "Cache RAM",
  "-cram": "Cache RAM",
};

/** Flags the launch REMOVES when the GPU picker owns placement. Not a shadow the user wins:
 *  `_strip_device_extra_args` deletes these whenever gpu_ids is set. */
const GPU_SELECTION_STRIPPED_FLAGS: Record<string, string> = {
  "--device": "GPU selection",
  "-dev": "GPU selection",
  "--main-gpu": "GPU selection",
  "-mg": "GPU selection",
};

/** Flags Manual GPU memory REMOVES without translating, unlike the layer count. `/load` calls
 *  strip_shadowing_flags(strip_offload=True) whenever gpu_memory_mode is manual; an -ngl
 *  survives because the route writes it into the first-class field first, while the MoE count
 *  and the fitter are dropped and the controls' own values run. */
const MANUAL_OFFLOAD_STRIPPED_FLAGS: Record<string, string> = {
  "--n-cpu-moe": "MoE Layers on CPU",
  "-ncmoe": "MoE Layers on CPU",
  "--cpu-moe": "MoE Layers on CPU",
  "-cmoe": "MoE Layers on CPU",
  "--fit": "GPU Memory",
  "-fit": "GPU Memory",
};

/** Flags Model Memory removes, per setting. apply_model_memory_policy runs before the extras reach
 *  the command line: "Keep model in GPU memory" emits its own load mode and strips every
 *  other load-mode-bearing flag, since a trailing one resets the whole mode; "Don't reserve
 *  system RAM" drops the flags that would hold a full host copy. */
const KEEP_RESIDENT_STRIPPED_FLAGS: Record<string, string> = {
  "--mlock": "Keep model in GPU memory",
  "-mlock": "Keep model in GPU memory",
  "--load-mode": "Keep model in GPU memory",
  "-lm": "Keep model in GPU memory",
  "--no-mmap": "Keep model in GPU memory",
  "-no-mmap": "Keep model in GPU memory",
  "--mmap": "Keep model in GPU memory",
  "--direct-io": "Keep model in GPU memory",
  "-dio": "Keep model in GPU memory",
  "--no-direct-io": "Keep model in GPU memory",
  "-ndio": "Keep model in GPU memory",
};

/** The subset no-reserve vetoes: mmap and dio hold no full host copy, so they stay. */
const NO_RAM_RESERVE_STRIPPED_FLAGS: Record<string, string> = {
  "--mlock": "Don't reserve system RAM",
  "-mlock": "Don't reserve system RAM",
  "--no-mmap": "Don't reserve system RAM",
  "-no-mmap": "Don't reserve system RAM",
  "--no-direct-io": "Don't reserve system RAM",
  "-ndio": "Don't reserve system RAM",
};

/** Smallest value the backend's own parser accepts, per flag. parse_ctx_override refuses a negative
 *  context; parse_gpu_layers_override accepts -1 and nothing below. The rest are checked for
 *  being integers only. */
const INTEGER_VALUE_MINIMUM: Record<string, number> = {
  "--ctx-size": 0,
  "-c": 0,
  "--gpu-layers": -1,
  "--n-gpu-layers": -1,
  "-ngl": -1,
};


/** Values the backend parses as integers, and refuses the load over. */
/** Flags whose value the backend reads with _last_flag_value, which raises when it is missing or
 *  empty. Not integers, so only presence is checked. */
const VALUE_REQUIRED_FLAGS = new Set([
  "--cache-type-k",
  "-ctk",
  "--cache-type-v",
  "-ctv",
  "--split-mode",
  "-sm",
  "--load-mode",
  "-lm",
  "--spec-draft-type-k",
  "-ctkd",
  "--cache-type-k-draft",
  "--spec-draft-type-v",
  "-ctvd",
  "--cache-type-v-draft",
]);

/** The pass-through spellings of the batch size the floor above applies to. */
const BATCH_SIZE_FLAGS = new Set(["--batch-size", "-b"]);

const INTEGER_VALUE_FLAGS = new Set([
  "--ctx-size",
  "-c",
  "--gpu-layers",
  "--n-gpu-layers",
  "-ngl",
  "--n-cpu-moe",
  "-ncmoe",
  "--parallel",
  "--batch-size",
  "-b",
  "--ubatch-size",
  "-ub",
  "--ctx-checkpoints",
  "-ctxcp",
  "--swa-checkpoints",
  // -1 (no limit) and 0 (disable) are both valid, so only integer-ness is checked
  "--cache-ram",
  "-cram",
]);

/** Sampling belongs to the conversation, not the launch. */
const REQUEST_SCOPED_FLAGS = new Set([
  "--temp",
  "--temperature",
  "--top-p",
  "--top-k",
  "--min-p",
  "--repeat-penalty",
  "--presence-penalty",
  "--frequency-penalty",
  "-n",
  "--predict",
  "--n-predict",
]);

/** The load settings that decide which flags survive to the command line. */
export type ExtraArgsContext = {
  /** The GPU picker owns placement, which removes the device flags. */
  gpuSelectionActive?: boolean;
  /** GPU Memory is Manual, which removes the offload flags its controls own. */
  manualGpuMemory?: boolean;
  /** Smallest --batch-size this launch can run, max(slots, 2). */
  batchFloor?: number;
  /** Model Memory keeps the weights resident, which owns the load mode. */
  keepResident?: boolean;
  /** Model Memory reserves no system RAM, which vetoes the reserving flags. */
  noRamReserve?: boolean;
};

export function diagnoseExtraArgs(
  input: string,
  catalog: LlamaFlagCatalog | null,
  /** What this load's own settings do to the arguments, so the row can say which of them the launch
   *  will remove. An object rather than a run of booleans: a positional list of five would be
   *  read wrong at the call site long before it stopped compiling. */
  context: ExtraArgsContext = {},
): ExtraArgsDiagnostic[] {
  const gpuSelectionActive = context.gpuSelectionActive ?? false;
  const manualGpuMemory = context.manualGpuMemory ?? false;
  const batchFloor = Math.max(2, context.batchFloor ?? 2);
  const keepResident = context.keepResident ?? false;
  const noRamReserve = context.noRamReserve ?? false;
  const out: ExtraArgsDiagnostic[] = [];
  const { tokens, unterminatedQuote, quotedIndices } = parseExtraArgs(input);
  // Tokens the walk below consumed as somebody's value. A quoted token in value POSITION is a value
  // whatever it starts with, since llama.cpp takes the next argv element without looking.
  // Position matters as much as the quotes: a user quoting out of habit still wrote a flag.
  const valueIndices = new Set<number>();

  if (unterminatedQuote) {
    out.push({
      level: "error",
      message: `Unclosed ${unterminatedQuote === '"' ? "double" : "single"} quote.`,
    });
  }
  if (tokens.length > EXTRA_ARGS_MAX_TOKENS) {
    out.push({
      level: "error",
      message: `Too many arguments: ${tokens.length}, limit ${EXTRA_ARGS_MAX_TOKENS}.`,
    });
  }
  // The other half of the backend's bounds: a grammar or JSON schema is one long token, so a payload
  // can sit inside the token cap and still be refused on size. The host's own limit when it has
  // told us one, smaller on Windows, where the whole command line shares a 32767-character budget.
  const maxBytes = catalog?.maxBytes || EXTRA_ARGS_MAX_BYTES;
  const bytes = TEXT_ENCODER.encode(tokens.join("")).length;
  if (bytes > maxBytes) {
    out.push({
      level: "error",
      message: `Arguments are too large: ${bytes} bytes, limit ${maxBytes}.`,
    });
  }
  // And on Windows, what the quoting makes of them: a backslash run before a quote doubles, so bytes
  // alone do not say whether the launch fits.
  if (catalog?.windowsCommandBudget) {
    const quoted = windowsCommandLength(tokens);
    if (quoted > catalog.windowsCommandBudget) {
      out.push({
        level: "error",
        message: `Arguments are too long for a Windows command line: ${quoted} characters after quoting, limit ${catalog.windowsCommandBudget}.`,
      });
    }
  }

  // The backend refuses any token carrying one, and a command copied out of coloured terminal output
  // is the usual way one arrives.
  if (tokens.some((token) => CONTROL_CHARACTERS.test(token))) {
    out.push({
      level: "error",
      message: "Arguments cannot contain control characters.",
    });
  } else if (tokens.some((token) => isUnusableInArgv(token))) {
    // What is left once control characters are ruled out: half a surrogate pair, which pasting a
    // truncated string can produce. Popen raises while encoding argv rather than starting
    // llama-server; an emoji is a whole pair and stays fine.
    out.push({
      level: "error",
      message: "Arguments contain an incomplete character.",
    });
  }

  // A token belonging to no flag. llama-server answers "invalid argument" and refuses to start, and
  // validate_extra_args refuses it outright, so saying so here is the difference between a red
  // line and a failed load. Two-value flags are allowed for.
  let pendingValues = 0;
  // The flag those values are owed to, for the checks below.
  let pendingOwner: string | null = null;
  // A flag left waiting for its value. Reported only when the arity is known: the catalogue read
  // this build's own help, or it is the two-value flag whose arity needs no probe. An
  // unverified flag keeps the benefit of the doubt.
  const owedValues: string[] = [];
  const noteOwed = (owner: string | null, pending: number) => {
    if (owner === null || pending <= 0) {
      return;
    }
    if (
      TWO_VALUE_FLAGS.has(owner) ||
      (catalog?.probeOk === true &&
        owner in catalog.flags &&
        !catalog.switches.has(owner))
    ) {
      owedValues.push(owner);
    }
  };
  for (const [index, token] of tokens.entries()) {
    const quotedValue = pendingValues > 0 && quotedIndices.has(index);
    const flag = quotedValue ? null : extraArgFlagName(token);
    if (flag === null) {
      if (pendingValues <= 0) {
        out.push({
          level: "error",
          message: `"${token}" belongs to no flag. Every value has to follow its flag.`,
        });
        break;
      }
      valueIndices.add(index);
      pendingValues -= 1;
      if (pendingValues <= 0) {
        pendingOwner = null;
      }
      continue;
    }
    // Before the obligation is replaced: "--numa --verbose" leaves --numa without the value it needs,
    // and only this point in the walk still knows that.
    noteOwed(pendingOwner, pendingValues);
    const attached = valueIsAttached(token, flag);
    pendingValues = TWO_VALUE_FLAGS.has(flag)
      ? // An attached value is ONE of the two, so the option still owes its END.
        attached
        ? 1
        : 2
      : OPTIONAL_SECOND_VALUE_FLAGS.has(flag)
        ? // Allowed, not owed: the second token is taken if it is there.
          attached
          ? 1
          : 2
        : attached
          ? 0
          // A flag THIS build documents as taking no value claims none, so the next bare token has
          // no owner. Only when the catalogue actually knows the flag.
          :
            catalog?.switches.has(flag)
            ? 0
            : 1;
    // Never reported as owing a value: only the flags whose arity is certain are.
    pendingOwner =
      pendingValues > 0 && !OPTIONAL_SECOND_VALUE_FLAGS.has(flag) ? flag : null;
  }
  noteOwed(pendingOwner, pendingValues);

  const seen = new Set<string>();
  const unknown: string[] = [];
  const shadowed: string[] = [];
  const stripped: string[] = [];
  const manualStripped: string[] = [];
  const memoryStripped: [string, string][] = [];
  const reportedValues = new Set<string>();
  for (const [index, token] of tokens.entries()) {
    // Judged the same way the walk above judged it, or a quoted value beginning with a hyphen would be
    // reported here as an unknown flag.
    const flag = valueIndices.has(index) ? null : extraArgFlagName(token);
    if (flag === null) {
      continue;
    }
    // Before the de-duplication below, because llama.cpp reads the LAST occurrence: in `-ngl 20 -ngl
    // many` it is the second one the backend refuses.
    if (INTEGER_VALUE_FLAGS.has(flag) || VALUE_REQUIRED_FLAGS.has(flag)) {
      const attached = valueIsAttached(token, flag);
      const value = attached ? token.split("=")[1] : tokens[index + 1];
      // A flag whose value is the next token has none when that token is itself a flag: `--ctx-size
      // --numa` reads --numa as the value in a shell and as a missing one here, as the parser does.
      const missing =
        value === undefined ||
        value === "" ||
        (!attached &&
          !valueIndices.has(index + 1) &&
          extraArgFlagName(value) !== null);
      const minimum = INTEGER_VALUE_MINIMUM[flag];
      const numeric = INTEGER_VALUE_FLAGS.has(flag);
      let message: string | null = null;
      if (missing) {
        message = numeric
          ? `${flag} needs a number after it.`
          : `${flag} needs a value after it.`;
      } else if (!numeric) {
        message = null;
      } else if (!INTEGER.test(value.trim())) {
        message = `${flag} takes a number, and "${value}" is not one.`;
      } else if (minimum !== undefined && Number(value.trim()) < minimum) {
        message =
          minimum === 0
            ? `${flag} cannot be negative.`
            : `${flag} takes ${minimum} or more.`;
      } else if (
        BATCH_SIZE_FLAGS.has(flag) &&
        Number(value.trim()) < Math.max(2, batchFloor)
      ) {
        // Not raised the way the first-class control is: this flag is appended after the launcher's own
        // --batch-size and wins it, so the load starts and llama-server aborts on the assertion.
        const floor = Math.max(2, batchFloor);
        message =
          floor > 2
            ? `${flag} takes ${floor} or more here: llama-server aborts on a batch below the ${floor} parallel slot(s) it serves.`
            : `${flag} takes 2 or more: llama-server aborts on a batch of 1.`;
      }
      if (message !== null && !reportedValues.has(message)) {
        reportedValues.add(message);
        out.push({ level: "error", message });
      }
    }
    if (seen.has(flag)) {
      continue;
    }
    seen.add(flag);
    const managedControl = MANAGED_CONTROL_FLAGS[flag];
    const isManaged =
      managedControl !== undefined || Boolean(catalog?.managed.has(flag));

    if (token !== token.trim()) {
      // The padding is part of the token llama.cpp looks up, so a quoted "--top-k " never arrives as a
      // flag: it answers "error: invalid argument: --top-k", naming a flag that looks correct in
      // the log. Reported whether or not a control owns the flag.
      out.push({
        level: "error",
        message: `Remove the spaces around "${token}". llama-server reads them as part of the flag.`,
      });
    } else if (token.includes("=") && !isManaged) {
      // Measured on b10342 and b10360: "--top-k=20", "--ctx-size=4096" and "--flash-attn=on" each exit
      // with "error: invalid argument". A managed flag is left to the message below.
      out.push({
        level: "error",
        message: `llama-server does not read "${flag}=value". Write ${flag} and its value as two arguments.`,
      });
    }

    if (isManaged) {
      const control = managedControl ?? CONTROL_OWNED_FLAGS[flag];
      out.push({
        level: "error",
        message: control
          ? `${flag} is set by ${control} above and cannot be passed here.`
          : `${flag} is managed by Unsloth and cannot be passed here.`,
      });
      continue;
    }
    if (gpuSelectionActive && GPU_SELECTION_STRIPPED_FLAGS[flag]) {
      stripped.push(flag);
      continue;
    }
    if (manualGpuMemory && MANUAL_OFFLOAD_STRIPPED_FLAGS[flag]) {
      manualStripped.push(flag);
      continue;
    }
    // No-reserve first: with both settings on it is the one that runs, and its own veto is what removes the flag.
    if (noRamReserve && NO_RAM_RESERVE_STRIPPED_FLAGS[flag]) {
      memoryStripped.push([flag, NO_RAM_RESERVE_STRIPPED_FLAGS[flag]]);
      continue;
    }
    if (keepResident && !noRamReserve && KEEP_RESIDENT_STRIPPED_FLAGS[flag]) {
      memoryStripped.push([flag, KEEP_RESIDENT_STRIPPED_FLAGS[flag]]);
      continue;
    }
    const control = CONTROL_OWNED_FLAGS[flag];
    if (control) {
      shadowed.push(`${flag} (${control})`);
      continue;
    }
    if (REQUEST_SCOPED_FLAGS.has(flag)) {
      out.push({
        level: "note",
        message: `${flag} only sets a default here. Sampling for a conversation lives in its chat settings.`,
      });
      continue;
    }
    // Only when the catalogue was actually read: a build we could not probe must not have every one of
    // its flags called a typo.
    if (catalog?.probeOk && !(flag in catalog.flags)) {
      unknown.push(flag);
    }
  }

  if (stripped.length > 0) {
    out.push({
      level: "warning",
      message: `${stripped.join(", ")} will be removed: the GPU selection above owns placement. Set GPU Memory to Default to pass it yourself.`,
    });
  }
  for (const owner of owedValues) {
    const message = TWO_VALUE_FLAGS.has(owner)
      ? `${owner} needs two values, a start and an end layer.`
      : INTEGER_VALUE_FLAGS.has(owner)
        ? `${owner} needs a number after it.`
        : `${owner} needs a value after it.`;
    if (!reportedValues.has(message)) {
      reportedValues.add(message);
      out.push({ level: "error", message });
    }
  }

  if (memoryStripped.length > 0) {
    const setting = memoryStripped[0][1];
    out.push({
      level: "warning",
      message: `${memoryStripped.map(([flag]) => flag).join(", ")} will be removed: ${setting} in Settings owns how the weights are held.`,
    });
  }
  if (manualStripped.length > 0) {
    out.push({
      level: "warning",
      message: `${manualStripped.join(", ")} will be removed: GPU Memory is Manual, and its controls own offload. Set GPU Memory to Default to pass it yourself.`,
    });
  }
  if (shadowed.length > 0) {
    out.push({
      level: "note",
      message: `Passed after the controls above, so ${shadowed.join(", ")} wins.`,
    });
  }
  if (unknown.length > 0) {
    out.push({
      level: "warning",
      message:
        unknown.length === 1
          ? `${unknown[0]} is not in this llama-server's --help. It will still be passed.`
          : `${unknown.join(", ")} are not in this llama-server's --help. They will still be passed.`,
    });
  }
  return out;
}

/** True when nothing here would stop the load. */
export function extraArgsAreLoadable(
  diagnostics: readonly ExtraArgsDiagnostic[],
): boolean {
  return !diagnostics.some((d) => d.level === "error");
}
