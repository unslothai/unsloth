// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Not a shell: quotes and backslashes exist only so a template or grammar can hold spaces. */

// Hoisted: these run per token, and biome flags a literal in a hot path.
const NEEDS_QUOTING = /[\s"'\\]/;
const DOUBLE_QUOTE_ESCAPES = /(["\\$`])/g;
const DIGIT = /[0-9]/;
const UNDERSCORE = /_/g;
// biome-ignore lint/suspicious/noControlCharactersInRegex: mirroring the backend's own check
// Same rule as CONTROL_IN_ARGV below and the backend's check.
const CONTROL_CHARACTERS = /[\u0000-\u0008\u000b-\u001f]/;
const INTEGER = /^-?[0-9]+$/;
/** C0 controls except tab and newline, matching backend _has_control_characters. */
// biome-ignore lint/suspicious/noControlCharactersInRegex: that is exactly what this finds
const CONTROL_IN_ARGV = /[\u0000-\u0008\u000B-\u001F]/;

/** Walks the string because a class would match both halves of a valid pair; only unpaired
 *  surrogates are refused, as Popen does. */
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
const TEXT_ENCODER = new TextEncoder();

/** Mirrors MAX_EXTRA_ARGS_BYTES in llama_server_args.py. */
export const EXTRA_ARGS_MAX_BYTES = 32 * 1024;
/** Mirrors MAX_EXTRA_ARG_TOKENS in llama_server_args.py. */
export const EXTRA_ARGS_MAX_TOKENS = 256;

const TWO_VALUE_FLAGS = new Set(["--control-vector-layer-range"]);

/** Newer llama.cpp writes the scale into the value, older took a separate token; allowed, not required. */
const OPTIONAL_SECOND_VALUE_FLAGS = new Set([
  "--lora-scaled",
  "--control-vector-scaled",
]);

export type ExtraArgsParse = {
  tokens: string[];
  unterminatedQuote: '"' | "'" | null;
  /** Quoted token indices: `--chat-template '- hello'` must not read the value as a flag. */
  quotedIndices: ReadonlySet<number>;
};

/** Port of CPython list2cmdline (used by Popen): backslash runs before a quote double. */
export function windowsCommandLength(tokens: readonly string[]): number {
  const result: string[] = [];
  for (const token of tokens) {
    if (result.length > 0) {
      result.push(" ");
    }
    // Quoted only for whitespace or emptiness, as CPython does.
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
      // Again inside quotes, or the run would escape the closing one.
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

/** Only "=" and attached shorts like -np8 carry a value; the folded name must not be compared.
 *  Mirrors _value_is_attached. */
function valueIsAttached(token: string, flag: string): boolean {
  const raw = token.trim();
  if (raw.includes("=")) {
    return true;
  }
  return raw.replace(UNDERSCORE, "-") !== flag;
}

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

/** Mirrors drop_managed_flags so an install upgraded across a denylist change still loads; a flag
 *  never outlives its value (an orphan reads as the model path). */
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
      continue;
    }
    kept.push(token);
  }
  // Drop shapes validate_extra_args refuses outright: ownerless tokens and half-written two-value options.
  let bounded = dropUnvalidatableTokens(dropUnusableValues(kept));
  // Shed from the tail with the host's limits (Windows takes 24 KiB, not 32).
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
    // extraArgFlagName folds underscores, so check attachment rather than the normalized spelling.
    if (last !== undefined && lastFlag !== null && !valueIsAttached(last, lastFlag)) {
      bounded.pop();
    }
    // Re-applied after every cut, as the backend re-validates.
    bounded = dropUnvalidatableTokens(dropUnusableValues(bounded));
  }
  return bounded;
}

/** Removes flags whose value the backend parsers refuse (a 400), keeping the rest of a legacy list. */
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
      // false: a stored list's mode is unknown, so drop only what no mode can run.
      (RATIO_VALUE_FLAGS.has(flag) &&
        ratioValueProblem(flag, value, false) !== null) ||
      (INTEGER_VALUE_FLAGS.has(flag) &&
        (!INTEGER.test(value.trim()) ||
          (minimum !== undefined && Number(value.trim()) < minimum)));
    if (!unusable) {
      out.push(token);
      continue;
    }
    // The value goes too, or llama-server reads it as a model path.
    skipNext = !attached && !missing;
  }
  return out;
}

/** Drops ownerless tokens (read as model path) and incomplete two-value options. */
function dropUnvalidatableTokens(tokens: readonly string[]): string[] {
  const out: string[] = [];
  let pending = 0;
  let twoValuePending = 0;
  let droppedOwes = 0;
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
        // Matches drop_managed_flags.
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
      out.length = ownerAt;
    }
    if (token.includes("=")) {
      // llama.cpp looks up the whole token, so "--top-k=20" is an unknown option.
      pending = 0;
      twoValuePending = 0;
      ownerAt = -1;
      continue;
    }
    if (token !== token.trim()) {
      // Padding is part of the looked-up token; drop its value too or it becomes positional.
      droppedOwes = valueIsAttached(token, flag) ? 0 : 1;
      pending = 0;
      twoValuePending = 0;
      ownerAt = -1;
      continue;
    }
    const attached = valueIsAttached(token, flag);
    if (TWO_VALUE_FLAGS.has(flag)) {
      pending = attached ? 1 : 2;
      twoValuePending = pending;
      ownerAt = out.length;
    } else if (OPTIONAL_SECOND_VALUE_FLAGS.has(flag)) {
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

/** Newlines separate like spaces, so each flag may sit on its own line. */
export function parseExtraArgs(input: string): ExtraArgsParse {
  const tokens: string[] = [];
  const quotedIndices = new Set<number>();
  let current = "";
  let started = false;
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

    // Backslash is literal inside single quotes, which keeps '\d' usable in a grammar.
    if (ch === "\\" && quote !== "'" && i + 1 < input.length) {
      const next = input[i + 1];
      if (quote === '"' && !['"', "\\", "$", "`", "\n"].includes(next)) {
        current += ch;
        started = true;
        continue;
      }
      // Line continuation; leave `started` alone or the next line's indent emits an empty token.
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
      // An empty quoted string is still a token.
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

/** Must round-trip: quote only what needs it, or each reopen adds escaping. */
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
      if (!token.includes("'")) {
        return `'${token}'`;
      }
      return `"${token.replace(DOUBLE_QUOTE_ESCAPES, "\\$1")}"`;
    })
    .join(" ");
}

/** Mirrors `_flag_name`. */
export function extraArgFlagName(token: string): string | null {
  const trimmed = token.trim();
  if (!trimmed.startsWith("-") || trimmed === "-" || trimmed === "--") {
    return null;
  }
  // A negative number is a value: shorts always start with a letter.
  if (trimmed.length >= 2 && (DIGIT.test(trimmed[1]) || trimmed[1] === ".")) {
    return null;
  }
  let name = trimmed.split("=", 1)[0];
  if (name.startsWith("--")) {
    name = name.replace(UNDERSCORE, "-");
  }
  // Attached `-np8` normalises to `-np`, or a denied flag slips through. Mirrors _flag_name.
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

// Kept here: the node test harness has no bundler, so a tested helper cannot import a sibling.

import type { LlamaFlagCatalog } from "../api/llama-flags";

/** `error` blocks the load, `warning` is unverifiable, `note` is correct usage worth stating. */
export type ExtraArgsDiagnostic = {
  level: "error" | "warning" | "note";
  message: string;
};

/** Hard-coded since the backend always denies these, even before the async catalogue loads. */
const MANAGED_CONTROL_FLAGS: Record<string, string> = {
  "--parallel": "Parallel Slots",
  "--n-parallel": "Parallel Slots",
  "-np": "Parallel Slots",
};

/** The backend appends these last and reconciles sizing ones, so this explains which value wins. */
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
  // Both halves and both spellings, since builds differ.
  "--spec-draft-type-k": "Spec Decoding KV Cache Dtype",
  "-ctkd": "Spec Decoding KV Cache Dtype",
  "--cache-type-k-draft": "Spec Decoding KV Cache Dtype",
  "--spec-draft-type-v": "Spec Decoding KV Cache Dtype",
  "-ctvd": "Spec Decoding KV Cache Dtype",
  "--cache-type-v-draft": "Spec Decoding KV Cache Dtype",
  "--ctx-checkpoints": "Checkpoints",
  "-ctxcp": "Checkpoints",
  // Older upstream spelling.
  "--swa-checkpoints": "Checkpoints",
  "--cache-ram": "Cache RAM",
  "-cram": "Cache RAM",
};

/** `_strip_device_extra_args` deletes these whenever gpu_ids is set. */
const GPU_SELECTION_STRIPPED_FLAGS: Record<string, string> = {
  "--device": "GPU selection",
  "-dev": "GPU selection",
  "--main-gpu": "GPU selection",
  "-mg": "GPU selection",
};

/** strip_shadowing_flags(strip_offload=True) drops these in manual mode; -ngl is promoted instead. */
const MANUAL_OFFLOAD_STRIPPED_FLAGS: Record<string, string> = {
  "--n-cpu-moe": "MoE Layers on CPU",
  "-ncmoe": "MoE Layers on CPU",
  "--cpu-moe": "MoE Layers on CPU",
  "-cmoe": "MoE Layers on CPU",
  "--fit": "GPU Memory",
  "-fit": "GPU Memory",
};

/** apply_model_memory_policy strips these: keep-resident owns the load mode. */
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

/** mmap and dio hold no full host copy, so they stay. */
const NO_RAM_RESERVE_STRIPPED_FLAGS: Record<string, string> = {
  "--mlock": "Don't reserve system RAM",
  "-mlock": "Don't reserve system RAM",
  "--no-mmap": "Don't reserve system RAM",
  "-no-mmap": "Don't reserve system RAM",
  "--no-direct-io": "Don't reserve system RAM",
  "-ndio": "Don't reserve system RAM",
};

/** Mirrors parse_ctx_override / parse_gpu_layers_override minimums. */
const INTEGER_VALUE_MINIMUM: Record<string, number> = {
  "--ctx-size": 0,
  "-c": 0,
  "--gpu-layers": -1,
  "--n-gpu-layers": -1,
  "-ngl": -1,
};


/** Read with _last_flag_value, which raises when missing or empty. */
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
  // Read with _last_flag_value, so a bare -ts is a 400.
  "--tensor-split",
  "-ts",
]);

const GPU_LAYERS_FLAGS = new Set(["--gpu-layers", "--n-gpu-layers", "-ngl"]);

/** Mirrors parse_gpu_layers_override only for: is the resolved count non-negative. */
function lastIntegerFlagValue(
  tokens: readonly string[],
  flags: ReadonlySet<string>,
): number | null {
  let found: number | null = null;
  for (const [index, token] of tokens.entries()) {
    const flag = extraArgFlagName(token);
    if (flag === null || !flags.has(flag)) {
      continue;
    }
    const attached = valueIsAttached(token, flag);
    const raw = attached ? token.split("=")[1] : tokens[index + 1];
    if (raw !== undefined && INTEGER.test(raw.trim())) {
      found = Number(raw.trim());
    }
  }
  return found;
}

/** See parse_tensor_split_override. */
const RATIO_VALUE_FLAGS = new Set(["--tensor-split", "-ts"]);

/** llama.cpp splits on this class, so "3/1" is "3,1". */
const RATIO_DELIMITER = /[,/]+/;

const NON_FINITE = /^[+-]?(nan|inf(inity)?)$/i;

/** Python float() grammar: Number() accepts 0x/0b/0o and refuses PEP 515 `1_0`. */
const PY_FLOAT =
  /^[+-]?(?:(?:\d(?:_?\d)*)?\.\d(?:_?\d)*|\d(?:_?\d)*\.?)(?:[eE][+-]?\d(?:_?\d)*)?$/;

/** std::stof throws out_of_range above this. */
const FLOAT32_MAX = 3.4028234663852886e38;

/** libstdc++ std::stof reports subnormals as ERANGE too; exactly 0 is fine. */
const FLOAT32_MIN_NORMAL = 1.1754943508222875e-38;

const toFloat32 = (value: number): number => Math.fround(value);

/** The manual launcher writes f"{x:g}" (six digits), so judge the text the child parses. */
const asEmitted = (value: number): number => Number(value.toPrecision(6));

function ratioValueProblem(
  flag: string,
  value: string,
  reserialized: boolean,
): string | null {
  const parts = value
    .split(RATIO_DELIMITER)
    .filter((part) => part.trim() !== "");
  if (parts.length === 0) {
    return `${flag} takes a comma- or slash-separated list of numbers.`;
  }
  const trimmed = parts.map((part) => part.trim());
  // Readability follows Python's grammar, never Number().
  if (trimmed.some((part) => !PY_FLOAT.test(part) && !NON_FINITE.test(part))) {
    return `${flag} takes a comma- or slash-separated list of numbers, and "${value}" is not one.`;
  }
  const numbers = trimmed.map((part) => Number(part.replace(/_/g, "")));
  if (numbers.some((entry) => !Number.isFinite(entry) || entry < 0)) {
    return `${flag} entries must be finite and non-negative.`;
  }
  if (numbers.reduce((total, entry) => total + entry, 0) <= 0) {
    return `${flag} must have a positive total.`;
  }
  // Pass-through keeps the user's text; only manual mode rewrites to six digits.
  const shares = numbers.map((entry) =>
    toFloat32(reserialized ? asEmitted(entry) : entry),
  );
  if (shares.some((share) => !Number.isFinite(share))) {
    return `${flag} entries must fit in a 32-bit float (at most ${FLOAT32_MAX.toExponential(4)}).`;
  }
  if (
    shares.some(
      (share, at) => numbers[at] !== 0 && share < FLOAT32_MIN_NORMAL,
    )
  ) {
    return `${flag} entries must be 0 or at least ${FLOAT32_MIN_NORMAL.toExponential(4)}.`;
  }
  // Float32 step-by-step prefix sum, as llama.cpp does; a float64 reduction disagrees near the top.
  let running = 0;
  for (const share of shares) {
    running = toFloat32(running + share);
    if (!Number.isFinite(running)) {
      return `${flag} adds up past the 32-bit float range.`;
    }
  }
  return null;
}

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
  // -1 (no limit) and 0 (disable) are both valid.
  "--cache-ram",
  "-cram",
]);

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

export type ExtraArgsContext = {
  gpuSelectionActive?: boolean;
  manualGpuMemory?: boolean;
  /** Only manual mode with a resolved count >= 0 rewrites --tensor-split; at Auto it is dropped. */
  gpuLayers?: number;
  batchFloor?: number;
  keepResident?: boolean;
  noRamReserve?: boolean;
};

export function diagnoseExtraArgs(
  input: string,
  catalog: LlamaFlagCatalog | null,
  context: ExtraArgsContext = {},
): ExtraArgsDiagnostic[] {
  const gpuSelectionActive = context.gpuSelectionActive ?? false;
  const manualGpuMemory = context.manualGpuMemory ?? false;
  const batchFloor = Math.max(2, context.batchFloor ?? 2);
  const keepResident = context.keepResident ?? false;
  const noRamReserve = context.noRamReserve ?? false;
  const out: ExtraArgsDiagnostic[] = [];
  const { tokens, unterminatedQuote, quotedIndices } = parseExtraArgs(input);
  // Mirrors _should_strip_tensor_split: an extras -ngl is promoted first.
  const nglOverride = lastIntegerFlagValue(tokens, GPU_LAYERS_FLAGS);
  const resolvedGpuLayers = nglOverride ?? context.gpuLayers ?? -1;
  const reserializesSplit = manualGpuMemory && resolvedGpuLayers >= 0;
  // A token in value position is a value whatever it starts with; llama.cpp takes the next argv blindly.
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
  // A grammar or schema can fit the token cap yet exceed the byte cap (smaller on Windows).
  const maxBytes = catalog?.maxBytes || EXTRA_ARGS_MAX_BYTES;
  const bytes = TEXT_ENCODER.encode(tokens.join("")).length;
  if (bytes > maxBytes) {
    out.push({
      level: "error",
      message: `Arguments are too large: ${bytes} bytes, limit ${maxBytes}.`,
    });
  }
  // Windows quoting doubles backslash runs, so bytes alone do not say whether it fits.
  if (catalog?.windowsCommandBudget) {
    const quoted = windowsCommandLength(tokens);
    if (quoted > catalog.windowsCommandBudget) {
      out.push({
        level: "error",
        message: `Arguments are too long for a Windows command line: ${quoted} characters after quoting, limit ${catalog.windowsCommandBudget}.`,
      });
    }
  }

  if (tokens.some((token) => CONTROL_CHARACTERS.test(token))) {
    out.push({
      level: "error",
      message: "Arguments cannot contain control characters.",
    });
  } else if (tokens.some((token) => isUnusableInArgv(token))) {
    // Unpaired surrogate (truncated paste): Popen raises encoding argv. Emoji pairs are fine.
    out.push({
      level: "error",
      message: "Arguments contain an incomplete character.",
    });
  }

  // Ownerless tokens: llama-server refuses to start and validate_extra_args refuses them.
  let pendingValues = 0;
  let pendingOwner: string | null = null;
  // Reported only when arity is known; unverified flags get the benefit of the doubt.
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
    // Before the obligation is replaced: "--numa --verbose" leaves --numa owed.
    noteOwed(pendingOwner, pendingValues);
    const attached = valueIsAttached(token, flag);
    pendingValues = TWO_VALUE_FLAGS.has(flag)
      ?
        attached
        ? 1
        : 2
      : OPTIONAL_SECOND_VALUE_FLAGS.has(flag)
        ?
          attached
          ? 1
          : 2
        : attached
          ? 0
          // A flag the catalogue documents as valueless claims no next token.
          :
            catalog?.switches.has(flag)
            ? 0
            : 1;
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
    // Judged as the walk above did, or a quoted hyphen value reads as an unknown flag.
    const flag = valueIndices.has(index) ? null : extraArgFlagName(token);
    if (flag === null) {
      continue;
    }
    // Before de-duplication: llama.cpp reads the LAST occurrence, so `-ngl 20 -ngl many` must 400 here.
    if (INTEGER_VALUE_FLAGS.has(flag) || VALUE_REQUIRED_FLAGS.has(flag)) {
      const attached = valueIsAttached(token, flag);
      const value = attached ? token.split("=")[1] : tokens[index + 1];
      // `--ctx-size --numa` reads as a missing value, as the backend parser does.
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
      } else if (RATIO_VALUE_FLAGS.has(flag)) {
        message = ratioValueProblem(flag, value, reserializesSplit);
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
        // Appended after the launcher's --batch-size so it wins; llama-server then aborts on the assertion.
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
      // Padding is part of the looked-up token, so a quoted "--top-k " fails as an invalid argument.
      out.push({
        level: "error",
        message: `Remove the spaces around "${token}". llama-server reads them as part of the flag.`,
      });
    } else if (token.includes("=") && !isManaged) {
      // llama.cpp rejects "--flag=value" as an invalid argument (e.g. "--top-k=20").
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
    // No-reserve first: with both settings on, its veto is the one that removes the flag.
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
    // Only when the catalogue was read, so an unprobed build's flags are not all called typos.
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

export function extraArgsAreLoadable(
  diagnostics: readonly ExtraArgsDiagnostic[],
): boolean {
  return !diagnostics.some((d) => d.level === "error");
}
