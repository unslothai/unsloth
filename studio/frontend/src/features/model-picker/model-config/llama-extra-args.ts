// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Turning what the user types into the argv tokens the API takes.
 *
 * `LoadRequest.llama_extra_args` is a list with one argv token per entry, which is
 * what the CLI already sends and what the overrides table already stores. The
 * single string only ever lives in the control, so the split happens here.
 *
 * Deliberately not a shell: there is no `sh -c` anywhere on this path (the child is
 * spawned from a list), so expanding `$HOME`, globbing, or honouring `;` and `|`
 * would invent meanings the backend does not implement. Quotes and backslashes are
 * handled because a chat template or a grammar genuinely needs a space inside one
 * token; everything else is a literal character.
 */

// Hoisted: these run per token, and biome flags a literal in a hot path.
const NEEDS_QUOTING = /[\s"'\\]/;
const DOUBLE_QUOTE_ESCAPES = /(["\\$`])/g;
const DIGIT = /[0-9]/;
const UNDERSCORE = /_/g;
// biome-ignore lint/suspicious/noControlCharactersInRegex: mirroring the backend's own check
const CONTROL_CHARACTERS = /[\u0000-\u0008\u000b-\u001f\u007f]/;
const INTEGER = /^-?[0-9]+$/;
/**
 * Characters execve cannot carry: a NUL or any other control character, and an
 * unpaired surrogate, which the backend refuses because Popen raises while encoding
 * argv rather than starting llama-server.
 */
// biome-ignore lint/suspicious/noControlCharactersInRegex: that is exactly what this finds
const UNUSABLE_IN_ARGV = /[\u0000-\u0008\u000A-\u001F\u007F]|[\uD800-\uDFFF]/;
// Hoisted for the same reason as the patterns above: this runs on every keystroke.
const TEXT_ENCODER = new TextEncoder();

/** Longest input the editor will parse, mirroring MAX_EXTRA_ARGS_BYTES in llama_server_args.py. */
export const EXTRA_ARGS_MAX_BYTES = 32 * 1024;
/** Mirrors MAX_EXTRA_ARG_TOKENS in llama_server_args.py. */
export const EXTRA_ARGS_MAX_TOKENS = 256;

export type ExtraArgsParse = {
  tokens: string[];
  /** Set when a quote is left open, so the row can say so instead of silently dropping it. */
  unterminatedQuote: '"' | "'" | null;
};

/**
 * A stored list with the flags this build refuses removed, values and all.
 *
 * The panel hydrates from the stored override and then sends what it holds as an
 * explicit list, which /load validates strictly rather than putting through the
 * carry-over paths that drop such a flag quietly. An install upgraded across a
 * denylist change would therefore fail to load a model that worked the day before.
 * Mirrors drop_managed_flags: a flag takes its value with it, or llama-server reads
 * the orphan as a positional model path.
 */
/** Whether this token's value is the NEXT token rather than part of itself. */
function takesNextToken(
  token: string,
  flag: string,
  next: string | undefined,
): boolean {
  if (token.includes("=") || flag !== token.trim()) {
    return false;
  }
  return next !== undefined && extraArgFlagName(next) === null;
}

/**
 * A stored list reduced to what THIS build would accept.
 *
 * Mirrors drop_managed_flags: denied flags go, tokens carrying control characters or
 * unpaired surrogates go, anything past the size bounds goes, and a flag never
 * outlives the value that went with it (an orphaned value is a bare positional,
 * which llama-server reads as the model path).
 *
 * The panel needs this because hydrating turns a stored list into an EXPLICIT
 * request, which /load validates strictly instead of putting it through the very
 * carry-over paths that exist to drop such a token quietly. Without it, an install
 * upgraded across any of those rules stops loading a model that worked yesterday.
 */
export function sanitizeStoredExtraArgs(
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
    const next = tokens[index + 1];
    if (flag !== null && managed.has(flag)) {
      skipNext = takesNextToken(token, flag, next);
      continue;
    }
    if (UNUSABLE_IN_ARGV.test(token)) {
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
      UNUSABLE_IN_ARGV.test(next)
    ) {
      // Its value is about to be dropped, so the flag goes with it.
      continue;
    }
    kept.push(token);
  }
  // Then the bounds, shed from the tail, never leaving a flag without its value.
  while (
    kept.length > EXTRA_ARGS_MAX_TOKENS ||
    TEXT_ENCODER.encode(kept.join("")).length > EXTRA_ARGS_MAX_BYTES
  ) {
    kept.pop();
    const last = kept[kept.length - 1];
    if (
      last !== undefined &&
      extraArgFlagName(last) === last.trim() &&
      !last.includes("=")
    ) {
      kept.pop();
    }
  }
  return kept;
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

/**
 * Split a command-line fragment into argv tokens.
 *
 * Newlines are separators like spaces, so a multi-line box reads as one command and
 * a user can put each flag on its own line.
 */
export function parseExtraArgs(input: string): ExtraArgsParse {
  const tokens: string[] = [];
  let current = "";
  let started = false;
  let quote: '"' | "'" | null = null;

  for (let i = 0; i < input.length; i += 1) {
    const ch = input[i];

    if (
      quote === null &&
      (ch === " " || ch === "\t" || ch === "\n" || ch === "\r")
    ) {
      if (started) {
        tokens.push(current);
        current = "";
        started = false;
      }
      continue;
    }

    // A backslash escapes the next character, but only where a shell would: inside
    // single quotes it is literal, which is what makes '\d' usable in a grammar.
    if (ch === "\\" && quote !== "'" && i + 1 < input.length) {
      const next = input[i + 1];
      // Inside double quotes only these are escapes; elsewhere the backslash stands.
      if (quote === '"' && !['"', "\\", "$", "`", "\n"].includes(next)) {
        current += ch;
        started = true;
        continue;
      }
      // A backslash-newline is a line continuation, so pasting a wrapped command
      // works. It contributes nothing, so `started` is left as it was: setting it
      // would make the indentation on the next line close an empty token, and a
      // wrapped command indented under its first line (the usual shape) would send
      // an empty positional argument that llama-server reads as a model path.
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
    tokens.push(current);
  }
  return { tokens, unterminatedQuote: quote };
}

/**
 * Render tokens back into one editable line.
 *
 * Round-tripping matters: the stored value is a token list, so this is what the box
 * shows when the panel reopens. Quote only what has to be quoted, or every reopen
 * would add another layer of escaping to the user's own text.
 */
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
      // Single quotes unless the token contains one, since they escape nothing and
      // leave a grammar or a template readable.
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
  // Attached `-np8` normalises to `-np`, or a denied flag slips through glued to
  // its value. Mirrors the same branch in _flag_name.
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

// --- diagnostics ------------------------------------------------------------
// Kept in this file rather than beside it: the node test harness resolves value
// imports at runtime with no bundler, so a tested helper importing a sibling by
// extensionless path cannot load. Every other tested module here is self-contained
// for the same reason.

import type { LlamaFlagCatalog } from "../api/llama-flags";

/**
 * What to tell the user about what they typed, before the load tries it.
 *
 * Three levels, and the difference is the whole point of the row: an `error` is
 * refused by the backend so the load cannot start, a `warning` is allowed through
 * because we may simply be unable to verify it, and a `note` is correct usage worth
 * stating out loud.
 */
export type ExtraArgsDiagnostic = {
  level: "error" | "warning" | "note";
  message: string;
};

/**
 * Controls in this panel that emit the same flag. Not a denial: the backend appends
 * extras last and reconciles the ones that move its own sizing (`parse_ctx_override`
 * and friends exist for exactly that), and the CLI has always allowed it. The user
 * just deserves to be told which one wins.
 */
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
};

/**
 * Flags the launch REMOVES when the GPU picker owns placement. Not a shadow the user
 * wins: `_strip_device_extra_args` deletes these from the command whenever gpu_ids is
 * set, so telling the reader theirs is taken from here would be false.
 */
const GPU_SELECTION_STRIPPED_FLAGS: Record<string, string> = {
  "--device": "GPU selection",
  "-dev": "GPU selection",
  "--main-gpu": "GPU selection",
  "-mg": "GPU selection",
};

/**
 * Smallest value the backend's own parser accepts, per flag.
 *
 * parse_ctx_override refuses a negative context; parse_gpu_layers_override accepts
 * -1 (all layers) and nothing below it. The rest are checked for being integers
 * only, because that is all those parsers claim.
 */
const INTEGER_VALUE_MINIMUM: Record<string, number> = {
  "--ctx-size": 0,
  "-c": 0,
  "--gpu-layers": -1,
  "--n-gpu-layers": -1,
  "-ngl": -1,
};

/** Values the backend parses as integers, and refuses the load over. */
/**
 * Flags whose value the backend reads with _last_flag_value, which raises when it is
 * missing or empty. Not integers, so only presence is checked here.
 */
const VALUE_REQUIRED_FLAGS = new Set([
  "--cache-type-k",
  "-ctk",
  "--cache-type-v",
  "-ctv",
  "--split-mode",
  "-sm",
]);

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

export function diagnoseExtraArgs(
  input: string,
  catalog: LlamaFlagCatalog | null,
  /** True when the GPU picker owns placement, which removes the device flags. */
  gpuSelectionActive = false,
): ExtraArgsDiagnostic[] {
  const out: ExtraArgsDiagnostic[] = [];
  const { tokens, unterminatedQuote } = parseExtraArgs(input);

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
  // The other half of the backend's bounds. A grammar or a JSON schema is one long
  // token, so a payload can sit well inside the token cap and still be refused on
  // size; without this the Load button starts a request that cannot succeed.
  const bytes = TEXT_ENCODER.encode(tokens.join("")).length;
  if (bytes > EXTRA_ARGS_MAX_BYTES) {
    out.push({
      level: "error",
      message: `Arguments are too large: ${bytes} bytes, limit ${EXTRA_ARGS_MAX_BYTES}.`,
    });
  }

  // The backend refuses any token carrying one, and a command copied out of
  // coloured terminal output is the usual way one arrives.
  if (tokens.some((token) => CONTROL_CHARACTERS.test(token))) {
    out.push({
      level: "error",
      message: "Arguments cannot contain control characters.",
    });
  }

  const seen = new Set<string>();
  const unknown: string[] = [];
  const shadowed: string[] = [];
  const stripped: string[] = [];
  const reportedValues = new Set<string>();
  for (const [index, token] of tokens.entries()) {
    const flag = extraArgFlagName(token);
    if (flag === null) {
      continue;
    }
    // Before the de-duplication below, because llama.cpp reads the LAST occurrence:
    // in `-ngl 20 -ngl many` it is the second one the backend parses and refuses, so
    // checking only the first would leave Load enabled for a request that 400s.
    if (INTEGER_VALUE_FLAGS.has(flag) || VALUE_REQUIRED_FLAGS.has(flag)) {
      const attached = flag !== token.trim();
      const value = attached ? token.split("=")[1] : tokens[index + 1];
      // A flag whose value is the next token has none when that token is itself a
      // flag: `--ctx-size --numa` reads --numa as the value in a shell and as a
      // missing one here, which is what the backend's parser says too.
      const missing =
        value === undefined ||
        value === "" ||
        (!attached && extraArgFlagName(value) !== null);
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

    if (catalog?.managed.has(flag)) {
      const control = CONTROL_OWNED_FLAGS[flag];
      out.push({
        level: "error",
        message: control
          ? `${flag} is set by ${control} above and cannot be passed here.`
          : `${flag} is managed by Unsloth Studio and cannot be passed here.`,
      });
      continue;
    }
    if (gpuSelectionActive && GPU_SELECTION_STRIPPED_FLAGS[flag]) {
      stripped.push(flag);
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
    // Only when the catalogue was actually read: a build we could not probe must
    // not have every one of its flags called a typo.
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
