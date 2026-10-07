// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Extensions read inline as text. Extensions other adapters claim stay off, since the first
 * match wins; .ts/.mts are TypeScript here, not MPEG-TS.
 */
export const TEXT_ATTACHMENT_EXTENSIONS = [
  ".txt",
  ".text",
  ".log",
  ".md",
  ".markdown",
  ".mdx",
  ".rst",
  ".adoc",
  ".asciidoc",
  ".org",
  ".textile",
  ".wiki",
  ".tex",
  ".latex",
  ".sty",
  ".cls",
  ".bib",
  ".rmd",
  ".qmd",
  ".srt",
  ".vtt",
  ".sbv",
  ".ass",
  ".ssa",
  ".sub",
  ".lrc",
  ".csv",
  ".tsv",
  ".psv",
  ".json",
  ".jsonl",
  ".ndjson",
  ".jsonc",
  ".json5",
  ".geojson",
  ".har",
  ".avsc",
  ".xml",
  ".yaml",
  ".yml",
  ".toml",
  ".ini",
  ".cfg",
  ".conf",
  ".cnf",
  ".env",
  ".properties",
  ".plist",
  ".edn",
  ".ron",
  ".cue",
  ".lock",
  ".mod",
  ".sum",
  ".reg",
  ".desktop",
  ".service",
  ".po",
  ".pot",
  ".strings",
  ".resx",
  ".xliff",
  ".xlf",
  ".ics",
  ".vcf",
  ".eml",
  ".mbox",
  ".m3u8",
  ".pls",
  ".css",
  ".scss",
  ".sass",
  ".less",
  ".styl",
  ".svg",
  ".vue",
  ".svelte",
  ".astro",
  ".pug",
  ".jade",
  ".haml",
  ".slim",
  ".ejs",
  ".erb",
  ".hbs",
  ".handlebars",
  ".mustache",
  ".njk",
  ".jinja",
  ".jinja2",
  ".j2",
  ".twig",
  ".liquid",
  ".cshtml",
  ".razor",
  ".aspx",
  ".jsp",
  ".tpl",
  ".qml",
  ".js",
  ".jsx",
  ".mjs",
  ".cjs",
  ".ts",
  ".tsx",
  ".mts",
  ".cts",
  ".py",
  ".pyi",
  ".pyx",
  ".pxd",
  ".ipynb",
  ".java",
  ".kt",
  ".kts",
  ".scala",
  ".groovy",
  ".gradle",
  ".sbt",
  ".clj",
  ".cljs",
  ".cljc",
  ".c",
  ".h",
  ".cc",
  ".cpp",
  ".hpp",
  ".cxx",
  ".hxx",
  ".hh",
  ".ipp",
  ".inl",
  ".cu",
  ".cuh",
  ".rs",
  ".go",
  ".zig",
  ".odin",
  ".nim",
  ".nims",
  ".nimble",
  ".cr",
  ".d",
  ".v",
  ".sv",
  ".svh",
  ".vhd",
  ".vhdl",
  ".asm",
  ".s",
  ".cs",
  ".vb",
  ".vbs",
  ".fs",
  ".fsi",
  ".fsx",
  ".csproj",
  ".vbproj",
  ".fsproj",
  ".sln",
  ".props",
  ".targets",
  ".m",
  ".mm",
  ".swift",
  ".applescript",
  ".metal",
  ".rb",
  ".rake",
  ".gemspec",
  ".podspec",
  ".php",
  ".pl",
  ".pm",
  ".r",
  ".jl",
  ".lua",
  ".tcl",
  ".dart",
  ".hx",
  ".hs",
  ".lhs",
  ".ml",
  ".mli",
  ".ex",
  ".exs",
  ".erl",
  ".hrl",
  ".rkt",
  ".scm",
  ".ss",
  ".lisp",
  ".lsp",
  ".cl",
  ".el",
  ".pas",
  ".pp",
  ".ada",
  ".adb",
  ".ads",
  ".cob",
  ".cbl",
  ".f",
  ".for",
  ".f90",
  ".f95",
  ".f03",
  ".sas",
  ".awk",
  ".sed",
  ".m4",
  ".sol",
  ".move",
  ".cairo",
  ".mojo",
  ".gd",
  ".sqf",
  ".sh",
  ".bash",
  ".zsh",
  ".fish",
  ".ksh",
  ".csh",
  ".tcsh",
  ".nu",
  ".ps1",
  ".psm1",
  ".psd1",
  ".bat",
  ".cmd",
  ".sql",
  ".psql",
  ".plsql",
  ".hql",
  ".cql",
  ".graphql",
  ".gql",
  ".proto",
  ".thrift",
  ".capnp",
  ".prisma",
  ".tf",
  ".tfvars",
  ".tfstate",
  ".hcl",
  ".nix",
  ".dhall",
  ".bicep",
  ".dockerfile",
  ".containerfile",
  ".makefile",
  ".mk",
  ".mak",
  ".cmake",
  ".ninja",
  ".bzl",
  ".bazel",
  ".star",
  ".starlark",
  ".gn",
  ".gni",
  ".pro",
  ".pri",
  ".cabal",
  ".opam",
  ".glsl",
  ".frag",
  ".vert",
  ".geom",
  ".comp",
  ".hlsl",
  ".wgsl",
  ".shader",
  ".mmd",
  ".mermaid",
  ".puml",
  ".plantuml",
  ".dot",
  ".gv",
  ".feature",
  ".robot",
  ".http",
  ".rest",
  ".diff",
  ".patch",
];

/** Must match MAX_NATIVE_TEXT_BYTES in native_intents.rs. */
export const MAX_TEXT_ATTACHMENT_BYTES = 20 * 1024 * 1024;

/** assistant-ui reads "Dockerfile" as ".dockerfile", so drop paths must match these too. */
export const TEXT_ATTACHMENT_BASENAMES = [
  "containerfile",
  "dockerfile",
  "makefile",
] as const;
const PATH_SEPARATOR_RE = /[\\/]/;

export function isTextAttachmentName(path: string): boolean {
  const segments = path.split(PATH_SEPARATOR_RE);
  const name = (segments[segments.length - 1] || path).toLowerCase();
  if ((TEXT_ATTACHMENT_BASENAMES as readonly string[]).includes(name)) {
    return true;
  }
  const dot = name.lastIndexOf(".");
  return dot > 0 && TEXT_ATTACHMENT_EXTENSIONS.includes(name.slice(dot));
}

/** HTML accept cannot express extensionless names, so omit the hint; adapters still validate. */
export function pickerAcceptForTextBasenames(accept: string): string {
  if (accept === "*") {
    return accept;
  }
  const tokens = accept.split(",").map((token) => token.trim().toLowerCase());
  return TEXT_ATTACHMENT_BASENAMES.some((name) => tokens.includes(`.${name}`))
    ? "*"
    : accept;
}

export async function isBinaryPropertyList(file: File): Promise<boolean> {
  if (!/\.(?:plist|strings)$/i.test(file.name)) {
    return false;
  }
  const header = new Uint8Array(await file.slice(0, 8).arrayBuffer());
  return header.length === 8 && String.fromCharCode(...header) === "bplist00";
}

export async function isBinaryVobSubSubtitle(file: File): Promise<boolean> {
  if (!file.name.toLowerCase().endsWith(".sub")) {
    return false;
  }
  const header = new Uint8Array(await file.slice(0, 4).arrayBuffer());
  return (
    header.length === 4 &&
    header[0] === 0x00 &&
    header[1] === 0x00 &&
    header[2] === 0x01 &&
    header[3] === 0xba
  );
}

const TRACKER_MOD_MAGICS = new Set([
  "M.K.",
  "M!K!",
  // ProTracker's other 4-channel marker at offset 1080; the Soundtracker fallback misses it.
  "!PM!",
  "PATT",
  "NSMS",
  "LARD",
  "M&K!",
  "FEST",
  "N.T.",
  "OKTA",
  "OCTA",
  "CD81",
  "CD61",
  "FLT4",
  "FLT8",
  "EXO4",
  "EXO8",
  ".M.K",
  "WARD",
  "M\0\0\0",
  "8\0\0\0",
]);
const TRACKER_SINGLE_CHANNEL_MAGIC_RE = /^[1-9]CHN$/;
const TRACKER_DOUBLE_CHANNEL_MAGIC_RE = /^[1-9][0-9](?:CH|CN)$/;
const TRACKER_TAKE_MAGIC_RE = /^TDZ[1-9]$/;
const TRACKER_DIGITAL_MAGIC_RE = /^FA0[4-8]$/;

const SOUNDTRACKER_HEADER_BYTES = 600;
const SOUNDTRACKER_PATTERN_BYTES = 1024;
const SOUNDTRACKER_MAX_PATTERN_ERRORS = 22;
const SOUNDTRACKER_PERIODS = new Set([
  856, 808, 762, 720, 678, 640, 604, 570, 538, 508, 480, 453, 428, 404,
  381, 360, 339, 320, 302, 285, 269, 254, 240, 226, 214, 202, 190, 180,
  170, 160, 151, 143, 135, 127, 120, 113, 763, 679, 641, 571, 539, 509,
  429, 340, 321, 300, 286, 270, 227, 191, 162,
]);

function bigEndianWord(bytes: Uint8Array, offset: number): number {
  return bytes[offset] * 256 + bytes[offset + 1];
}

function soundtrackerSampleBytes(header: Uint8Array): number | null {
  let sampleBytes = 0;
  for (let sample = 0; sample < 15; sample += 1) {
    const offset = 20 + sample * 30;
    const lengthWords = bigEndianWord(header, offset + 22);
    const finetune = header[offset + 24];
    const volume = header[offset + 25];
    const loopStart = bigEndianWord(header, offset + 26);
    const loopLength = bigEndianWord(header, offset + 28);
    if (
      volume > 0x40 ||
      (finetune & 0xf0) !== 0 ||
      lengthWords > 0x8000 ||
      loopLength > 0x8000 ||
      (loopStart >>> 1) > lengthWords ||
      (lengthWords > 0 && (loopStart >>> 1) === lengthWords) ||
      (lengthWords === 0 && loopStart > 0)
    ) {
      return null;
    }
    sampleBytes += lengthWords * 2;
  }
  return sampleBytes >= 8 ? sampleBytes : null;
}

function soundtrackerPatternCounts(
  header: Uint8Array,
): { all: number; used: number } | null {
  const songLength = header[470];
  if (songLength === 0 || songLength > 128) {
    return null;
  }
  let maxPattern = 0;
  let maxUsedPattern = 0;
  for (let index = 0; index < 128; index += 1) {
    const pattern = header[472 + index];
    if (pattern > 0x7f) {
      return null;
    }
    maxPattern = Math.max(maxPattern, pattern);
    if (index < songLength) {
      maxUsedPattern = Math.max(maxUsedPattern, pattern);
    }
  }
  return { all: maxPattern + 1, used: maxUsedPattern + 1 };
}

function soundtrackerPatternEnd(
  header: Uint8Array,
  fileSize: number,
): number | null {
  if (header.length < SOUNDTRACKER_HEADER_BYTES) {
    return null;
  }

  const sampleBytes = soundtrackerSampleBytes(header);
  const patternCounts = soundtrackerPatternCounts(header);
  if (sampleBytes === null || patternCounts === null) {
    return null;
  }

  let patternCount = patternCounts.all;
  const usedPatternCount = patternCounts.used;
  let expectedSize =
    SOUNDTRACKER_HEADER_BYTES +
    patternCount * SOUNDTRACKER_PATTERN_BYTES +
    sampleBytes;
  const usedExpectedSize =
    SOUNDTRACKER_HEADER_BYTES +
    usedPatternCount * SOUNDTRACKER_PATTERN_BYTES +
    sampleBytes;
  if (fileSize < expectedSize && fileSize === usedExpectedSize) {
    patternCount = usedPatternCount;
    expectedSize = usedExpectedSize;
  }
  // Soundtracker files may truncate sample data, but the pattern area must be complete.
  const patternEnd =
    SOUNDTRACKER_HEADER_BYTES + patternCount * SOUNDTRACKER_PATTERN_BYTES;
  return fileSize >= Math.floor((expectedSize * 93) / 100) &&
    fileSize >= patternEnd
    ? patternEnd
    : null;
}

function hasSoundtrackerPatternData(
  bytes: Uint8Array,
  patternEnd: number,
): boolean {
  if (bytes.length < patternEnd) {
    return false;
  }
  let errors = 0;
  for (
    let offset = SOUNDTRACKER_HEADER_BYTES;
    offset < patternEnd;
    offset += 4
  ) {
    const sample = (bytes[offset] & 0xf0) | (bytes[offset + 2] >>> 4);
    const period = ((bytes[offset] & 0x0f) << 8) | bytes[offset + 1];
    if (sample > 15) {
      errors += 1;
      if (errors > SOUNDTRACKER_MAX_PATTERN_ERRORS) {
        return false;
      }
    }
    if (period !== 0 && !SOUNDTRACKER_PERIODS.has(period)) {
      errors += 1;
      if (errors > SOUNDTRACKER_MAX_PATTERN_ERRORS) {
        return false;
      }
    }
  }
  return true;
}

/** gfortran writes a compiled `.mod` module as gzip; text `go.mod` never is. */
export async function isCompiledFortranModule(file: File): Promise<boolean> {
  if (!file.name.toLowerCase().endsWith(".mod")) {
    return false;
  }
  const header = new Uint8Array(await file.slice(0, 2).arrayBuffer());
  return header.length === 2 && header[0] === 0x1f && header[1] === 0x8b;
}

export async function isBinaryTrackerModule(file: File): Promise<boolean> {
  if (!file.name.toLowerCase().endsWith(".mod")) {
    return false;
  }
  const prefix = new Uint8Array(await file.slice(0, 1084).arrayBuffer());
  if (prefix.length >= 1084) {
    const magic = String.fromCharCode(...prefix.subarray(1080, 1084));
    if (
      TRACKER_MOD_MAGICS.has(magic) ||
      TRACKER_SINGLE_CHANNEL_MAGIC_RE.test(magic) ||
      TRACKER_DOUBLE_CHANNEL_MAGIC_RE.test(magic) ||
      TRACKER_TAKE_MAGIC_RE.test(magic) ||
      TRACKER_DIGITAL_MAGIC_RE.test(magic)
    ) {
      return true;
    }
  }

  const patternEnd = soundtrackerPatternEnd(prefix, file.size);
  if (patternEnd === null) {
    return false;
  }
  const patterns = new Uint8Array(
    await file.slice(0, patternEnd).arrayBuffer(),
  );
  return hasSoundtrackerPatternData(patterns, patternEnd);
}

const GETTEXT_HEADER_SCAN_BYTES = 64 * 1024;
const GETTEXT_CHARSET_ALIASES: Record<string, string> = {
  CP874: "windows-874",
  CP932: "shift_jis",
  CP949: "euc-kr",
  CP950: "big5",
};
const GETTEXT_HEADER_ENTRY_RE =
  /(?:^|\r?\n)msgid[ \t]+""[ \t]*\r?\nmsgstr[ \t]+("(?:[^"\\]|\\.)*"(?:[ \t]*\r?\n[ \t]*"(?:[^"\\]|\\.)*")*)/;
const GETTEXT_CHARSET_RE =
  /Content-Type:[^"\r\n]*?charset[ \t]*=[ \t]*([A-Za-z0-9._-]+)/i;

/** Gettext header charset; if the 64 KiB prefix holds no entry, read the whole file. */
function declaredGettextCharset(
  bytes: Uint8Array,
  fileName: string,
): string | null {
  if (!/\.(?:po|pot)$/i.test(fileName)) {
    return null;
  }
  const decoder = new TextDecoder("windows-1252");
  const cut = bytes.length > GETTEXT_HEADER_SCAN_BYTES;
  const fromPrefix = gettextHeaderCharset(
    decoder.decode(bytes.subarray(0, GETTEXT_HEADER_SCAN_BYTES)),
    cut,
  );
  if (fromPrefix || !cut) {
    return fromPrefix;
  }
  // A cutoff can split the entry itself, so retry on missing charset, not missing entry.
  return gettextHeaderCharset(decoder.decode(bytes));
}

const GETTEXT_ESCAPES: Record<string, string> = {
  a: "\x07",
  b: "\b",
  f: "\f",
  n: "\n",
  r: "\r",
  t: "\t",
  v: "\v",
};

/** Joins adjacent PO literals and resolves escapes; a charset can be split across pieces. */
function gettextStringValue(raw: string): string {
  let out = "";
  let inside = false;
  for (let index = 0; index < raw.length; index += 1) {
    const character = raw[index];
    if (!inside) {
      if (character === '"') inside = true;
      continue;
    }
    if (character === "\\" && index + 1 < raw.length) {
      const escaped = raw[index + 1];
      out += GETTEXT_ESCAPES[escaped] ?? escaped;
      index += 1;
    } else if (character === '"') {
      inside = false;
    } else {
      out += character;
    }
  }
  return out;
}

const GETTEXT_ENTRY_ENDED_RE = /\r?\n[ \t]*[^ \t"\r\n]/;

/** @param truncated Whether `text` is a prefix; a cut entry asks the caller to read it all. */
function gettextHeaderCharset(text: string, truncated = false): string | null {
  const match = text.match(GETTEXT_HEADER_ENTRY_RE);
  if (!match) {
    return null;
  }
  const after = text.slice((match.index ?? 0) + match[0].length);
  if (truncated && !GETTEXT_ENTRY_ENDED_RE.test(after)) {
    return null;
  }
  const charset = gettextStringValue(match[1]).match(GETTEXT_CHARSET_RE)?.[1];
  return charset && charset.toUpperCase() !== "CHARSET" ? charset : null;
}

// Header blocks only; every Content-Type counts (multipart parts, mbox messages).
const EMAIL_CONTENT_TYPE_RE =
  /(?:^|\r?\n)Content-Type:((?:[^\r\n]*)(?:\r?\n[ \t][^\r\n]*)*)/gi;
const CHARSET_LABEL_RE = /^[A-Za-z0-9._-]+/;

function headerParameters(value: string): string[] {
  const parameters: string[] = [];
  let start = 0;
  let quoted = false;
  for (let index = 0; index < value.length; index += 1) {
    const character = value[index];
    if (character === "\\" && quoted) {
      index += 1;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (character === ";" && !quoted) {
      parameters.push(value.slice(start, index));
      start = index + 1;
    }
  }
  parameters.push(value.slice(start));
  return parameters.slice(1);
}

/** Anchored parameter lookup, so quoted values like filename="charset=..." do not match. */
function headerParameter(value: string, name: string): string | undefined {
  for (const parameter of headerParameters(value)) {
    const equals = parameter.indexOf("=");
    if (equals === -1) continue;
    if (parameter.slice(0, equals).trim().toLowerCase() !== name) continue;
    const raw = parameter.slice(equals + 1).trim();
    if (!raw.startsWith('"')) return raw || undefined;
    return unquoteHeaderValue(raw) || undefined;
  }
  return undefined;
}

function unquoteHeaderValue(raw: string): string {
  let out = "";
  for (let index = 1; index < raw.length; index += 1) {
    const character = raw[index];
    if (character === "\\" && index + 1 < raw.length) {
      out += raw[index + 1];
      index += 1;
    } else if (character === '"') {
      break;
    } else {
      out += character;
    }
  }
  return out;
}

function headerCharsetParameter(value: string): string | undefined {
  return headerParameter(value, "charset")?.match(CHARSET_LABEL_RE)?.[0];
}

const VCARD_CHARSET_RE = /;[ \t]*CHARSET[ \t]*=[ \t]*"?([A-Za-z0-9._-]+)"?/gi;
const MULTIPART_TYPE_RE = /^[ \t]*multipart\//i;
const HEADER_FOLD_RE = /\r?\n[ \t]+/g;

/**
 * Header regions of a message or archive in one linear pass (rescanning was quadratic).
 * Only declared delimiters reopen headers.
 * @param mbox Whether the file is an archive; only then does a `From ` line separate messages.
 */
function emailHeaderBlocks(text: string, mbox: boolean): string[] {
  const blocks: string[] = [];
  const boundaries = new Set<string>();
  const closeBlock = (block: string) => {
    // Unfold first: multipart/mixed often wraps before its boundary parameter.
    const unfolded = block.replace(HEADER_FOLD_RE, " ");
    blocks.push(unfolded);
    for (const header of unfolded.matchAll(EMAIL_CONTENT_TYPE_RE)) {
      const value = header[1] ?? "";
      // Only a multipart header's boundary counts, not a quoted filename elsewhere.
      if (!MULTIPART_TYPE_RE.test(value)) continue;
      const boundary = headerParameter(value, "boundary");
      if (boundary) boundaries.add(boundary);
    }
  };
  let position = 0;
  let blockStart = 0;
  let inHeader = true;
  while (position <= text.length) {
    let lineBreak = text.indexOf("\n", position);
    if (lineBreak === -1) lineBreak = text.length;
    const contentEnd =
      lineBreak > position && text[lineBreak - 1] === "\r"
        ? lineBreak - 1
        : lineBreak;
    const isBlank = contentEnd === position;
    if (inHeader) {
      if (isBlank) {
        closeBlock(text.slice(blockStart, position));
        inHeader = false;
      }
    } else if (!isBlank) {
      let resumes = mbox && text.startsWith("From ", position);
      if (resumes) {
        // Boundaries belong to their message; carrying them over lets body lines reopen headers.
        boundaries.clear();
      } else if (boundaries.size > 0 && text.startsWith("--", position)) {
        // A closing delimiter keeps its trailing "--" in the token, so it does not reopen headers.
        const token = text
          .slice(position + 2, contentEnd)
          .replace(/[ \t]+$/, "");
        resumes = boundaries.has(token);
        if (!resumes && token.endsWith("--")) {
          boundaries.delete(token.slice(0, -2));
        }
      }
      if (resumes) {
        inHeader = true;
        blockStart = lineBreak + 1;
      }
    }
    if (lineBreak === text.length) break;
    position = lineBreak + 1;
  }
  if (inHeader && blockStart < text.length) {
    closeBlock(text.slice(blockStart));
  }
  return blocks;
}

/** vCard parameter sections: each ends at the first colon outside a quoted value. */
function vCardParameterSections(text: string): string[] {
  const sections: string[] = [];
  let section = "";
  let valueReached = true;
  let quoted = false;
  let position = 0;
  while (position <= text.length) {
    let lineBreak = text.indexOf("\n", position);
    if (lineBreak === -1) lineBreak = text.length;
    const contentEnd =
      lineBreak > position && text[lineBreak - 1] === "\r"
        ? lineBreak - 1
        : lineBreak;
    const folded = text[position] === " " || text[position] === "\t";
    let start = position;
    if (folded) {
      start = position + 1;
    } else {
      if (section) sections.push(section);
      section = "";
      valueReached = false;
      quoted = false;
    }
    if (!valueReached) {
      let cut = contentEnd;
      for (let index = start; index < contentEnd; index += 1) {
        const character = text[index];
        if (character === '"') {
          quoted = !quoted;
        } else if (character === ":" && !quoted) {
          cut = index;
          valueReached = true;
          break;
        }
      }
      section += text.slice(start, cut);
    }
    if (lineBreak === text.length) break;
    position = lineBreak + 1;
  }
  if (section) sections.push(section);
  return sections;
}

// The whitespace after `xml` separates a declaration from PIs like xml-stylesheet.
const XML_PROLOG_ENCODING_RE =
  /^<\?xml[ \t\r\n][^>]*?[ \t\r\n]encoding[ \t]*=[ \t]*["\']([A-Za-z0-9._-]+)["\']/i;
// XML's S production only; `\s` would also accept form feed and no-break space.
const XML_WHITESPACE_BYTES = new Set([0x20, 0x09, 0x0d, 0x0a]);

/** Reads to the declaration's own ">" since whitespace inside it is unbounded. */
function declaredXmlEncoding(bytes: Uint8Array): string | null {
  const decoder = new TextDecoder("windows-1252");
  if (decoder.decode(bytes.subarray(0, 5)).toLowerCase() !== "<?xml") {
    return null;
  }
  if (bytes.length < 6 || !XML_WHITESPACE_BYTES.has(bytes[5])) {
    return null;
  }
  const close = bytes.indexOf(0x3e);
  if (close === -1) {
    return null;
  }
  const declaration = decoder.decode(bytes.subarray(0, close + 1));
  return declaration.match(XML_PROLOG_ENCODING_RE)?.[1] ?? null;
}

/** Canonical name via TextDecoder, so cp1252/latin1/windows-1252 count as one charset. */
function canonicalCharset(charset: string): string {
  const label = GETTEXT_CHARSET_ALIASES[charset.toUpperCase()] ?? charset;
  try {
    return new TextDecoder(label).encoding;
  } catch {
    return charset.toLowerCase();
  }
}

function charsetCollector(): { found: string[]; add: (c?: string) => void } {
  const found: string[] = [];
  const seen = new Set<string>();
  return {
    found,
    add(charset?: string) {
      if (!charset) return;
      const canonical = canonicalCharset(charset);
      if (seen.has(canonical)) return;
      seen.add(canonical);
      found.push(charset);
    },
  };
}

// Scan the whole file, not a prefix: a later declaration is the one a cutoff would miss.

function declaredVCardCharsets(bytes: Uint8Array, fileName: string): string[] {
  if (!/\.vcf$/i.test(fileName)) {
    return [];
  }
  const text = new TextDecoder("windows-1252").decode(bytes);
  const { found, add } = charsetCollector();
  for (const section of vCardParameterSections(text)) {
    for (const property of section.matchAll(VCARD_CHARSET_RE)) {
      add(property[1]);
    }
  }
  return found;
}

function declaredEmailCharsets(bytes: Uint8Array, fileName: string): string[] {
  const isMbox = /\.mbox$/i.test(fileName);
  if (!isMbox && !/\.eml$/i.test(fileName)) {
    return [];
  }
  const text = new TextDecoder("windows-1252").decode(bytes);
  const { found, add } = charsetCollector();
  for (const block of emailHeaderBlocks(text, isMbox)) {
    for (const header of block.matchAll(EMAIL_CONTENT_TYPE_RE)) {
      add(headerCharsetParameter(header[1] ?? ""));
    }
  }
  return found;
}

/** Must be UndecodableTextError so the composer's instanceof check toasts it. */
function unsupportedCharsetError(
  fileName: string,
  charset: string,
): UndecodableTextError {
  return new UndecodableTextError(
    fileName,
    `It declares charset "${charset}", which this browser has no decoder for.`,
  );
}

function contradictedCharsetError(
  fileName: string,
  charset: string,
): UndecodableTextError {
  return new UndecodableTextError(
    fileName,
    `It declares charset "${charset}" but does not hold valid ${charset} text.`,
  );
}

function decodeWithCharset(
  bytes: Uint8Array,
  charset: string,
  fileName: string,
  truncated: boolean,
): string {
  const label = GETTEXT_CHARSET_ALIASES[charset.toUpperCase()] ?? charset;
  let decoder: TextDecoder;
  try {
    // Strict: bytes breaking a declared charset are corrupt (only multibyte charsets can fail).
    decoder = new TextDecoder(label, { fatal: true });
  } catch (error) {
    if (error instanceof RangeError) {
      throw unsupportedCharsetError(fileName, charset);
    }
    throw error;
  }
  try {
    return decoder.decode(bytes, { stream: truncated });
  } catch {
    throw contradictedCharsetError(fileName, charset);
  }
}

function strictDecoder(charset: string): TextDecoder | null {
  const label = GETTEXT_CHARSET_ALIASES[charset.toUpperCase()] ?? charset;
  try {
    return new TextDecoder(label, { fatal: true });
  } catch {
    return null;
  }
}

const ASCII_DECODER_LABEL = "windows-1252";

/**
 * Decodes a vCard per property, since CHARSET is per property in 2.1 cards.
 * Returns no text when unsure so the caller falls back; `failure` marks a bad declaration.
 * @param truncated Whether `bytes` is a prefix. Only its last line can end mid-character.
 */
function decodeVCardPerProperty(
  bytes: Uint8Array,
  truncated: boolean,
  fileName: string,
): { text: string } | { failure: Error | null } {
  const ascii = new TextDecoder(ASCII_DECODER_LABEL);
  const utf8 = new TextDecoder("utf-8", { fatal: true });
  const decoders = new Map<string, TextDecoder | null>();
  const decoderFor = (charset: string | null): TextDecoder | null => {
    if (charset === null) return utf8;
    const seen = decoders.get(charset);
    if (seen !== undefined) return seen;
    const made = strictDecoder(charset);
    decoders.set(charset, made);
    return made;
  };

  const out: string[] = [];
  let valueDecoder: TextDecoder | null = utf8;
  let valueCharset: string | null = null;
  let position = 0;
  while (position < bytes.length) {
    let lineBreak = bytes.indexOf(0x0a, position);
    if (lineBreak === -1) lineBreak = bytes.length;
    const contentEnd =
      lineBreak > position && bytes[lineBreak - 1] === 0x0d
        ? lineBreak - 1
        : lineBreak;
    const folded = bytes[position] === 0x20 || bytes[position] === 0x09;
    // Only a prefix's final line can end mid-character, so only it decodes leniently.
    const cut = truncated && lineBreak === bytes.length;
    try {
      if (folded) {
        if (!valueDecoder) return { failure: null };
        out.push(
          valueDecoder.decode(bytes.subarray(position, contentEnd), {
            stream: cut,
          }),
        );
      } else {
        const delimiter = vCardValueDelimiter(bytes, position, contentEnd);
        if (delimiter === -1) {
          out.push(
            utf8.decode(bytes.subarray(position, contentEnd), { stream: cut }),
          );
          valueDecoder = utf8;
          valueCharset = null;
        } else {
          const parameters = ascii.decode(bytes.subarray(position, delimiter));
          const charsets: string[] = [];
          for (const match of parameters.matchAll(VCARD_CHARSET_RE)) {
            if (match[1]) charsets.push(match[1]);
          }
          if (charsets.length > 1) {
            return {
              failure: new UndecodableTextError(
                fileName,
                `One of its properties declares two charsets (${charsets.join(", ")}).`,
              ),
            };
          }
          valueCharset = charsets[0] ?? null;
          valueDecoder = decoderFor(valueCharset);
          if (!valueDecoder) {
            return { failure: unsupportedCharsetError(fileName, valueCharset!) };
          }
          out.push(parameters, ":");
          out.push(
            valueDecoder.decode(bytes.subarray(delimiter + 1, contentEnd), {
              stream: cut,
            }),
          );
        }
      }
    } catch {
      return {
        failure: valueCharset
          ? contradictedCharsetError(fileName, valueCharset)
          : null,
      };
    }
    out.push(contentEnd === lineBreak ? "" : "\r");
    if (lineBreak === bytes.length) break;
    out.push("\n");
    position = lineBreak + 1;
  }
  return { text: out.join("") };
}

function vCardValueDelimiter(
  bytes: Uint8Array,
  start: number,
  end: number,
): number {
  let quoted = false;
  for (let index = start; index < end; index += 1) {
    const byte = bytes[index];
    if (byte === 0x22) quoted = !quoted;
    else if (byte === 0x3a && !quoted) return index;
  }
  return -1;
}

/**
 * @param truncated Whether `bytes` is a prefix, so a trailing partial character is a cut.
 * @param whole The complete file; declarations can sit anywhere in it.
 */
export function decodeTextAttachmentBytes(
  bytes: Uint8Array,
  fileName = "",
  truncated = false,
  whole: Uint8Array = bytes,
): string {
  // A BOM is a declaration too, so it decodes strictly.
  if (bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xfe) {
    return decodeWithCharset(bytes.subarray(2), "utf-16le", fileName, truncated);
  }
  if (bytes.length >= 2 && bytes[0] === 0xfe && bytes[1] === 0xff) {
    return decodeWithCharset(bytes.subarray(2), "utf-16be", fileName, truncated);
  }
  // Exporter-written declarations (XML, gettext, vCard) win over UTF-8; mail Content-Type is a
  // fallback below because clients mislabel 8-bit mail.
  const xmlEncoding = declaredXmlEncoding(whole);
  if (xmlEncoding) {
    return decodeWithCharset(bytes, xmlEncoding, fileName, truncated);
  }
  const gettextCharset = declaredGettextCharset(whole, fileName);
  if (gettextCharset) {
    return decodeWithCharset(bytes, gettextCharset, fileName, truncated);
  }
  const vCardCharsets = declaredVCardCharsets(whole, fileName);
  if (vCardCharsets.length > 0) {
    // CHARSET is per property, so one declaration never speaks for the whole file.
    const perProperty = decodeVCardPerProperty(bytes, truncated, fileName);
    if ("text" in perProperty) {
      return perProperty.text;
    }
    // A declaration the per-property read could not honour must be reported, not read past.
    if (perProperty.failure && vCardCharsets.length > 1) {
      throw perProperty.failure;
    }
  }
  if (vCardCharsets.length === 1) {
    return decodeWithCharset(bytes, vCardCharsets[0]!, fileName, truncated);
  }
  try {
    // Only a truncated read uses stream:true; a dangling lead byte in a whole file is an error.
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes, {
      stream: truncated,
    });
  } catch {
    const declared = vCardCharsets.length
      ? vCardCharsets
      : declaredEmailCharsets(whole, fileName);
    if (declared.length === 1) {
      return decodeWithCharset(bytes, declared[0]!, fileName, truncated);
    }
    if (declared.length > 1) {
      // Parts in different encodings cannot be decoded as one unit, and this is not a MIME parser.
      throw new UndecodableTextError(
        fileName,
        `It declares more than one charset (${declared.join(", ")}), and is read as one unit.`,
      );
    }
    // Legacy code pages are not knowable from bytes, so refuse rather than guess.
    throw new UndecodableTextError(fileName);
  }
}

export class UndecodableTextError extends Error {
  constructor(fileName: string, reason?: string) {
    super(
      `${fileName || "This file"} is not UTF-8 text. ${
        reason ?? "It looks like a legacy code page."
      } Convert it to UTF-8 before attaching it.`,
    );
    this.name = "UndecodableTextError";
  }
}

const OLE_COMPOUND_FILE_MAGIC = [
  0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1,
];

/** Word `.dot` and PowerPoint `.pot` are OLE files, unlike Graphviz `.dot` and gettext `.pot`. */
export async function isBinaryOfficeTemplate(file: File): Promise<boolean> {
  if (!/\.(?:dot|pot)$/i.test(file.name)) {
    return false;
  }
  const header = new Uint8Array(await file.slice(0, 8).arrayBuffer());
  return (
    header.length === 8 &&
    OLE_COMPOUND_FILE_MAGIC.every((byte, index) => header[index] === byte)
  );
}

export async function readTextAttachment(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  return decodeTextAttachmentBytes(bytes, file.name);
}

const decodedOnce = new WeakMap<File, string>();

/** Decode once per file: attaching already decodes, so sending must not re-read it. */
export async function readTextAttachmentOnce(file: File): Promise<string> {
  const cached = decodedOnce.get(file);
  if (cached !== undefined) {
    return cached;
  }
  const text = await readTextAttachment(file);
  decodedOnce.set(file, text);
  return text;
}

// MIME is unreliable for source files, so match by extension too.
export const TEXT_ATTACHMENT_ACCEPT = [
  "text/plain,text/markdown,text/csv,text/tab-separated-values,text/xml,text/json,text/css",
  "text/vtt,application/x-subrip,text/x-log,text/calendar,text/vcard,message/rfc822",
  "application/json,application/xml,application/yaml,application/toml,image/svg+xml",
  TEXT_ATTACHMENT_EXTENSIONS.join(","),
].join(",");
