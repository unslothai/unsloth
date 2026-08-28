// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Extensions `TextAttachmentAdapter` reads inline; shared with the drop path.
 *
 * Read as UTF-8 (or BOM-marked UTF-16) and sent as text, so anything an editor
 * opens belongs here.
 * Extensions another adapter claims stay off: the composite adapter takes the
 * first match, so .html, .pdf, .ods/.odt and the media containers are theirs.
 * .ts and .mts are TypeScript here, so video cannot take them for MPEG-TS.
 */
export const TEXT_ATTACHMENT_EXTENSIONS = [
  // Prose and documentation
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
  // Subtitles and captions
  ".srt",
  ".vtt",
  ".sbv",
  ".ass",
  ".ssa",
  ".sub",
  ".lrc",
  // Structured data
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
  // Localisation and interchange
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
  // Stylesheets and web templates
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
  // JavaScript and TypeScript
  ".js",
  ".jsx",
  ".mjs",
  ".cjs",
  ".ts",
  ".tsx",
  ".mts",
  ".cts",
  // Python
  ".py",
  ".pyi",
  ".pyx",
  ".pxd",
  ".ipynb",
  // JVM
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
  // Systems languages
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
  // .NET
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
  // Apple platforms
  ".m",
  ".mm",
  ".swift",
  ".applescript",
  ".metal",
  // Everything else with a compiler or interpreter
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
  // Shells
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
  // Queries and schemas
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
  // Infrastructure and build
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
  // Shaders
  ".glsl",
  ".frag",
  ".vert",
  ".geom",
  ".comp",
  ".hlsl",
  ".wgsl",
  ".shader",
  // Diagrams, specs and request files
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
  // Diffs
  ".diff",
  ".patch",
];

/** Matches MAX_NATIVE_TEXT_BYTES in native_intents.rs, so a file dropped from
 *  the desktop shell and one picked in the browser accept the same sizes. An
 *  .mbox can run to gigabytes, and reading one decodes it twice over in memory. */
export const MAX_TEXT_ATTACHMENT_BYTES = 20 * 1024 * 1024;

/** Conventional extensionless names matched through their dotted adapter tokens.
 *  assistant-ui reads "Dockerfile" as the extension ".dockerfile", so the picker
 *  already claims these; the drop paths have to agree. */
export const TEXT_ATTACHMENT_BASENAMES = [
  "containerfile",
  "dockerfile",
  "makefile",
] as const;
const PATH_SEPARATOR_RE = /[\\/]/;

/** Whether a path or filename belongs to the inline UTF-8 text adapter. */
export function isTextAttachmentName(path: string): boolean {
  const segments = path.split(PATH_SEPARATOR_RE);
  const name = (segments[segments.length - 1] || path).toLowerCase();
  if ((TEXT_ATTACHMENT_BASENAMES as readonly string[]).includes(name)) {
    return true;
  }
  const dot = name.lastIndexOf(".");
  return dot > 0 && TEXT_ATTACHMENT_EXTENSIONS.includes(name.slice(dot));
}

/**
 * HTML's `accept` attribute can express MIME types and extensions, not an exact
 * extensionless filename. Omit that picker hint when this adapter is present;
 * the attachment adapters still validate every selected file.
 */
export function pickerAcceptForTextBasenames(accept: string): string {
  if (accept === "*") {
    return accept;
  }
  const tokens = accept.split(",").map((token) => token.trim().toLowerCase());
  return TEXT_ATTACHMENT_BASENAMES.some((name) => tokens.includes(`.${name}`))
    ? "*"
    : accept;
}

/** Binary Apple property lists are not text despite using `.plist` or `.strings`. */
export async function isBinaryPropertyList(file: File): Promise<boolean> {
  if (!/\.(?:plist|strings)$/i.test(file.name)) {
    return false;
  }
  const header = new Uint8Array(await file.slice(0, 8).arrayBuffer());
  return header.length === 8 && String.fromCharCode(...header) === "bplist00";
}

/** VobSub `.sub` files are MPEG program streams containing bitmap subtitles. */
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
  // Some old files leave junk orders past the declared song length.
  if (fileSize < expectedSize && fileSize === usedExpectedSize) {
    patternCount = usedPatternCount;
    expectedSize = usedExpectedSize;
  }
  // Known Soundtracker files can have truncated trailing sample data, but the
  // complete pattern area must still be present.
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

/** Detect marker-bearing MODs and earlier 15-sample Soundtracker modules. */
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

/**
 * The charset a gettext catalog's header entry declares.
 *
 * The entry is the first one in the file by convention, so the prefix settles
 * it without decoding a catalog that can run to megabytes. A cutoff can still
 * miss it, though: a project with more than 64 KiB of translator comments above
 * the entry pushes it past the prefix, and the catalog was then refused for
 * carrying exactly the bytes its header describes. So when the prefix holds no
 * entry, read the rest, which the attachment cap already bounds.
 */
function declaredGettextCharset(
  bytes: Uint8Array,
  fileName: string,
): string | null {
  if (!/\.(?:po|pot)$/i.test(fileName)) {
    return null;
  }
  const decoder = new TextDecoder("windows-1252");
  const fromPrefix = gettextHeaderCharset(
    decoder.decode(bytes.subarray(0, GETTEXT_HEADER_SCAN_BYTES)),
  );
  if (fromPrefix || bytes.length <= GETTEXT_HEADER_SCAN_BYTES) {
    return fromPrefix;
  }
  // A cutoff can also split the entry itself, leaving a match that holds the
  // first continuation lines but not the Content-Type one, so retry on the
  // charset rather than on the entry.
  return gettextHeaderCharset(decoder.decode(bytes));
}

function gettextHeaderCharset(text: string): string | null {
  const headerEntry = text.match(GETTEXT_HEADER_ENTRY_RE)?.[1];
  const charset = headerEntry?.match(GETTEXT_CHARSET_RE)?.[1];
  // A .pot template ships the literal placeholder rather than a charset.
  return charset && charset.toUpperCase() !== "CHARSET" ? charset : null;
}

// Header block only: a body part can declare its own, but the message-level
// charset is what an 8-bit single-part mail is written in.
// Every Content-Type in the file, not just the first: a multipart message keeps
// the charset on its parts, and an mbox holds one per message, which can sit
// anywhere in the archive. Folded continuation lines start with space or tab.
const EMAIL_CONTENT_TYPE_RE =
  /(?:^|\r?\n)Content-Type:((?:[^\r\n]*)(?:\r?\n[ \t][^\r\n]*)*)/gi;
const EMAIL_CHARSET_RE = /charset[ \t]*=[ \t]*"?([A-Za-z0-9._-]+)"?/i;
// vCard 2.1 puts the encoding on the property: `FN;CHARSET=windows-1252:...`.
const VCARD_CHARSET_RE = /;[ \t]*CHARSET[ \t]*=[ \t]*"?([A-Za-z0-9._-]+)"?/gi;
// The delimiter a multipart header names for its own parts.
const MIME_BOUNDARY_RE =
  /;[ \t]*boundary[ \t]*=[ \t]*(?:"([^"\r\n]*)"|([^\s;"]+))/gi;

// A header block runs from the start of the file, an mbox "From " separator, or
// a MIME boundary, to the first blank line. Body text can hold a line that reads
// like a header, and treating that as a declaration refused valid files.
/**
 * The header regions of a message or archive, body text excluded.
 *
 * One forward pass over the lines, slicing only the header regions themselves.
 * Restarting a search from each candidate boundary instead is quadratic, and a
 * body of diff hunks is all candidate boundaries: 1 MB of them took 14 seconds.
 *
 * Only a delimiter a header actually declared reopens the headers. Any line
 * starting with `--` reopened them before, so a signature marker or a quoted
 * diff put the body back into header mode, and the next body line shaped like
 * `Content-Type: ... charset=...` became a second declaration that refused the
 * file. A closing delimiter, `--boundary--`, ends its part instead of starting
 * one.
 *
 * @param mbox Whether the file is an archive of messages. A `From ` line only
 * separates messages there, and an archive escapes the one a body starts with.
 * A single `.eml` has no separator at all, so an ordinary sentence opening with
 * "From " is body text and must not reopen the headers.
 */
function emailHeaderBlocks(text: string, mbox: boolean): string[] {
  const blocks: string[] = [];
  const boundaries = new Set<string>();
  const closeBlock = (block: string) => {
    blocks.push(block);
    for (const declared of block.matchAll(MIME_BOUNDARY_RE)) {
      const boundary = declared[1] ?? declared[2];
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
      // An mbox separator, or a delimiter one of the headers above declared.
      let resumes = mbox && text.startsWith("From ", position);
      if (resumes) {
        // A boundary belongs to the message that named it. Carrying one into
        // the next message lets a body line that happens to repeat it reopen
        // the headers there.
        boundaries.clear();
      } else if (boundaries.size > 0 && text.startsWith("--", position)) {
        // Trailing whitespace is allowed on a delimiter line and is not part of
        // the boundary token. A closing delimiter carries the extra "--" into
        // the token, so it does not match and does not reopen the headers.
        const token = text
          .slice(position + 2, contentEnd)
          .replace(/[ \t]+$/, "");
        resumes = boundaries.has(token);
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

/**
 * The property-parameter sections of a vCard, values excluded.
 *
 * `FN;CHARSET=windows-1252:name` declares a charset; `NOTE:see \;CHARSET=x`
 * does not, because that text is the value. The section therefore ends at the
 * property's value delimiter: the first colon outside a quoted parameter value.
 * Folded lines, which begin with a space or tab, continue the line above.
 */
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

// An XML prolog names the document's encoding and must be the first thing in it.
// Read from the bytes rather than the extension, so every XML dialect here
// (.resx, .xliff, .svg, a text .plist, the project files) is covered at once.
const XML_PROLOG_ENCODING_RE =
  /^<\?xml[^>]*?\sencoding[ \t]*=[ \t]*["\']([A-Za-z0-9._-]+)["\']/i;

function declaredXmlEncoding(bytes: Uint8Array): string | null {
  const prolog = new TextDecoder("windows-1252").decode(bytes.subarray(0, 256));
  return prolog.match(XML_PROLOG_ENCODING_RE)?.[1] ?? null;
}

/** Collects distinct charsets in encounter order, case-insensitively. */
function charsetCollector(): { found: string[]; add: (c?: string) => void } {
  const found: string[] = [];
  return {
    found,
    add(charset?: string) {
      if (!charset) return;
      if (!found.some((seen) => seen.toUpperCase() === charset.toUpperCase())) {
        found.push(charset);
      }
    },
  };
}

// Headers and property parameters are ASCII, so these scans cannot fail on the
// body's own encoding. They read the whole file rather than a prefix: a
// declaration further in is the one a cutoff would miss, leaving those messages
// decoded as the first. The attachment cap already bounds the work.

/** Charsets a `.vcf` declares, one per property parameter. */
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

/** Charsets an `.eml` or `.mbox` declares, one per message or part header. */
function declaredEmailCharsets(bytes: Uint8Array, fileName: string): string[] {
  const isMbox = /\.mbox$/i.test(fileName);
  if (!isMbox && !/\.eml$/i.test(fileName)) {
    return [];
  }
  const text = new TextDecoder("windows-1252").decode(bytes);
  const { found, add } = charsetCollector();
  for (const block of emailHeaderBlocks(text, isMbox)) {
    for (const header of block.matchAll(EMAIL_CONTENT_TYPE_RE)) {
      add(header[1]?.match(EMAIL_CHARSET_RE)?.[1]);
    }
  }
  return found;
}

/** Decode under a charset the file itself declared, or say why it could not. */
function decodeWithCharset(
  bytes: Uint8Array,
  charset: string,
  fileName: string,
  truncated: boolean,
): string {
  const label = GETTEXT_CHARSET_ALIASES[charset.toUpperCase()] ?? charset;
  let decoder: TextDecoder;
  try {
    // Strict, like the default path: a declaration is a claim about the bytes,
    // and bytes that break it are corrupt rather than readable. Single-byte
    // charsets map everything, so this only bites on the multibyte ones.
    decoder = new TextDecoder(label, { fatal: true });
  } catch (error) {
    if (error instanceof RangeError) {
      throw new Error(
        `Charset "${charset}" isn't supported. Convert the file to UTF-8 before attaching it.`,
      );
    }
    throw error;
  }
  try {
    return decoder.decode(bytes, { stream: truncated });
  } catch {
    throw new UndecodableTextError(
      fileName,
      `It declares charset "${charset}" but does not hold valid ${charset} text.`,
    );
  }
}

/**
 * @param truncated Whether `bytes` is a prefix of the file, in which case a
 * partial character at the end is a cut rather than a bad encoding.
 */
export function decodeTextAttachmentBytes(
  bytes: Uint8Array,
  fileName = "",
  truncated = false,
): string {
  // A BOM is a declaration too, so it decodes as strictly as the rest: an odd
  // trailing byte or an unpaired surrogate is corrupt, not readable.
  if (bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xfe) {
    return decodeWithCharset(bytes.subarray(2), "utf-16le", fileName, truncated);
  }
  if (bytes.length >= 2 && bytes[0] === 0xfe && bytes[1] === 0xff) {
    return decodeWithCharset(bytes.subarray(2), "utf-16be", fileName, truncated);
  }
  // A document that states its own encoding decides before UTF-8 is tried:
  // bytes that happen to be valid UTF-8 would otherwise decode into different
  // characters than the file says it holds. An XML prolog is that statement by
  // specification, a gettext header and a vCard property parameter by format,
  // and all three are written by the exporter rather than typed by a person.
  //
  // A mail Content-Type is the exception and stays a fallback below: clients
  // mislabel 8-bit mail constantly, so the bytes being valid UTF-8 is the
  // better evidence there.
  const xmlEncoding = declaredXmlEncoding(bytes);
  if (xmlEncoding) {
    return decodeWithCharset(bytes, xmlEncoding, fileName, truncated);
  }
  const gettextCharset = declaredGettextCharset(bytes, fileName);
  if (gettextCharset) {
    return decodeWithCharset(bytes, gettextCharset, fileName, truncated);
  }
  // Only when the card names one charset. Several is the ambiguous case, and it
  // keeps its existing path: valid UTF-8 wins, and anything else is refused
  // below with all of them named.
  const vCardCharsets = declaredVCardCharsets(bytes, fileName);
  if (vCardCharsets.length === 1) {
    return decodeWithCharset(bytes, vCardCharsets[0]!, fileName, truncated);
  }
  try {
    // A truncated read decodes with stream:true so the character the slice cut
    // in half is dropped rather than raising. A whole file gets no such licence:
    // there, a dangling lead byte is a bad encoding and has to be reported.
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes, {
      stream: truncated,
    });
  } catch {
    // An 8-bit mail says which charset it is, so honour it rather than refusing
    // a standards-valid message. Tried only here, so a modern one is never
    // remapped by a stale declaration. A vCard was tried above and only reaches
    // this line when it named more than one charset.
    const declared = vCardCharsets.length
      ? vCardCharsets
      : declaredEmailCharsets(bytes, fileName);
    if (declared.length === 1) {
      return decodeWithCharset(bytes, declared[0]!, fileName, truncated);
    }
    if (declared.length > 1) {
      // Parts in different encodings: decoding the container as one unit would
      // corrupt all but one of them, and this is not a MIME parser.
      throw new UndecodableTextError(
        fileName,
        `It declares more than one charset (${declared.join(", ")}), and is read as one unit.`,
      );
    }
    // Otherwise a legacy code page, but which one is not knowable from the
    // bytes: the same byte is a different letter in windows-1252, windows-1251
    // and Shift-JIS. Guessing sends confident mojibake, so say so instead.
    throw new UndecodableTextError(fileName);
  }
}

/** Bytes that are not UTF-8 and carry no marker saying what they are. */
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

/** Legacy Word `.dot` and PowerPoint `.pot` templates are OLE compound files,
 *  unlike Graphviz `.dot` and gettext `.pot`. */
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

/** Decode editor text, including the BOM emitted by Windows Registry Editor. */
export async function readTextAttachment(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  return decodeTextAttachmentBytes(bytes, file.name);
}

// Dropped with the File itself, so a removed attachment retains nothing.
const decodedOnce = new WeakMap<File, string>();

/** Decode once per file. The composer decodes while attaching, to report a bad
 *  encoding there, and sending the same file must not read all of it again. */
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
