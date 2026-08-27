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

function declaredGettextCharset(
  bytes: Uint8Array,
  fileName: string,
): string | null {
  if (!/\.(?:po|pot)$/i.test(fileName)) {
    return null;
  }
  const prefix = new TextDecoder("windows-1252").decode(
    bytes.subarray(0, GETTEXT_HEADER_SCAN_BYTES),
  );
  const headerEntry = prefix.match(GETTEXT_HEADER_ENTRY_RE)?.[1];
  const charset = headerEntry?.match(GETTEXT_CHARSET_RE)?.[1];
  return charset && charset.toUpperCase() !== "CHARSET" ? charset : null;
}

// Header block only: a body part can declare its own, but the message-level
// charset is what an 8-bit single-part mail is written in.
const DECLARATION_SCAN_BYTES = 64 * 1024;
// Every Content-Type in the scan, not just the first: a multipart message keeps
// the charset on its parts, and an mbox holds one per message. Folded
// continuation lines start with space or tab.
const EMAIL_CONTENT_TYPE_RE =
  /(?:^|\r?\n)Content-Type:((?:[^\r\n]*)(?:\r?\n[ \t][^\r\n]*)*)/gi;
const EMAIL_CHARSET_RE = /charset[ \t]*=[ \t]*"?([A-Za-z0-9._-]+)"?/i;
// vCard 2.1 puts the encoding on the property: `FN;CHARSET=windows-1252:...`.
const VCARD_CHARSET_RE = /;[ \t]*CHARSET[ \t]*=[ \t]*"?([A-Za-z0-9._-]+)"?/gi;

/** Distinct charsets a container declares about itself, in encounter order. */
function declaredContainerCharsets(
  bytes: Uint8Array,
  fileName: string,
): string[] {
  const isEmail = /\.(?:eml|mbox)$/i.test(fileName);
  const isVCard = /\.vcf$/i.test(fileName);
  if (!isEmail && !isVCard) {
    return [];
  }
  // Headers are ASCII, so this scan cannot fail on the body's own encoding.
  const prefix = new TextDecoder("windows-1252").decode(
    bytes.subarray(0, DECLARATION_SCAN_BYTES),
  );
  const found: string[] = [];
  const add = (charset: string | undefined) => {
    if (!charset) return;
    if (!found.some((seen) => seen.toUpperCase() === charset.toUpperCase())) {
      found.push(charset);
    }
  };
  if (isEmail) {
    for (const header of prefix.matchAll(EMAIL_CONTENT_TYPE_RE)) {
      add(header[1]?.match(EMAIL_CHARSET_RE)?.[1]);
    }
  } else {
    for (const property of prefix.matchAll(VCARD_CHARSET_RE)) {
      add(property[1]);
    }
  }
  return found;
}

/** Decode under a charset the file itself declared, or say why it could not. */
function decodeWithCharset(bytes: Uint8Array, charset: string): string {
  const label = GETTEXT_CHARSET_ALIASES[charset.toUpperCase()] ?? charset;
  try {
    return new TextDecoder(label).decode(bytes);
  } catch (error) {
    if (error instanceof RangeError) {
      throw new Error(
        `Charset "${charset}" isn't supported. Convert the file to UTF-8 before attaching it.`,
      );
    }
    throw error;
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
  if (bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xfe) {
    return new TextDecoder("utf-16le").decode(bytes.subarray(2));
  }
  if (bytes.length >= 2 && bytes[0] === 0xfe && bytes[1] === 0xff) {
    return new TextDecoder("utf-16be").decode(bytes.subarray(2));
  }
  const gettextCharset = declaredGettextCharset(bytes, fileName);
  if (gettextCharset) {
    return decodeWithCharset(bytes, gettextCharset);
  }
  try {
    // A truncated read decodes with stream:true so the character the slice cut
    // in half is dropped rather than raising. A whole file gets no such licence:
    // there, a dangling lead byte is a bad encoding and has to be reported.
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes, {
      stream: truncated,
    });
  } catch {
    // An 8-bit mail or vCard says which charset it is, so honour it rather than
    // refusing a standards-valid file. Tried only after UTF-8 fails, so a modern
    // one is never remapped by a stale declaration.
    const declared = declaredContainerCharsets(bytes, fileName);
    if (declared.length === 1) {
      return decodeWithCharset(bytes, declared[0]!);
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
