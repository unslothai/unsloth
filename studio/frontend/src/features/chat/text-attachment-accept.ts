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
  ".m3u",
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

/** Conventional extensionless names matched through their dotted adapter tokens. */
export const TEXT_ATTACHMENT_BASENAMES = ["containerfile"] as const;
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

/** Tracker MODs put a four-byte format marker at byte offset 1080. */
export async function isBinaryTrackerModule(file: File): Promise<boolean> {
  if (!file.name.toLowerCase().endsWith(".mod") || file.size < 1084) {
    return false;
  }
  const marker = new Uint8Array(await file.slice(1080, 1084).arrayBuffer());
  const magic = String.fromCharCode(...marker);
  return (
    TRACKER_MOD_MAGICS.has(magic) ||
    /^[1-9]CHN$/.test(magic) ||
    /^[1-9][0-9](?:CH|CN)$/.test(magic) ||
    /^TDZ[1-9]$/.test(magic) ||
    /^FA0[4-8]$/.test(magic)
  );
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

export function decodeTextAttachmentBytes(
  bytes: Uint8Array,
  fileName = "",
): string {
  if (bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xfe) {
    return new TextDecoder("utf-16le").decode(bytes.subarray(2));
  }
  if (bytes.length >= 2 && bytes[0] === 0xfe && bytes[1] === 0xff) {
    return new TextDecoder("utf-16be").decode(bytes.subarray(2));
  }
  const gettextCharset = declaredGettextCharset(bytes, fileName);
  if (gettextCharset) {
    const label =
      GETTEXT_CHARSET_ALIASES[gettextCharset.toUpperCase()] ?? gettextCharset;
    try {
      return new TextDecoder(label).decode(bytes);
    } catch (error) {
      if (error instanceof RangeError) {
        throw new Error(
          `Gettext charset "${gettextCharset}" isn't supported. Convert the catalog to UTF-8 before attaching it.`,
        );
      }
      throw error;
    }
  }
  return new TextDecoder().decode(bytes);
}

/** Decode editor text, including the BOM emitted by Windows Registry Editor. */
export async function readTextAttachment(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  return decodeTextAttachmentBytes(bytes, file.name);
}

// MIME is unreliable for source files, so match by extension too.
export const TEXT_ATTACHMENT_ACCEPT = [
  "text/plain,text/markdown,text/csv,text/tab-separated-values,text/xml,text/json,text/css",
  "text/vtt,application/x-subrip,text/x-log,text/calendar,text/vcard,message/rfc822",
  "application/json,application/xml,application/yaml,application/toml,image/svg+xml",
  TEXT_ATTACHMENT_EXTENSIONS.join(","),
].join(",");
