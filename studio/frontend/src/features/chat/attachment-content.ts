// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type Unzipped, strFromU8, unzipSync, zipSync } from "fflate";
import {
  TEXT_ATTACHMENT_ACCEPT,
  decodeTextAttachmentBytes,
  isTextAttachmentName,
} from "./text-attachment-accept";

import {
  MAX_OPEN_DOCUMENT_ARCHIVE_BYTES,
  MAX_OPEN_DOCUMENT_XML_BYTES,
  readOpenDocumentAttachmentContent,
} from "./open-document";
import {
  OPEN_DOCUMENT_SPREADSHEET_MIME,
  OPEN_DOCUMENT_TEXT_MIME,
} from "./open-document-accept";

export type AttachmentTextLabel = "PDF" | "DOCX" | "HTML" | "ODS" | "ODT";

export { TEXT_ATTACHMENT_ACCEPT };

export type AttachmentText = {
  label: AttachmentTextLabel | null;
  text: string;
  // True when the file was only read up to the preview cap, so the dialog can
  // say so even if the extracted text ends up short.
  truncated: boolean;
};

const AUDIO_ATTACHMENT_RE =
  /\.(wav|mp3|mp2|m4a|ogg|oga|opus|flac|webm|mp4|aac|aiff|aif|aifc|caf|wma|amr)$/i;
const AUDIO_MIME_RE = /^audio\//i;
const PDF_ATTACHMENT_RE = /\.pdf$/i;
const DOCX_ATTACHMENT_RE = /\.docx$/i;
const HTML_ATTACHMENT_RE = /\.x?html?$/i;
const OPEN_DOCUMENT_ATTACHMENT_RE = /\.(ods|odt)$/i;
const LABELLED_ATTACHMENT_TEXT_RE = /^\[(PDF|DOCX|HTML|ODS|ODT): [^\n]*\]\n/;
const ATTACHMENT_TAG_OPEN_RE = /^<attachment name=[^\n]*>\n/;
const ATTACHMENT_TAG_CLOSE = "\n</attachment>";
// Both wrappers start on the first line, so only a prefix is matched against.
const MAX_ATTACHMENT_WRAPPER_LENGTH = 4096;
const DOCX_MIME =
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
// mammoth picks the parts it parses out of the relationships, not out of the
// filenames, so a target may be called anything ("payload.bin") and still be
// inflated and parsed as XML, while an .xml part nothing points at (customXml,
// the glossary document) is never opened at all. The bound therefore follows
// docx-reader.js: the two package parts it always reads, the main document and
// its relationships, and the five parts resolved out of those relationships,
// each with mammoth's own "word/<name>.xml" fallback. Image targets stay lazy
// and are never read by extractRawText.
const DOCX_CONTENT_TYPES_PART = "[Content_Types].xml";
const DOCX_PACKAGE_RELATIONSHIPS = "_rels/.rels";
const DOCX_RELATIONSHIP_NAMESPACE =
  "http://schemas.openxmlformats.org/officeDocument/2006/relationships/";
const DOCX_MAIN_DOCUMENT_TYPE = `${DOCX_RELATIONSHIP_NAMESPACE}officeDocument`;
const DOCX_RELATED_PART_NAMES = [
  "comments",
  "endnotes",
  "footnotes",
  "numbering",
  "styles",
];
// readXmlFileWithBody opens the relationships of every part it reads a body
// from, so these three carry a .rels part of their own.
const DOCX_BODY_PART_NAMES = new Set(["comments", "endnotes", "footnotes"]);
const DOCX_MAIN_DOCUMENT_FALLBACK = "word/document.xml";
// A <Relationship> tag, with the element prefix mammoth's namespace mapping
// accepts, and skipping any ">" that sits inside an attribute value.
const DOCX_RELATIONSHIP_TAG_RE =
  /<(?:[^\s/>"'=]+:)?Relationship(?=[\s/>])(?:"[^"]*"|'[^']*'|[^"'>])*>/g;
// Attribute names are matched by consuming whole name="value" pairs, so a
// value that itself contains Target= cannot be mistaken for an attribute.
// mammoth reads child.attributes.Target, which a prefixed r:Target never
// populates, so prefixed names are deliberately not accepted here.
const XML_ATTRIBUTE_RE = /([^\s/>"'=]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
/** Non-element markup: a `<Relationship>` inside a comment, CDATA section or processing instruction is text to mammoth's parser, and each ends at its first delimiter the way XML ends it. */
const XML_NON_ELEMENT_RE =
  /<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>|<\?[\s\S]*?\?>/g;
const XML_ENTITY_RE = /&(?:#(\d+)|#[xX]([\da-fA-F]+)|([a-zA-Z]+));/g;
const XML_NAMED_ENTITIES: Record<string, string> = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
};
/**
 * Ceiling on everything a DOCX unpacks to, and the reason mammoth never sees
 * the file the user chose.
 *
 * jszip takes each part's size from the central directory and inflates the part
 * in full before anything can reject it, so an entry declaring 1 KB that
 * expands to 1 GB exhausts the webview. fflate allocates every entry at its
 * declared size and stops there, so mammoth is handed a repack of fflate's
 * output and can only inflate what has already been bounded here.
 */
const MAX_DOCX_UNPACKED_BYTES = 2 * MAX_OPEN_DOCUMENT_ARCHIVE_BYTES;
const AUDIO_EXTENSION_MIMES: Record<string, string> = {
  wav: "audio/wav",
  mp3: "audio/mpeg",
  mp2: "audio/mpeg",
  m4a: "audio/mp4",
  mp4: "audio/mp4",
  ogg: "audio/ogg",
  oga: "audio/ogg",
  opus: "audio/opus",
  flac: "audio/flac",
  aac: "audio/aac",
  aiff: "audio/aiff",
  aif: "audio/aiff",
  aifc: "audio/aiff",
  caf: "audio/x-caf",
  wma: "audio/x-ms-wma",
  amr: "audio/amr",
  webm: "audio/webm",
};
// Node.TEXT_NODE and Node.ELEMENT_NODE, spelled out so the extractor runs where the DOM globals do not.
const TEXT_NODE = 3;
const ELEMENT_NODE = 1;
/** The elements that end a line of rendered text; everything else stays inline. */
const HTML_BLOCK_TAGS = new Set([
  "address",
  "article",
  "aside",
  "blockquote",
  "dd",
  "div",
  "dl",
  "dt",
  "fieldset",
  "figcaption",
  "figure",
  "footer",
  "form",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "header",
  "hr",
  "li",
  "main",
  "nav",
  "ol",
  "p",
  "pre",
  "section",
  "table",
  "tbody",
  "td",
  "tfoot",
  "th",
  "thead",
  "tr",
  "ul",
]);
/** The extensions the text and html previews colour, mapped to shiki ids; one missing here previews unstyled. */
const CODE_ATTACHMENT_LANGUAGES: Record<string, string> = {
  ada: "ada",
  adb: "ada",
  adoc: "asciidoc",
  ads: "ada",
  applescript: "applescript",
  asciidoc: "asciidoc",
  asm: "asm",
  astro: "astro",
  avsc: "json",
  awk: "awk",
  bash: "shellscript",
  bat: "bat",
  bazel: "python",
  bib: "bibtex",
  bicep: "bicep",
  bzl: "python",
  c: "c",
  cabal: "haskell",
  cairo: "cairo",
  cbl: "cobol",
  cc: "cpp",
  cfg: "ini",
  cjs: "javascript",
  cl: "lisp",
  clj: "clojure",
  cljc: "clojure",
  cljs: "clojure",
  cls: "latex",
  cmake: "cmake",
  cmd: "bat",
  cnf: "ini",
  cob: "cobol",
  comp: "glsl",
  conf: "ini",
  containerfile: "docker",
  cpp: "cpp",
  cql: "sql",
  cr: "crystal",
  cs: "csharp",
  csh: "shellscript",
  cshtml: "razor",
  csproj: "xml",
  css: "css",
  cts: "typescript",
  cu: "cuda",
  cuh: "cuda",
  cxx: "cpp",
  d: "d",
  dart: "dart",
  desktop: "ini",
  dhall: "dhall",
  diff: "diff",
  dockerfile: "docker",
  dot: "dot",
  edn: "clojure",
  ejs: "ejs",
  el: "lisp",
  env: "dotenv",
  erb: "erb",
  erl: "erlang",
  ex: "elixir",
  exs: "elixir",
  f: "fortran-free-form",
  f03: "fortran-free-form",
  f90: "fortran-free-form",
  f95: "fortran-free-form",
  feature: "gherkin",
  fish: "fish",
  for: "fortran-free-form",
  frag: "glsl",
  fs: "fsharp",
  fsi: "fsharp",
  fsproj: "xml",
  fsx: "fsharp",
  gd: "gdscript",
  gemspec: "ruby",
  geojson: "json",
  geom: "glsl",
  glsl: "glsl",
  gn: "python",
  gni: "python",
  go: "go",
  gql: "graphql",
  gradle: "groovy",
  graphql: "graphql",
  groovy: "groovy",
  gv: "dot",
  h: "c",
  haml: "haml",
  handlebars: "handlebars",
  har: "json",
  hbs: "handlebars",
  hcl: "hcl",
  hh: "cpp",
  hlsl: "hlsl",
  hpp: "cpp",
  hql: "sql",
  hrl: "erlang",
  hs: "haskell",
  htm: "html",
  html: "html",
  http: "http",
  hx: "haxe",
  hxx: "cpp",
  ini: "ini",
  inl: "cpp",
  ipp: "cpp",
  j2: "jinja",
  jade: "pug",
  java: "java",
  jinja: "jinja",
  jinja2: "jinja",
  jl: "julia",
  js: "javascript",
  json: "json",
  json5: "json5",
  jsonc: "jsonc",
  jsonl: "json",
  jsx: "jsx",
  ksh: "shellscript",
  kt: "kotlin",
  kts: "kotlin",
  latex: "latex",
  less: "less",
  lhs: "haskell",
  liquid: "liquid",
  lisp: "lisp",
  lsp: "lisp",
  lua: "lua",
  mak: "make",
  makefile: "make",
  markdown: "markdown",
  md: "markdown",
  mdx: "mdx",
  mermaid: "mermaid",
  metal: "cpp",
  mjs: "javascript",
  mk: "make",
  ml: "ocaml",
  mli: "ocaml",
  mmd: "mermaid",
  mojo: "mojo",
  move: "move",
  mts: "typescript",
  mustache: "handlebars",
  ndjson: "json",
  nim: "nim",
  nimble: "nim",
  nims: "nim",
  ninja: "ninja",
  nix: "nix",
  njk: "jinja",
  nu: "nushell",
  pas: "pascal",
  php: "php",
  pl: "perl",
  plantuml: "plantuml",
  plist: "xml",
  plsql: "sql",
  pm: "perl",
  podspec: "ruby",
  pp: "pascal",
  prisma: "prisma",
  properties: "ini",
  props: "xml",
  proto: "proto",
  ps1: "powershell",
  psd1: "powershell",
  psm1: "powershell",
  psql: "sql",
  psv: "csv",
  pug: "pug",
  puml: "plantuml",
  pxd: "python",
  py: "python",
  pyi: "python",
  pyx: "python",
  qmd: "markdown",
  qml: "qml",
  r: "r",
  rake: "ruby",
  razor: "razor",
  rb: "ruby",
  reg: "ini",
  rest: "http",
  resx: "xml",
  rkt: "racket",
  rmd: "markdown",
  ron: "rust",
  rs: "rust",
  rst: "rst",
  s: "asm",
  sas: "sas",
  sass: "sass",
  sbt: "scala",
  scala: "scala",
  scm: "scheme",
  scss: "scss",
  service: "ini",
  sh: "shellscript",
  shader: "hlsl",
  sol: "solidity",
  sql: "sql",
  ss: "scheme",
  star: "python",
  starlark: "python",
  sty: "latex",
  styl: "stylus",
  sv: "system-verilog",
  svelte: "svelte",
  svg: "xml",
  svh: "system-verilog",
  swift: "swift",
  targets: "xml",
  tcl: "tcl",
  tcsh: "shellscript",
  tex: "latex",
  tf: "terraform",
  tfstate: "json",
  tfvars: "terraform",
  toml: "toml",
  ts: "typescript",
  tsx: "tsx",
  twig: "twig",
  v: "v",
  vb: "vb",
  vbproj: "xml",
  vbs: "vb",
  vert: "glsl",
  vhd: "vhdl",
  vhdl: "vhdl",
  vue: "vue",
  wgsl: "wgsl",
  xlf: "xml",
  xliff: "xml",
  xml: "xml",
  yaml: "yaml",
  yml: "yaml",
  zig: "zig",
  zsh: "shellscript",
};
// Long attachments still have to render inside a dialog, so the preview stops
// well before the point where a single <pre> stalls the webview.
const MAX_PREVIEW_TEXT_LENGTH = 200_000;
// Text and HTML have no size limit at upload, so a preview reads a bounded slice
// instead of the whole file. Five bytes per character keeps the slice past the
// character cap for any UTF-8 input, so truncation is still detected.
const MAX_PREVIEW_TEXT_BYTES = MAX_PREVIEW_TEXT_LENGTH * 5;

/** Own-property lookup: an extension or entity named "constructor" would otherwise resolve to a member of Object.prototype. */
function lookUp(
  table: Record<string, string>,
  key: string,
): string | undefined {
  return Object.hasOwn(table, key) ? table[key] : undefined;
}

export function isAudioAttachment(
  name: string | undefined,
  contentType: string | undefined,
): boolean {
  return (
    AUDIO_MIME_RE.test(contentType ?? "") ||
    AUDIO_ATTACHMENT_RE.test(name ?? "")
  );
}

// The audio part keeps only the coarse format the backend needs ("mp3" or
// "wav"), so the content type wins, then the extension for uploads the browser
// typed as empty, and the part format only as a last resort.
export function attachmentAudioSrc(
  audio: { data: string; format: string },
  contentType: string | undefined,
  name: string | undefined,
): string {
  const extension = name?.toLowerCase().split(".").pop() ?? "";
  const mime = AUDIO_MIME_RE.test(contentType ?? "")
    ? (contentType as string)
    : (lookUp(AUDIO_EXTENSION_MIMES, extension) ??
      (audio.format === "mp3" ? "audio/mpeg" : "audio/wav"));
  return `data:${mime};base64,${audio.data}`;
}

export function isPdfAttachment(
  name: string | undefined,
  contentType: string | undefined,
): boolean {
  return (
    contentType === "application/pdf" || PDF_ATTACHMENT_RE.test(name ?? "")
  );
}

export function isDocxAttachment(
  name: string | undefined,
  contentType: string | undefined,
): boolean {
  return contentType === DOCX_MIME || DOCX_ATTACHMENT_RE.test(name ?? "");
}

export function isHtmlAttachment(
  name: string | undefined,
  contentType: string | undefined,
): boolean {
  return contentType === "text/html" || HTML_ATTACHMENT_RE.test(name ?? "");
}

export function isOpenDocumentAttachment(
  name: string | undefined,
  contentType: string | undefined,
): boolean {
  return (
    contentType === OPEN_DOCUMENT_SPREADSHEET_MIME ||
    contentType === OPEN_DOCUMENT_TEXT_MIME ||
    OPEN_DOCUMENT_ATTACHMENT_RE.test(name ?? "")
  );
}

// CompositeAttachmentAdapter selects the first accept string that matches.
// Text comes before the document-specific adapters, so previews must apply the
// same MIME-or-extension match before looking at PDF/DOCX/HTML names.
function isTextAttachment(
  name: string,
  contentType: string | undefined,
): boolean {
  const extension = `.${name.split(".").pop()?.toLowerCase() ?? ""}`;
  const mime = contentType?.toLowerCase() ?? "";
  return TEXT_ATTACHMENT_ACCEPT.split(",").some((entry) => {
    const accepted = entry.trim().toLowerCase();
    if (accepted.startsWith(".")) {
      return accepted === extension;
    }
    if (accepted.endsWith("/*")) {
      return mime.startsWith(`${accepted.slice(0, -1)}`);
    }
    return accepted === mime;
  });
}

// unpdf and mammoth decode the whole file on the main thread, so both refuse a
// document past the ceiling the OpenDocument path already enforces, and refuse
// it before the read rather than after. The adapters call this from add() as
// well: the composer clears its text and attachments before it awaits send(),
// so a throw there loses the typed message along with the file.
export function getDocumentAttachmentSizeError(
  file: File,
  label: "PDF" | "DOCX",
): string | null {
  return file.size > MAX_OPEN_DOCUMENT_ARCHIVE_BYTES
    ? `${label} file is too large: ${file.name}`
    : null;
}

export function assertDocumentAttachmentSize(
  file: File,
  label: "PDF" | "DOCX",
): void {
  const error = getDocumentAttachmentSizeError(file, label);
  if (error) {
    throw new Error(error);
  }
}

// mammoth's joinPath: an absolute target drops the base path.
function joinDocxPath(basePath: string, target: string): string {
  const joined = target.startsWith("/")
    ? target
    : [basePath, target].filter(Boolean).join("/");
  return joined.startsWith("/") ? joined.slice(1) : joined;
}

// XML attribute values are entity-decoded by the parser mammoth hands the
// relationships to, so a target only matches the archive once decoded.
function decodeXmlEntities(value: string): string {
  return value.replace(XML_ENTITY_RE, (match, decimal, hex, name) => {
    const code = decimal
      ? Number.parseInt(decimal, 10)
      : hex
        ? Number.parseInt(hex, 16)
        : Number.NaN;
    if (Number.isNaN(code)) {
      return lookUp(XML_NAMED_ENTITIES, name) ?? match;
    }
    return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : match;
  });
}

// The relationship parts are XML, but only their targets are needed, so they
// are scanned rather than parsed: DOMParser is not available where this also
// runs under test, and a malformed rels file is mammoth's to report. The scan
// accepts every attribute form mammoth's parser resolves (both quote styles,
// entity-encoded values, a prefixed element name, ">" inside a value), so a
// crafted rels file cannot hide a target from the bound below.
function readDocxXmlTargets(
  rels: Uint8Array | undefined,
  basePath: string,
): Map<string, string[]> {
  const targets = new Map<string, string[]>();
  if (!rels) {
    return targets;
  }
  const markup = strFromU8(rels).replace(XML_NON_ELEMENT_RE, "");
  for (const tag of markup.match(DOCX_RELATIONSHIP_TAG_RE) ?? []) {
    let type: string | undefined;
    let target: string | undefined;
    for (const [, name, quoted, apostrophed] of tag.matchAll(
      XML_ATTRIBUTE_RE,
    )) {
      const value = quoted ?? apostrophed ?? "";
      if (name === "Type") {
        type = decodeXmlEntities(value);
      } else if (name === "Target") {
        target = decodeXmlEntities(value);
      }
    }
    if (type && target) {
      const resolved = targets.get(type) ?? [];
      resolved.push(joinDocxPath(basePath, target));
      targets.set(type, resolved);
    }
  }
  return targets;
}

// The .rels part that names the XML parts of the part at `path`.
function docxRelationshipsPath(path: string): string {
  const cut = path.lastIndexOf("/");
  const dirname = cut === -1 ? "" : path.slice(0, cut);
  const basename = path.slice(cut + 1);
  return joinDocxPath(dirname, `_rels/${basename}.rels`);
}

type DocxArchive = {
  entries: Unzipped;
  /** Every name the central directory declares, unpacked entries included, so a target resolves the way findPartPath resolves it. */
  names: Set<string>;
  oversized: Set<string>;
};

/**
 * Inflates the archive under fflate's declared-size allocation.
 *
 * An entry past the XML ceiling is left out rather than refused: mammoth opens
 * the package parts and whatever the relationships point at, so a large
 * unreferenced part, custom XML or an embedded image, must still preview.
 * `assertDocxPartSizes` refuses the ones mammoth would have parsed.
 */
function unpackDocxEntries(filename: string, bytes: Uint8Array): DocxArchive {
  const names = new Set<string>();
  const oversized = new Set<string>();
  let unpacked = 0;

  const entries = unzipSync(bytes, {
    filter: (entry) => {
      names.add(entry.name);
      if (entry.originalSize > MAX_OPEN_DOCUMENT_XML_BYTES) {
        oversized.add(entry.name);
        return false;
      }
      unpacked += entry.originalSize;
      if (unpacked > MAX_DOCX_UNPACKED_BYTES) {
        throw new Error(`DOCX file is too large: ${filename}`);
      }
      return true;
    },
  });

  return { entries, names, oversized };
}

/**
 * Refuses the parts mammoth goes on to parse when they exceed the XML ceiling.
 *
 * mammoth takes no entry filter, so the set is resolved the way findPartPaths
 * resolves it: the main document out of "_rels/.rels" and the styles,
 * numbering and note parts out of the document's own .rels, each falling back
 * to a fixed name when no target resolves. Both sides read the same bytes,
 * since mammoth is handed the repack of these entries.
 */
function assertDocxPartSizes(filename: string, archive: DocxArchive): void {
  const { entries, names, oversized } = archive;
  const bound = (path: string) => {
    if (oversized.has(path)) {
      throw new Error(`DOCX XML file is too large: ${filename}:${path}`);
    }
  };
  // findPartPath keeps the first target that exists, and every target it
  // discards is one mammoth never opens.
  const resolve = (targets: string[] | undefined, fallback: string) =>
    targets?.find((path) => names.has(path)) ?? fallback;

  bound(DOCX_CONTENT_TYPES_PART);
  bound(DOCX_PACKAGE_RELATIONSHIPS);
  const mainDocument = resolve(
    readDocxXmlTargets(entries[DOCX_PACKAGE_RELATIONSHIPS], "").get(
      DOCX_MAIN_DOCUMENT_TYPE,
    ),
    DOCX_MAIN_DOCUMENT_FALLBACK,
  );
  bound(mainDocument);
  const mainDocumentRels = docxRelationshipsPath(mainDocument);
  bound(mainDocumentRels);

  const cut = mainDocument.lastIndexOf("/");
  const documentTargets = readDocxXmlTargets(
    entries[mainDocumentRels],
    cut === -1 ? "" : mainDocument.slice(0, cut),
  );
  for (const name of DOCX_RELATED_PART_NAMES) {
    const path = resolve(
      documentTargets.get(`${DOCX_RELATIONSHIP_NAMESPACE}${name}`),
      `word/${name}.xml`,
    );
    bound(path);
    if (DOCX_BODY_PART_NAMES.has(name)) {
      bound(docxRelationshipsPath(path));
    }
  }
}

/** The archive mammoth is given: fflate's own output, so a part that lies about its size arrives truncated rather than inflated in full. */
export function repackDocxAttachmentArchive(
  filename: string,
  bytes: Uint8Array,
): Uint8Array {
  const archive = unpackDocxEntries(filename, bytes);
  assertDocxPartSizes(filename, archive);
  return zipSync(archive.entries, { level: 0 });
}

/**
 * The bytes of a view, as an ArrayBuffer, without copying when it owns one.
 *
 * jszip reads the whole buffer it is handed and looks for the end-of-directory
 * record at its tail, so a view that does not span its buffer would arrive as a
 * corrupt archive. zipSync happens to return an exact-fit array today, which is
 * not something its API promises.
 */
function toArrayBuffer(view: Uint8Array): ArrayBuffer {
  const spansBuffer =
    view.byteOffset === 0 && view.byteLength === view.buffer.byteLength;
  return (
    spansBuffer
      ? view.buffer
      : view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength)
  ) as ArrayBuffer;
}

function isDocxSizeError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.message.startsWith("DOCX file is too large:") ||
      error.message.startsWith("DOCX XML file is too large:"))
  );
}

/**
 * The verdict add() needs before the attachment exists.
 *
 * The composer clears its text and attachments before it awaits send(), so a
 * DOCX that only fails there discards the typed message along with the file.
 * add() calls this instead, the way the audio adapter checks its own ceiling.
 */
export async function getDocxAttachmentError(
  file: File,
): Promise<string | null> {
  const sizeError = getDocumentAttachmentSizeError(file, "DOCX");
  if (sizeError) {
    return sizeError;
  }
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    assertDocxPartSizes(file.name, unpackDocxEntries(file.name, bytes));
  } catch (error) {
    return isDocxSizeError(error)
      ? (error as Error).message
      : `DOCX file could not be read: ${file.name}`;
  }
  return null;
}

export async function extractPdfAttachmentText(file: File): Promise<string> {
  assertDocumentAttachmentSize(file, "PDF");
  const [{ extractText, getDocumentProxy }, buffer] = await Promise.all([
    import("unpdf"),
    file.arrayBuffer().then((bytes) => new Uint8Array(bytes)),
  ]);
  const pdf = await getDocumentProxy(buffer);
  try {
    // per page rather than merged: mergePages folds every newline pdf.js marks into one space
    const { text } = await extractText(pdf);
    return normalizeExtractedText(text.join("\n\n"));
  } finally {
    await pdf.destroy();
  }
}

export async function extractDocxAttachmentText(file: File): Promise<string> {
  assertDocumentAttachmentSize(file, "DOCX");
  const [{ default: mammoth }, buffer] = await Promise.all([
    import("mammoth"),
    file.arrayBuffer(),
  ]);
  const repacked = repackDocxAttachmentArchive(
    file.name,
    new Uint8Array(buffer),
  );
  const { value } = await mammoth.extractRawText({
    arrayBuffer: toArrayBuffer(repacked),
  });
  return value;
}

export function extractHtmlAttachmentText(html: string): string {
  const doc = new DOMParser().parseFromString(html, "text/html");
  for (const el of doc.querySelectorAll("script, style, noscript, template")) {
    el.remove();
  }
  return normalizeExtractedText(collectHtmlBlockText(doc.body));
}

/**
 * Text with the line structure the source had.
 *
 * `textContent` runs a whole page together, so every block-level element and
 * every `<br>` contributes a break of its own and the inline runs between them
 * are joined as written.
 */
function collectHtmlBlockText(node: Node | null): string {
  if (!node) {
    return "";
  }
  if (node.nodeType === TEXT_NODE) {
    return node.nodeValue ?? "";
  }
  if (node.nodeType !== ELEMENT_NODE) {
    return "";
  }

  const element = node as Element;
  const tag = element.tagName.toLowerCase();
  if (tag === "br") {
    return "\n";
  }

  const text = Array.from(element.childNodes)
    .map(collectHtmlBlockText)
    .join("");
  return HTML_BLOCK_TAGS.has(tag) ? `\n${text}\n` : text;
}

/** Collapses the spaces an extractor leaves between positioned runs while keeping the line breaks the source marked. */
function normalizeExtractedText(text: string): string {
  return text
    .replace(/[^\S\n]+/g, " ")
    .replace(/ ?\n ?/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// reads what the matching adapter would send, except html, which previews as raw markup
export async function readAttachmentText(
  file: File,
  name: string,
  contentType: string | undefined,
): Promise<AttachmentText> {
  if (isTextAttachment(name, contentType)) {
    return { label: null, ...(await readBoundedText(file)) };
  }
  if (isPdfAttachment(name, contentType)) {
    return {
      label: "PDF",
      text: await extractPdfAttachmentText(file),
      truncated: false,
    };
  }
  if (isDocxAttachment(name, contentType)) {
    return {
      label: "DOCX",
      text: await extractDocxAttachmentText(file),
      truncated: false,
    };
  }
  // raw markup, not the extraction; kept before opendocument to match the adapters
  if (isHtmlAttachment(name, contentType)) {
    return { label: null, ...(await readBoundedText(file)) };
  }
  if (isOpenDocumentAttachment(name, contentType)) {
    const { label, text } = await readOpenDocumentAttachmentContent(
      file,
      name,
      contentType ?? "",
    );
    return { label, text, truncated: false };
  }
  return { label: null, ...(await readBoundedText(file)) };
}

// Formats that state their own encoding somewhere other than the first bytes: a
// gettext header sits below the translator comments, and a mail or vCard
// declaration below whatever came before it in the archive.
const DECLARES_ITS_CHARSET_RE = /\.(?:po|pot|eml|mbox|vcf)$/i;

async function readBoundedText(
  file: File,
): Promise<{ text: string; truncated: boolean }> {
  const truncated = file.size > MAX_PREVIEW_TEXT_BYTES;
  const slice = truncated ? file.slice(0, MAX_PREVIEW_TEXT_BYTES) : file;
  // Strict decoding belongs to the files the text adapter owns, where refusing
  // is better than sending mojibake to the model. A preview of someone else's
  // file may not be stricter than the adapter that accepted it: .html goes to
  // the HTML adapter, which sends a windows-1252 page happily, and reading the
  // preview through the strict path meant opening one threw where it used to
  // render.
  if (!isTextAttachmentName(file.name)) {
    return { text: await slice.text(), truncated };
  }
  const bytes = new Uint8Array(await slice.arrayBuffer());
  // The declaration can sit past the preview slice, and looking for it inside
  // the slice reported an error for a file the attachment itself decodes. Only
  // the formats that can carry one that far in pay for the second read.
  const whole =
    truncated && DECLARES_ITS_CHARSET_RE.test(file.name)
      ? new Uint8Array(await file.arrayBuffer())
      : bytes;
  return {
    text: decodeTextAttachmentBytes(bytes, file.name, truncated, whole),
    truncated,
  };
}

// A sent attachment keeps only the text its adapter produced, so the preview
// unwraps the adapter's header or tag rather than showing it to the user. The
// stored payload has no size limit, so the wrapper is matched on a prefix and
// only the capped body is copied out.
export function parseAttachmentText(raw: string): AttachmentText {
  const head = raw.slice(0, MAX_ATTACHMENT_WRAPPER_LENGTH);

  const labelled = head.match(LABELLED_ATTACHMENT_TEXT_RE);
  if (labelled) {
    return {
      label: labelled[1] as AttachmentTextLabel,
      ...sliceAttachmentBody(raw, labelled[0].length, raw.length),
    };
  }

  const tagOpen = head.match(ATTACHMENT_TAG_OPEN_RE);
  if (tagOpen && raw.endsWith(ATTACHMENT_TAG_CLOSE)) {
    return {
      label: null,
      ...sliceAttachmentBody(
        raw,
        tagOpen[0].length,
        raw.length - ATTACHMENT_TAG_CLOSE.length,
      ),
    };
  }

  return { label: null, ...sliceAttachmentBody(raw, 0, raw.length) };
}

function sliceAttachmentBody(
  raw: string,
  start: number,
  end: number,
): { text: string; truncated: boolean } {
  const bodyEnd = Math.max(start, end);
  const cappedEnd = Math.min(bodyEnd, start + MAX_PREVIEW_TEXT_LENGTH);
  return {
    text: raw.slice(start, cappedEnd),
    truncated: cappedEnd < bodyEnd,
  };
}

export function truncateAttachmentPreviewText(text: string): {
  text: string;
  truncated: boolean;
} {
  if (text.length <= MAX_PREVIEW_TEXT_LENGTH) {
    return { text, truncated: false };
  }
  return { text: text.slice(0, MAX_PREVIEW_TEXT_LENGTH), truncated: true };
}

/**
 * The shiki language a plain-text attachment previews as, or null for prose.
 *
 * Only the filename decides. Text pulled out of a PDF, a DOCX or a spreadsheet
 * is prose whatever the document was called, so callers pass the extracted
 * label instead of the name in that case.
 */
export function attachmentTextLanguage(
  name: string | undefined,
  label: AttachmentTextLabel | null,
): string | null {
  if (label) {
    return null;
  }
  const lower = name?.toLowerCase() ?? "";
  const extension = lower.includes(".") ? lower.split(".").pop() : lower;
  return lookUp(CODE_ATTACHMENT_LANGUAGES, extension ?? "") ?? null;
}

export function countAttachmentTextLines(text: string): number {
  if (!text) {
    return 0;
  }
  return text.split("\n").length;
}
