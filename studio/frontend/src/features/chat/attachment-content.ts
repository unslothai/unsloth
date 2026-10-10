// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type Unzipped, strFromU8, strToU8, unzipSync, zipSync } from "fflate";
import {
  MAX_TEXT_ATTACHMENT_BYTES,
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
  RTF_MIMES,
  isRtfAttachmentName,
  isToolOnlyAttachmentName,
} from "./open-document-accept";
import { readRtfAttachmentContent } from "./rtf";

export type AttachmentTextLabel =
  | "PDF"
  | "DOCX"
  | "HTML"
  | "ODS"
  | "ODT"
  | "XLSX"
  | "PPTX"
  | "RTF";

export { TEXT_ATTACHMENT_ACCEPT };

export type AttachmentText = {
  label: AttachmentTextLabel | null;
  text: string;
  // True when the file was only read up to the preview cap, so the dialog can say so even if
  // the extracted text ends up short.
  truncated: boolean;
};

const AUDIO_ATTACHMENT_RE =
  /\.(wav|mp3|mp2|m4a|ogg|oga|opus|flac|webm|mp4|aac|aiff|aif|aifc|caf|wma|amr)$/i;
const AUDIO_MIME_RE = /^audio\//i;
const PDF_ATTACHMENT_RE = /\.pdf$/i;
const DOCX_ATTACHMENT_RE = /\.docx$/i;
const HTML_ATTACHMENT_RE = /\.x?html?$/i;
const OPEN_DOCUMENT_ATTACHMENT_RE = /\.(ods|odt)$/i;
const LABELLED_ATTACHMENT_TEXT_RE = /^\[(PDF|DOCX|HTML|ODS|ODT|XLSX|PPTX|RTF): [^\n]*\]\n/;
const ATTACHMENT_TAG_OPEN_RE = /^<attachment name=[^\n]*>\n/;
const ATTACHMENT_TAG_CLOSE = "\n</attachment>";
// Both wrappers start on the first line, so only a prefix is matched against.
const MAX_ATTACHMENT_WRAPPER_LENGTH = 4096;
const DOCX_MIME =
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
// mammoth picks the parts it parses out of the relationships, not the filenames, so a target
// may be called anything and still be parsed as XML, while an .xml part nothing points at
// is never opened. The bound therefore follows docx-reader.js: the two package parts it
// always reads plus the five resolved out of those relationships, each with mammoth's own
// "word/<name>.xml" fallback. Image targets stay lazy and are never read by extractRawText.
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
// readXmlFileWithBody opens the relationships of every part it reads a body from, so these
// three carry a .rels part of their own.
const DOCX_BODY_PART_NAMES = new Set(["comments", "endnotes", "footnotes"]);
const DOCX_MAIN_DOCUMENT_FALLBACK = "word/document.xml";
// A <Relationship> tag with the element prefix mammoth's namespace mapping accepts, skipping
// any ">" that sits inside an attribute value.
const DOCX_RELATIONSHIP_TAG_RE =
  /<(?:[^\s/>"'=]+:)?Relationship(?=[\s/>])(?:"[^"]*"|'[^']*'|[^"'>])*>/g;
// Attribute names are matched by consuming whole name="value" pairs, so a value containing
// Target= cannot be mistaken for an attribute. mammoth reads child.attributes.Target,
// which a prefixed r:Target never populates, so prefixed names are not accepted.
const XML_ATTRIBUTE_RE = /([^\s/>"'=]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
/** Non-element markup: a `<Relationship>` inside a comment, CDATA section or processing
 *  instruction is text to mammoth's parser, and each ends at its first delimiter. */
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
/** Ceiling on everything a DOCX unpacks to, and why mammoth never sees the file the user
 *  chose. jszip takes each part's size from the central directory and inflates it in full
 *  before anything can reject it, so an entry declaring 1 KB that expands to 1 GB exhausts
 *  the webview. fflate allocates at the declared size and stops, so mammoth is handed a
 *  repack of fflate's output. */
const MAX_DOCX_UNPACKED_BYTES = 2 * MAX_OPEN_DOCUMENT_ARCHIVE_BYTES;
const MAX_DOCX_ENTRIES = 100_000;
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
// Long attachments still render inside a dialog, so the preview stops well before a single
// <pre> stalls the webview.
const MAX_PREVIEW_TEXT_LENGTH = 200_000;
// Text and HTML have no upload size limit, so a preview reads a bounded slice. Five bytes
// per character keeps the slice past the character cap for any UTF-8 input, so truncation
// is still detected.
const MAX_PREVIEW_TEXT_BYTES = MAX_PREVIEW_TEXT_LENGTH * 5;

/** Own-property lookup: an extension or entity named "constructor" would otherwise resolve to
 *  a member of Object.prototype. */
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

// The audio part keeps only the coarse format the backend needs, so the content type wins,
// then the extension for uploads the browser typed as empty, then the part format.
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

export function isRtfAttachment(
  name: string | undefined,
  contentType: string | undefined,
): boolean {
  return (
    RTF_MIMES.includes(contentType?.toLowerCase() ?? "") ||
    isRtfAttachmentName(name ?? "")
  );
}

// CompositeAttachmentAdapter selects the first matching accept string. Text comes before the
// document-specific adapters, so previews must apply the same MIME-or-extension match
// before looking at PDF/DOCX/HTML names.
/** Whether the text adapter claims the file; it runs before the document ones. */
export function isTextAttachment(
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

// unpdf and mammoth decode the whole file on the main thread, so both refuse a document past
// the OpenDocument ceiling, and refuse before the read. The adapters call this from add()
// too: the composer clears text and attachments before awaiting send(), so a throw there
// loses the typed message along with the file.
export function getDocumentAttachmentSizeError(
  file: File,
  label: "PDF" | "DOCX" | "XLSX" | "PPTX",
): string | null {
  return file.size > MAX_OPEN_DOCUMENT_ARCHIVE_BYTES
    ? `${label} file is too large: ${file.name}`
    : null;
}

export function assertDocumentAttachmentSize(
  file: File,
  label: "PDF" | "DOCX" | "XLSX" | "PPTX",
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

// XML attribute values are entity-decoded by the parser mammoth uses, so a target only
// matches the archive once decoded.
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

// The relationship parts are XML, but only their targets are needed, so they are scanned
// rather than parsed: DOMParser is not available under test, and a malformed rels file is
// mammoth's to report. The scan accepts every attribute form mammoth's parser resolves, so
// a crafted rels file cannot hide a target from the bound below.
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
  /** Every name the central directory declares, unpacked entries included, so a target resolves
   *  the way findPartPath resolves it. */
  names: Set<string>;
  oversized: Set<string>;
};

/** Inflates the archive under fflate's declared-size allocation. An entry past the XML ceiling
 *  is left out rather than refused: mammoth opens the package parts and whatever the
 *  relationships point at, so a large unreferenced part must still preview.
 *  `assertDocxPartSizes` refuses the ones mammoth would have parsed. `keepLarge` keeps them
 *  (still marked oversized) for the viewer, which needs large images as well as text. */
const DOCX_IMAGE_PART = /\.(png|jpe?g|gif|bmp|tiff?|emf|wmf|svg|webp)$/i;

function docxPreviewImages(bytes: Uint8Array): { isImage: (name: string) => boolean; used: Set<string> } {
  const names = new Set<string>();
  const read = (name: string) =>
    unzipSync(bytes, {
      filter: (entry) => {
        names.add(entry.name);
        return entry.name === name && entry.originalSize <= MAX_OPEN_DOCUMENT_XML_BYTES;
      },
    })[name];
  const types = read(DOCX_CONTENT_TYPES_PART);
  const defaults = new Map<string, string>();
  const overrides = new Map<string, string>();
  const markup = types ? strFromU8(types).replace(XML_NON_ELEMENT_RE, "") : "";
  for (const [, tag, attributes] of markup.matchAll(/<(?:[\w.-]+:)?(Default|Override)\b([^>]*)>/g)) {
    const values = new Map<string, string>();
    for (const [, key, double, single] of attributes!.matchAll(XML_ATTRIBUTE_RE)) {
      values.set(key!, double ?? single ?? "");
    }
    const type = (values.get("ContentType") ?? "").toLowerCase();
    if (tag === "Default") defaults.set((values.get("Extension") ?? "").toLowerCase(), type);
    else overrides.set((values.get("PartName") ?? "").replace(/^\//, "").toLowerCase(), type);
  }
  const isImage = (name: string) => {
    const dot = name.lastIndexOf(".");
    const type =
      overrides.get(name.toLowerCase()) ?? (dot === -1 ? undefined : defaults.get(name.slice(dot + 1).toLowerCase()));
    return type ? type.startsWith("image/") : DOCX_IMAGE_PART.test(name);
  };
  const targetsOf = (path: string) =>
    readDocxXmlTargets(read(docxRelationshipsPath(path)), path.slice(0, Math.max(0, path.lastIndexOf("/"))));
  const resolve = (targets: string[] | undefined, fallback: string) =>
    targets?.find((path) => names.has(path)) ?? fallback;
  const main = resolve(targetsOf("").get(DOCX_MAIN_DOCUMENT_TYPE), DOCX_MAIN_DOCUMENT_FALLBACK);
  const mainTargets = targetsOf(main);
  const used = new Set([...mainTargets.values()].flat());
  for (const name of DOCX_BODY_PART_NAMES) {
    const path = resolve(mainTargets.get(`${DOCX_RELATIONSHIP_NAMESPACE}${name}`), `word/${name}.xml`);
    for (const target of [...targetsOf(path).values()].flat()) used.add(target);
  }
  return { isImage, used };
}

function unpackDocxEntries(
  filename: string,
  bytes: Uint8Array,
  keepLarge = false,
  skipImages = false,
): DocxArchive {
  const names = new Set<string>();
  const oversized = new Set<string>();
  let count = 0;
  unzipSync(bytes, {
    filter: () => {
      if (++count > MAX_DOCX_ENTRIES) throw new Error(`DOCX file is too large: ${filename}`);
      return false;
    },
  });
  const images = keepLarge ? docxPreviewImages(bytes) : null;
  let unpacked = 0;

  const entries = unzipSync(bytes, {
    filter: (entry) => {
      names.add(entry.name);
      const image = images?.isImage(entry.name) ?? false;
      if (image && (skipImages || !images!.used.has(entry.name))) return false;
      if (entry.originalSize > MAX_OPEN_DOCUMENT_XML_BYTES) {
        oversized.add(entry.name);
        if (!image) return false;
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

/** Refuses the parts mammoth goes on to parse when they exceed the XML ceiling. mammoth takes
 *  no entry filter, so the set is resolved as findPartPaths resolves it, each falling back
 *  to a fixed name when no target resolves. Both sides read the same bytes. */
function assertDocxPartSizes(filename: string, archive: DocxArchive): string {
  const { entries, names, oversized } = archive;
  const bound = (path: string) => {
    if (oversized.has(path)) {
      throw new Error(`DOCX XML file is too large: ${filename}:${path}`);
    }
  };
  // findPartPath keeps the first target that exists, and every target it discards is one mammoth never opens.
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
  return mainDocument;
}

/** The archive mammoth is given: fflate's own output, so a part that lies about its size
 *  arrives truncated rather than inflated in full. */
export function repackDocxAttachmentArchive(
  filename: string,
  bytes: Uint8Array,
): Uint8Array {
  const archive = unpackDocxEntries(filename, bytes);
  assertDocxPartSizes(filename, archive);
  return zipSync(archive.entries, { level: 0 });
}

const WORDPROCESSINGML_NAMESPACES = new Set([
  "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
  "http://purl.oclc.org/ooxml/wordprocessingml/main",
]);
const DOCX_NOTE_BREAKS = new Set(["p", "tab", "br", "cr"]);
const DOCX_NOTE_SKIP = new Set(["del", "moveFrom", "rt", "Fallback"]);

function childElements(node: Node, ns: string, name: string): Element[] {
  return Array.from(node.childNodes).filter(
    (child): child is Element =>
      child.nodeType === 1 &&
      (child as Element).localName === name &&
      (child as Element).namespaceURI === ns,
  );
}

/** An unfilled content control holds Word's prompt, not a value. */
function isDocxPlaceholder(element: Element, ns: string): boolean {
  const flag = childElements(element, ns, "sdtPr").flatMap((pr) =>
    childElements(pr, ns, "showingPlcHdr"),
  )[0];
  return (
    flag !== undefined &&
    !["0", "false", "off"].includes(flag.getAttributeNS(ns, "val") ?? "")
  );
}

function docxNoteText(node: Node, ns: string): string {
  let text = "";
  for (const child of Array.from(node.childNodes)) {
    if (child.nodeType !== 1) continue;
    const element = child as Element;
    const name = element.localName;
    if (DOCX_NOTE_SKIP.has(name)) continue;
    if (name === "sdt" && isDocxPlaceholder(element, ns)) continue;
    if (name === "t" && element.namespaceURI === ns) {
      text += element.textContent ?? "";
    } else if (name === "noBreakHyphen") {
      text += "-";
    } else {
      text +=
        (DOCX_NOTE_BREAKS.has(name) ? " " : "") + docxNoteText(element, ns);
    }
  }
  return text;
}

const OMML_NAMESPACES = new Set([
  "http://schemas.openxmlformats.org/officeDocument/2006/math",
  "http://purl.oclc.org/ooxml/officeDocument/math",
]);
const OMML_ROWS: Record<string, [string, string]> = {
  oMathPara: ["oMath", "\n"],
  eqArr: ["e", "\n"],
  m: ["mr", " \\\\ "],
  mr: ["e", " & "],
};

/** Linear LaTeX-style text for an equation; mirrors the backend's _docx_math_text. */
function docxMathText(element: Element, w: string): string {
  const name = element.localName;
  const ns = element.namespaceURI ?? "";
  if (DOCX_NOTE_SKIP.has(name) || (name === "sdt" && isDocxPlaceholder(element, w))) return "";
  if (ns === w && name === "r") return docxNoteText(element, w);
  if (OMML_NAMESPACES.has(ns) && name === "t") return element.textContent ?? "";
  const children = (key: string) => childElements(element, ns, key);
  const join = (nodes: Element[], sep = "") => nodes.map((node) => docxMathText(node, w)).join(sep);
  const arg = (key: string) =>
    ["0", "false", "off"].includes(prop(`${key}Hide`, "off")) ? join(children(key).slice(0, 1)) : "";
  const prop = (key: string, fallback: string) => {
    const node = children(`${name}Pr`).flatMap((pr) => childElements(pr, ns, key))[0];
    return node ? (node.getAttributeNS(ns, "val") ?? "") : fallback;
  };
  const scripts = (sub: string, sup: string) => (sub ? `_{${sub}}` : "") + (sup ? `^{${sup}}` : "");
  const all = Array.from(element.childNodes).filter((node): node is Element => node.nodeType === 1);
  if (!OMML_NAMESPACES.has(ns)) return join(all);
  switch (name) {
    case "f":
      return prop("type", "bar") === "noBar"
        ? `{${arg("num")} \\atop ${arg("den")}}`
        : `\\frac{${arg("num")}}{${arg("den")}}`;
    case "phant":
      return ["0", "false", "off"].includes(prop("show", "on")) ? "" : join(all);
    case "sSub":
    case "sSup":
    case "sSubSup":
      return arg("e") + scripts(arg("sub"), arg("sup"));
    case "sPre":
      return `{}${scripts(arg("sub"), arg("sup"))}${arg("e")}`;
    case "limLow":
      return arg("e") + scripts(arg("lim"), "");
    case "limUpp":
      return arg("e") + scripts("", arg("lim"));
    case "nary":
      return prop("chr", "∫") + scripts(arg("sub"), arg("sup")) + arg("e");
    case "rad": {
      const deg = arg("deg");
      return deg ? `\\sqrt[${deg}]{${arg("e")}}` : `\\sqrt{${arg("e")}}`;
    }
    case "acc":
      return arg("e") + prop("chr", "̂");
    case "bar":
    case "groupChr": {
      const side = prop("pos", "bot") === "top" ? "over" : "under";
      if (name === "bar") return `\\${side}line{${arg("e")}}`;
      const mark = prop("chr", "⏟");
      if (mark === "⏞" || mark === "⏟") return `\\${side}brace{${arg("e")}}`;
      return `\\${side}set{${mark}}{${arg("e")}}`;
    }
    case "func":
      return `${arg("fName")} ${arg("e")}`;
    case "d":
      return prop("begChr", "(") + join(children("e"), prop("sepChr", "|")) + prop("endChr", ")");
  }
  const row = OMML_ROWS[name];
  return row ? join(children(row[0]), row[1]) : join(all);
}

/** Each equation becomes a plain run where it sits; extractRawText drops OMML. */
export function linearizeDocxMath(archive: Uint8Array): Uint8Array {
  const rewritten: Record<string, Uint8Array> = {};
  const parts = unzipSync(archive, { filter: (entry) => entry.name.endsWith(".xml") });
  for (const [name, bytes] of Object.entries(parts)) {
    const xml = strFromU8(bytes);
    if (!xml.includes("oMath")) continue;
    const doc = new DOMParser().parseFromString(xml, "application/xml");
    const root = doc.documentElement;
    const w = root?.namespaceURI ?? "";
    // Chromium keeps the root of malformed XML and drops everything after the error.
    if (!WORDPROCESSINGML_NAMESPACES.has(w) || doc.getElementsByTagName("parsererror").length) continue;
    const tag = (local: string) => (root.prefix ? `${root.prefix}:${local}` : local);
    let found = false;
    const visit = (node: Node) => {
      for (const child of Array.from(node.childNodes)) {
        if (child.nodeType !== 1) continue;
        const element = child as Element;
        if (
          !OMML_NAMESPACES.has(element.namespaceURI ?? "") ||
          (element.localName !== "oMath" && element.localName !== "oMathPara")
        ) {
          visit(element);
          continue;
        }
        const run = doc.createElementNS(w, tag("r"));
        const text = doc.createElementNS(w, tag("t"));
        text.setAttributeNS("http://www.w3.org/XML/1998/namespace", "xml:space", "preserve");
        text.appendChild(doc.createTextNode(docxMathText(element, w)));
        run.appendChild(text);
        node.replaceChild(run, element);
        found = true;
      }
    };
    visit(doc);
    if (found) rewritten[name] = strToU8(new XMLSerializer().serializeToString(doc));
  }
  if (!Object.keys(rewritten).length) return archive;
  return zipSync({ ...unzipSync(archive), ...rewritten }, { level: 0 });
}

const W14_NAMESPACE = "http://schemas.microsoft.com/office/word/2010/wordml";
const DOCX_BREAK_OR_CHECKBOX_RE = /<(?:[\w.-]+:)?(?:br|cr|checkbox|checkBox)[\s/>]/;

export function writeDocxBreaksAndCheckboxes(archive: Uint8Array): Uint8Array {
  const rewritten: Record<string, Uint8Array> = {};
  const parts = unzipSync(archive, { filter: (entry) => entry.name.endsWith(".xml") });
  for (const [name, bytes] of Object.entries(parts)) {
    const xml = strFromU8(bytes);
    if (!DOCX_BREAK_OR_CHECKBOX_RE.test(xml)) continue;
    const doc = new DOMParser().parseFromString(xml, "application/xml");
    const root = doc.documentElement;
    const w = root?.namespaceURI ?? "";
    if (!WORDPROCESSINGML_NAMESPACES.has(w) || doc.getElementsByTagName("parsererror").length) continue;
    const tag = (local: string) => (root.prefix ? `${root.prefix}:${local}` : local);
    const text = (value: string) => {
      const t = doc.createElementNS(w, tag("t"));
      t.setAttributeNS("http://www.w3.org/XML/1998/namespace", "xml:space", "preserve");
      t.appendChild(doc.createTextNode(value));
      return t;
    };
    const isOn = (flag: Element | undefined, ns: string) =>
      flag !== undefined && !["0", "false", "off"].includes(flag.getAttributeNS(ns, "val") ?? "");
    // [anchor, checked, inRun]: a legacy field's glyph goes inside its run, before the fldChar.
    const boxes: [Element, boolean, boolean][] = [];
    for (const box of Array.from(doc.getElementsByTagNameNS(W14_NAMESPACE, "checkbox"))) {
      const sdt = box.parentNode?.parentNode as Element | null;
      if ((box.parentNode as Element).localName === "sdtPr" && sdt?.localName === "sdt") {
        boxes.push([sdt, isOn(childElements(box, W14_NAMESPACE, "checked")[0], W14_NAMESPACE), false]);
      }
    }
    for (const box of Array.from(doc.getElementsByTagNameNS(w, "checkBox"))) {
      const fldChar = box.parentNode?.parentNode as Element | null;
      if (fldChar?.localName === "fldChar" && fldChar.parentNode) {
        const flag = childElements(box, w, "checked")[0] ?? childElements(box, w, "default")[0];
        boxes.push([fldChar, isOn(flag, w), true]);
      }
    }
    for (const [anchor, checked, inRun] of boxes) {
      const glyph = text(checked ? "☒" : "☐");
      const node = inRun ? glyph : doc.createElementNS(w, tag("r"));
      if (!inRun) node.appendChild(glyph);
      anchor.parentNode?.insertBefore(node, anchor);
    }
    // mammoth drops page and column breaks, which would join surrounding words.
    const breaks = [
      ...Array.from(doc.getElementsByTagNameNS(w, "br")),
      ...Array.from(doc.getElementsByTagNameNS(w, "cr")),
    ];
    for (const br of breaks) br.parentNode?.replaceChild(text("\n"), br);
    if (boxes.length || breaks.length) rewritten[name] = strToU8(new XMLSerializer().serializeToString(doc));
  }
  if (!Object.keys(rewritten).length) return archive;
  return zipSync({ ...unzipSync(archive), ...rewritten }, { level: 0 });
}

const DOCX_TABLE_RE = /<(?:[\w.-]+:)?tbl[\s>]/;

export function writeDocxTableRows(archive: Uint8Array): Uint8Array {
  const rewritten: Record<string, Uint8Array> = {};
  const parts = unzipSync(archive, { filter: (entry) => entry.name.endsWith(".xml") });
  for (const [name, bytes] of Object.entries(parts)) {
    const xml = strFromU8(bytes);
    if (!DOCX_TABLE_RE.test(xml)) continue;
    const doc = new DOMParser().parseFromString(xml, "application/xml");
    const root = doc.documentElement;
    const w = root?.namespaceURI ?? "";
    if (!WORDPROCESSINGML_NAMESPACES.has(w) || doc.getElementsByTagName("parsererror").length) continue;
    const tag = (local: string) => (root.prefix ? `${root.prefix}:${local}` : local);
    const run = (child: Element) => {
      const r = doc.createElementNS(w, tag("r"));
      r.appendChild(child);
      return r;
    };
    const space = () => {
      const t = doc.createElementNS(w, tag("t"));
      t.setAttributeNS("http://www.w3.org/XML/1998/namespace", "xml:space", "preserve");
      t.appendChild(doc.createTextNode(" "));
      return t;
    };
    const tab = () => run(doc.createElementNS(w, tag("tab")));
    const grid = (node: Element, pr: string, name: string) =>
      Math.min(Number(childElements(node, w, pr).flatMap((e) => childElements(e, w, name))[0]?.getAttributeNS(w, "val")) || 0, 1000);
    const outermost = (p: Element, cell: Element) => {
      for (let node = p.parentNode; node && node !== cell; node = node.parentNode) {
        if ((node as Element).localName === "p") return false;
      }
      return true;
    };
    // flatten nested tables before joining their paragraphs into the cell.
    for (const table of Array.from(doc.getElementsByTagNameNS(w, "tbl")).reverse()) {
      for (const row of Array.from(table.getElementsByTagNameNS(w, "tr"))) {
        const cells = Array.from(row.getElementsByTagNameNS(w, "tc"));
        const before = grid(row, "trPr", "gridBefore");
        const after = grid(row, "trPr", "gridAfter");
        if (cells.length + before + after < 2) {
          for (const cell of cells) {
            for (const child of Array.from(cell.childNodes)) {
              if ((child as Element).localName !== "tcPr") table.parentNode?.insertBefore(child, table);
            }
          }
          continue;
        }
        const line = doc.createElementNS(w, tag("p"));
        for (let i = 0; i < before; i++) line.appendChild(tab());
        cells.forEach((cell, index) => {
          if (index) line.appendChild(tab());
          // cell tabs and line breaks, including those in nested tables, would look like column or row separators.
          for (const local of ["tab", "br", "cr"]) {
            for (const mark of Array.from(cell.getElementsByTagNameNS(w, local))) {
              if ((mark.parentNode as Element | null)?.localName === "r") mark.parentNode?.replaceChild(space(), mark);
            }
          }
          for (const t of Array.from(cell.getElementsByTagNameNS(w, "t"))) {
            for (const text of Array.from(t.childNodes)) {
              if (text.nodeValue?.includes("\n")) t.replaceChild(doc.createTextNode(text.nodeValue.replace(/\n/g, " ")), text);
            }
          }
          Array.from(cell.getElementsByTagNameNS(w, "p"))
            .filter((p) => outermost(p, cell))
            .forEach((p, i) => {
              if (i) line.appendChild(run(space()));
              for (const child of Array.from(p.childNodes)) {
                if ((child as Element).localName !== "pPr") line.appendChild(child);
              }
            });
          for (let i = 1; i < (grid(cell, "tcPr", "gridSpan") || 1); i++) line.appendChild(tab());
        });
        for (let i = 0; i < after; i++) line.appendChild(tab());
        table.parentNode?.insertBefore(line, table);
      }
      table.parentNode?.removeChild(table);
    }
    rewritten[name] = strToU8(new XMLSerializer().serializeToString(doc));
  }
  if (!Object.keys(rewritten).length) return archive;
  return zipSync({ ...unzipSync(archive), ...rewritten }, { level: 0 });
}

function listNumber(n: number, format: string | undefined): string {
  if (format === "none") return "";
  const lower = format?.startsWith("lower");
  // Out-of-range counters read as decimals, like CSS (roman stops at 3999); this also bounds the loops below.
  if (n < 1 || n > (format?.endsWith("Roman") ? 3999 : 32767) || !Number.isInteger(n) || !(lower || format?.startsWith("upper"))) {
    return format === "decimalZero" && n >= 0 && n < 10 ? `0${n}` : String(n);
  }
  let text = "";
  if (format!.endsWith("Roman")) text = romanNumeral(n);
  // CSS lower-alpha is bijective (z, aa, ab); Word's lowerLetter repeats the letter (z, aa, bb).
  else if (format!.endsWith("Alpha")) for (let k = n; k > 0; k = Math.floor((k - 1) / 26)) text = String.fromCharCode(97 + ((k - 1) % 26)) + text;
  else text = String.fromCharCode(97 + ((n - 1) % 26)).repeat(Math.ceil(n / 26));
  return lower ? text : text.toUpperCase();
}

function wordValue(node: Element | undefined, name: string): string | undefined {
  const ns = node?.namespaceURI ?? "";
  return (node && childElements(node, ns, name)[0]?.getAttributeNS(ns, "val")) || undefined;
}

// Labels are an extra: a numbering part this cannot read leaves the text as Mammoth reads it.
export function writeDocxListNumbers(archive: Uint8Array): Uint8Array {
  try {
    return injectDocxListNumbers(archive);
  } catch {
    return archive;
  }
}

function injectDocxListNumbers(archive: Uint8Array): Uint8Array {
  const parts = unzipSync(archive, { filter: (entry) => /\.(?:xml|rels)$/.test(entry.name) });
  const resolve = (targets: string[] | undefined, fallback: string) =>
    targets?.find((path) => Object.hasOwn(parts, path)) ?? fallback;
  const main = resolve(readDocxXmlTargets(parts[DOCX_PACKAGE_RELATIONSHIPS], "").get(DOCX_MAIN_DOCUMENT_TYPE), DOCX_MAIN_DOCUMENT_FALLBACK);
  const targets = readDocxXmlTargets(parts[docxRelationshipsPath(main)], main.slice(0, Math.max(0, main.lastIndexOf("/"))));
  const parse = (path: string) => {
    const bytes = Object.hasOwn(parts, path) ? parts[path] : undefined;
    if (!bytes) return null;
    const doc = new DOMParser().parseFromString(strFromU8(bytes), "application/xml");
    const root = doc.documentElement;
    if (!WORDPROCESSINGML_NAMESPACES.has(root?.namespaceURI ?? "") || doc.getElementsByTagName("parsererror").length) return null;
    return { doc, root, w: root.namespaceURI ?? "" };
  };
  const related = (name: string) => parse(resolve(targets.get(`${DOCX_RELATIONSHIP_NAMESPACE}${name}`), `word/${name}.xml`));
  const numbering = related("numbering");
  const body = numbering && parse(main);
  if (!numbering || !body) return archive;

  const byId = (parent: Element, ns: string, name: string, id: string) =>
    new Map(childElements(parent, ns, name).map((node) => [node.getAttributeNS(ns, id) ?? "", node]));
  const n = numbering.w;
  const abstracts = byId(numbering.root, n, "abstractNum", "abstractNumId");
  const nums = byId(numbering.root, n, "num", "numId");
  const styles = related("styles");
  const styleById = styles ? byId(styles.root, styles.w, "style", "styleId") : new Map<string, Element>();
  const styleNumPr = (id: string | undefined, depth = 0): Element | undefined => {
    const style = id === undefined ? undefined : styleById.get(id);
    if (!style || depth > 20) return undefined;
    const pPr = childElements(style, style.namespaceURI ?? "", "pPr")[0];
    return (pPr && childElements(pPr, style.namespaceURI ?? "", "numPr")[0]) ?? styleNumPr(wordValue(style, "basedOn"), depth + 1);
  };

  type Level = { lvl?: Element; start: number; format?: string; restart?: string; legal: boolean; text: string };
  const firstByLevel = (nodes: Element[]) => {
    const byLevel = new Map<string, Element>();
    for (const node of nodes) {
      const at = node.getAttributeNS(n, "ilvl") ?? "";
      if (!byLevel.has(at)) byLevel.set(at, node);
    }
    return byLevel;
  };
  // Each instance's nine levels are resolved once, not per paragraph.
  type Instance = { abstractId: string; levels: Level[]; restarts: number[] };
  const instances = new Map<string, Instance | undefined>();
  const instance = (numId: string): Instance | undefined => {
    if (instances.has(numId)) return instances.get(numId);
    const num = nums.get(numId);
    const abstractId = wordValue(num, "abstractNumId") ?? "";
    const abstract = abstracts.get(abstractId);
    let resolved: Instance | undefined;
    if (num && abstract) {
      const overrides = firstByLevel(childElements(num, n, "lvlOverride"));
      const defined = firstByLevel(childElements(abstract, n, "lvl"));
      const levels = Array.from({ length: 9 }, (_, index): Level => {
        const override = overrides.get(String(index));
        const lvl = (override && childElements(override, n, "lvl")[0]) ?? defined.get(String(index));
        return {
          lvl,
          start: Number(wordValue(override, "startOverride") ?? wordValue(lvl, "start") ?? 0) || 0,
          format: wordValue(lvl, "numFmt"),
          restart: wordValue(lvl, "lvlRestart"),
          // isLgl (legal numbering) shows every level's number in Arabic digits: "Section 1.01" under "Article I".
          legal: !!lvl && childElements(lvl, n, "isLgl").some((node) => !/^(?:0|false|off)$/.test(node.getAttributeNS(n, "val") ?? "")),
          text: wordValue(lvl, "lvlText") ?? "",
        };
      });
      const restarts = Array.from(overrides.entries())
        .filter(([at, node]) => /^[0-8]$/.test(at) && childElements(node, n, "startOverride").length)
        .map(([at]) => Number(at));
      resolved = { abstractId, levels, restarts };
    }
    instances.set(numId, resolved);
    return resolved;
  };
  const W15 = "http://schemas.microsoft.com/office/word/2012/wordml";
  const restartsAfterBreak = new Set(
    Array.from(abstracts.entries())
      .filter(([, node]) => /^(?:1|true|on)$/.test(node.getAttributeNS(W15, "restartNumberingAfterBreak") ?? ""))
      .map(([id]) => id),
  );

  const counters = new Map<string, (number | undefined)[]>();
  const started = new Set<string>();
  const { doc, w } = body;
  const tag = (local: string) => (body.root.prefix ? `${body.root.prefix}:${local}` : local);
  let found = false;
  const label = (p: Element, pPr: Element | undefined) => {
    // A tracked-deleted paragraph mark removes the item; Mammoth folds its text into the next paragraph.
    const mark = pPr && childElements(pPr, w, "rPr")[0];
    if (mark && (childElements(mark, w, "del").length || childElements(mark, w, "moveFrom").length)) return;
    const direct = pPr && childElements(pPr, w, "numPr")[0];
    const styled = styleNumPr(wordValue(pPr, "pStyle"));
    const numId = wordValue(direct, "numId") ?? wordValue(styled, "numId");
    const list = numId === undefined ? undefined : instance(numId);
    if (!list) return;
    const { abstractId, levels, restarts } = list;
    const ilvl = Math.min(8, Math.max(0, Math.trunc(Number(wordValue(direct, "ilvl") ?? wordValue(styled, "ilvl") ?? 0) || 0)));
    const { lvl, format, legal, text } = levels[ilvl];
    if (!lvl) return;
    // Instances of one abstract definition share its counters, as in Word; a start override
    // restarts them once, when its instance is first used.
    const counts = counters.get(abstractId) ?? [];
    counters.set(abstractId, counts);
    if (!started.has(numId!)) {
      started.add(numId!);
      for (const at of restarts) if (at < counts.length) counts.length = at;
    }
    for (let i = 0; i < ilvl; i++) counts[i] ??= levels[i].start;
    const current = counts[ilvl];
    counts[ilvl] = current === undefined ? levels[ilvl].start : current + 1;
    // A deeper level restarts after any shallower one unless lvlRestart (1-based, 0 = never) says otherwise.
    for (let i = ilvl + 1; i < counts.length; i++) {
      const restart = levels[i].restart;
      if (restart === undefined || ilvl < Number(restart)) counts[i] = undefined;
    }
    // Word caps a number format far below this; a longer one is not a label.
    if (format === "bullet" || text.length > 256) return;
    const value = text.replace(/%([1-9])/g, (_, digit: string) => {
      const { start, format } = levels[Number(digit) - 1];
      return listNumber(counts[Number(digit) - 1] ?? start, legal && format !== "none" && !format?.startsWith("decimal") ? "decimal" : format);
    });
    if (!value.trim()) return;
    const run = doc.createElementNS(w, tag("r"));
    const t = doc.createElementNS(w, tag("t"));
    t.setAttributeNS("http://www.w3.org/XML/1998/namespace", "xml:space", "preserve");
    t.appendChild(doc.createTextNode(`${value} `));
    run.appendChild(t);
    p.insertBefore(run, pPr ? pPr.nextSibling : p.firstChild);
    found = true;
  };
  for (const p of Array.from(doc.getElementsByTagNameNS(w, "p"))) {
    const pPr = childElements(p, w, "pPr")[0];
    label(p, pPr);
    // A section break restarts the lists that opt in (Word's "restart numbering after break").
    if (pPr && childElements(pPr, w, "sectPr").length) for (const id of restartsAfterBreak) counters.delete(id);
  }
  if (!found) return archive;
  return zipSync({ ...unzipSync(archive), [main]: strToU8(new XMLSerializer().serializeToString(doc)) }, { level: 0 });
}

const DOCX_NOTE_REFERENCE_RE =
  /<(?:([\w.-]+):)?(footnote|endnote)Reference\b([^>]*?)(\/?)>(\s*<\/(?:[\w.-]+:)?\2Reference\s*>)?/g;
const DOCX_NOTE_ID_RE = /(?:^|\s)(?:[\w.-]+:)?id\s*=\s*["']([^"']*)["']/;

function romanNumeral(n: number): string {
  let out = "";
  for (const [value, digits] of [
    [1000, "m"],
    [900, "cm"],
    [500, "d"],
    [400, "cd"],
    [100, "c"],
    [90, "xc"],
    [50, "l"],
    [40, "xl"],
    [10, "x"],
    [9, "ix"],
    [5, "v"],
    [4, "iv"],
    [1, "i"],
  ] as const) {
    for (; n >= value; n -= value) out += digits;
  }
  return out;
}


/** Sentinels after note references; `label` numbers those extractRawText kept (1, 2 / i, ii) and appends the notes. */
export function markDocxNotes(archive: Uint8Array): {
  archive: Uint8Array;
  label: (text: string) => string;
} {
  const names = new Set<string>();
  const read = (name: string) =>
    unzipSync(archive, {
      filter: (entry) => {
        names.add(entry.name);
        return entry.name === name;
      },
    })[name];
  const targetsOf = (path: string) =>
    readDocxXmlTargets(
      read(docxRelationshipsPath(path)),
      path.slice(0, Math.max(0, path.lastIndexOf("/"))),
    );
  const resolve = (targets: string[] | undefined, fallback: string) =>
    targets?.find((path) => names.has(path)) ?? fallback;
  const main = resolve(
    targetsOf("").get(DOCX_MAIN_DOCUMENT_TYPE),
    DOCX_MAIN_DOCUMENT_FALLBACK,
  );
  const mainTargets = targetsOf(main);
  const notesOf = (heading: string, label: (n: number) => string) => ({
    heading,
    label,
    ns: "",
    bodies: new Map<string, string>(),
    referenced: new Set<string>(),
  });
  const kinds = {
    footnote: notesOf("Footnotes", String),
    endnote: notesOf("Endnotes", romanNumeral),
  };
  for (const [kind, notes] of Object.entries(kinds)) {
    const xml = read(
      resolve(
        mainTargets.get(`${DOCX_RELATIONSHIP_NAMESPACE}${kind}s`),
        `word/${kind}s.xml`,
      ),
    );
    if (!xml) continue;
    const doc = new DOMParser().parseFromString(
      strFromU8(xml),
      "application/xml",
    );
    const ns = doc.documentElement?.namespaceURI ?? "";
    if (!WORDPROCESSINGML_NAMESPACES.has(ns)) continue;
    notes.ns = ns;
    for (const note of Array.from(doc.getElementsByTagNameNS(ns, kind))) {
      const type = note.getAttributeNS(ns, "type");
      if (type && type !== "normal") continue;
      notes.bodies.set(
        note.getAttributeNS(ns, "id") ?? "",
        docxNoteText(note, ns).replace(/\s+/g, " ").trim(),
      );
    }
  }
  if (!kinds.footnote.bodies.size && !kinds.endnote.bodies.size) {
    return { archive, label: (text) => text };
  }

  const refs: { kind: keyof typeof kinds; id: string }[] = [];
  // Nonce: document text shaped like a sentinel stays as written.
  const nonce = Math.random().toString(36).slice(2, 10);
  const sentinel = new RegExp(`\\uE000${nonce}\\.(\\d+)\\uE001`, "g");
  const mainXml = read(main);
  const markedXml = mainXml
    ? strFromU8(mainXml).replace(
        DOCX_NOTE_REFERENCE_RE,
        (
          reference,
          _prefix: string | undefined,
          kind: keyof typeof kinds,
          attributes: string,
          selfClosing: string,
          endTag: string | undefined,
        ) => {
          const id = DOCX_NOTE_ID_RE.exec(attributes)?.[1];
          const notes = kinds[kind];
          // Never write inside a reference whose end tag is not right after its start.
          if (!selfClosing && !endTag) return reference;
          if (id === undefined || !notes.bodies.has(id)) return reference;
          notes.referenced.add(id);
          refs.push({ kind, id });
          const marker = `\uE000${nonce}.${refs.length - 1}\uE001`;
          // Own xmlns: the reference's prefix may be declared on the reference alone.
          return `${reference}<t xmlns="${notes.ns}">${marker}</t>`;
        },
      )
    : "";
  const label = (text: string) => {
    const numbers = {
      footnote: new Map<string, number>(),
      endnote: new Map<string, number>(),
    };
    const body = text.replace(sentinel, (match, index: string) => {
      const ref = refs[Number(index)];
      if (!ref) return match;
      const { kind, id } = ref;
      const seen = numbers[kind];
      if (!seen.has(id)) seen.set(id, seen.size + 1);
      return `[${kinds[kind].label(seen.get(id)!)}]`;
    });
    const sections: string[] = [];
    for (const [kind, notes] of Object.entries(kinds)) {
      const seen = numbers[kind as keyof typeof kinds];
      // Unreferenced notes stay; ones referenced only from deleted or moved text go.
      for (const id of notes.bodies.keys()) {
        if (!notes.referenced.has(id) && !seen.has(id)) {
          seen.set(id, seen.size + 1);
        }
      }
      const lines = [...seen]
        .sort((a, b) => a[1] - b[1])
        .filter(([id]) => notes.bodies.get(id))
        .map(
          ([id, number]) => `[${notes.label(number)}] ${notes.bodies.get(id)}`,
        );
      if (lines.length) sections.push([notes.heading, ...lines].join("\n"));
    }
    return body + sections.join("\n\n");
  };
  if (!refs.length) return { archive, label };
  const entries = unzipSync(archive);
  entries[main] = strToU8(markedXml);
  return { archive: zipSync(entries, { level: 0 }), label };
}

const XML_TOKEN_RE =
  /<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>|<[?!][\s\S]*?>|<(\/?)([^\s/>]+)(?:\s+[^\s=/>]+\s*=\s*(?:"[^"]*"|'[^']*'))*\s*(\/?)>/g;
const PARAGRAPH_RE = /<(?:[\w.-]+:)?p[\s/>]/;

function cutDocxParagraphs(xml: string, max: number): string | null {
  const open: string[] = [];
  let count = 0;
  XML_TOKEN_RE.lastIndex = 0;
  for (let match = XML_TOKEN_RE.exec(xml); match; match = XML_TOKEN_RE.exec(xml)) {
    const [, closing, name, empty] = match;
    if (!name) continue;
    if (!closing && !empty) {
      open.push(name);
      continue;
    }
    if (closing) open.pop();
    if (/(?:^|:)p$/.test(name) && ++count === max) {
      const end = XML_TOKEN_RE.lastIndex;
      if (!PARAGRAPH_RE.test(xml.slice(end))) return null;
      return `${xml.slice(0, end)}${open.reverse().map((tag) => `</${tag}>`).join("")}`;
    }
  }
  return null;
}

export function repackDocxPreviewArchive(
  filename: string,
  bytes: Uint8Array,
  maxParagraphs: number,
  { keptImagesOnly = false } = {},
): { archive: Uint8Array; truncated: boolean } {
  const archive = unpackDocxEntries(filename, bytes, true, keptImagesOnly);
  const mainDocument = assertDocxPartSizes(filename, archive);
  const main = archive.entries[mainDocument];
  const cut = main ? cutDocxParagraphs(strFromU8(main), maxParagraphs) : null;
  if (cut !== null) archive.entries[mainDocument] = strToU8(cut);
  if (keptImagesOnly && main) {
    addKeptDocxImages(bytes, archive, mainDocument, cut ?? strFromU8(main));
  }
  return { archive: zipSync(archive.entries, { level: 0 }), truncated: cut !== null };
}

// Element tags only, skipping any ">" inside an attribute value.
const XML_ELEMENT_TAG_RE = /<[^\s/>!?](?:"[^"]*"|'[^']*'|[^"'>])*>/g;
// Relationship id attributes, whatever their prefix.
const DOCX_RELATIONSHIP_ID_ATTRIBUTE_RE = /^[\w.-]+:(?:embed|link|id)$/;
const DOCX_IMAGE_RELATIONSHIP_TYPE_RE = /\/image$/;

type DocxRelationship = { id: string; type: string; path: string };

function docxRelationships(rels: Uint8Array | undefined, base: string): DocxRelationship[] {
  if (!rels) return [];
  const list: DocxRelationship[] = [];
  const markup = strFromU8(rels).replace(XML_NON_ELEMENT_RE, "");
  for (const tag of markup.match(DOCX_RELATIONSHIP_TAG_RE) ?? []) {
    let id = "";
    let target = "";
    let type = "";
    for (const [, name, double, single] of tag.matchAll(XML_ATTRIBUTE_RE)) {
      const value = decodeXmlEntities(double ?? single ?? "");
      if (name === "Id") id = value;
      else if (name === "Target") target = value;
      else if (name === "Type") type = value;
    }
    if (id && target) list.push({ id, type, path: joinDocxPath(base, target) });
  }
  return list;
}

/** Each element tag's local name and attributes; comments, CDATA and text are skipped. */
function* docxElementTags(xml: string): Generator<{ local: string; attributes: Map<string, string> }> {
  for (const tag of xml.replace(XML_NON_ELEMENT_RE, "").match(XML_ELEMENT_TAG_RE) ?? []) {
    const name = /^<([^\s/>]+)/.exec(tag)![1]!;
    const attributes = new Map<string, string>();
    for (const [, key, double, single] of tag.matchAll(XML_ATTRIBUTE_RE)) {
      attributes.set(key!, decodeXmlEntities(double ?? single ?? ""));
    }
    yield { local: name.slice(name.indexOf(":") + 1), attributes };
  }
}

function relationshipIdsIn(xml: string): Set<string> {
  const ids = new Set<string>();
  for (const { attributes } of docxElementTags(xml)) {
    for (const [name, value] of attributes) if (DOCX_RELATIONSHIP_ID_ATTRIBUTE_RE.test(name)) ids.add(value);
  }
  return ids;
}

const DOCX_NOTE_REFERENCES: Record<string, string> = {
  footnoteReference: "footnote",
  endnoteReference: "endnote",
  commentReference: "comment",
};

/** Inflates only the images the kept body, and the notes it refers to, reference. */
function addKeptDocxImages(bytes: Uint8Array, archive: DocxArchive, mainDocument: string, body: string): void {
  const dirname = (path: string) => path.slice(0, Math.max(0, path.lastIndexOf("/")));
  const wanted = new Set<string>();
  const addImages = (relationships: DocxRelationship[], ids: Set<string>) => {
    for (const rel of relationships) {
      // Images only: an oversized altChunk or OLE part the first pass left out stays out.
      if (ids.has(rel.id) && DOCX_IMAGE_RELATIONSHIP_TYPE_RE.test(rel.type)) wanted.add(rel.path);
    }
  };
  const mainRels = docxRelationships(archive.entries[docxRelationshipsPath(mainDocument)], dirname(mainDocument));
  addImages(mainRels, relationshipIdsIn(body));
  // Notes the kept body refers to, by element name ("footnote") and id.
  const notes = new Map<string, Set<string>>();
  for (const { local, attributes } of docxElementTags(body)) {
    const note = DOCX_NOTE_REFERENCES[local];
    const id = [...attributes].find(([name]) => name === "id" || name.endsWith(":id"))?.[1];
    if (note && id !== undefined) notes.set(note, (notes.get(note) ?? new Set()).add(id));
  }
  for (const [note, noteIds] of notes) {
    const part = mainRels.find((rel) => rel.type.endsWith(`/${note}s`))?.path;
    const xml = part ? archive.entries[part] : undefined;
    if (!part || !xml) continue;
    // Just the referenced notes' elements, not the whole part.
    const element = new RegExp(`<([\\w.-]+:)?${note}(?=[\\s/>])(?:"[^"]*"|'[^']*'|[^"'>])*>[\\s\\S]*?</\\1?${note}>`, "g");
    const kept = [...strFromU8(xml).replace(XML_NON_ELEMENT_RE, "").matchAll(element)]
      .map(([whole]) => whole)
      .filter((whole) => {
        const open = docxElementTags(whole).next().value;
        const id = open && [...open.attributes].find(([name]) => name === "id" || name.endsWith(":id"))?.[1];
        return id !== undefined && noteIds.has(id);
      });
    addImages(docxRelationships(archive.entries[docxRelationshipsPath(part)], dirname(part)), relationshipIdsIn(kept.join("")));
  }
  if (wanted.size === 0) return;
  // One budget with the parts already unpacked.
  let unpacked = Object.values(archive.entries).reduce((total, entry) => total + entry.length, 0);
  const images = unzipSync(bytes, {
    filter: (entry) => {
      if (!wanted.has(entry.name) || archive.entries[entry.name]) return false;
      if (unpacked + entry.originalSize > MAX_DOCX_UNPACKED_BYTES) return false;
      unpacked += entry.originalSize;
      return true;
    },
  });
  Object.assign(archive.entries, images);
}

/** The bytes of a view, as an ArrayBuffer, without copying when it owns one. jszip reads the
 *  whole buffer and looks for the end-of-directory record at its tail, so a view that does
 *  not span its buffer would arrive as a corrupt archive. */
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

/** The verdict add() needs before the attachment exists. The composer clears its text and
 *  attachments before it awaits send(), so a DOCX that only fails there discards the typed
 *  message along with the file. */
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

export function getPdfAttachmentTextError(
  fileName: string,
  text: string,
  pythonToolOpensFile: boolean,
): string | null {
  return text || pythonToolOpensFile
    ? null
    : `PDF has no readable text: ${fileName}. Scanned pages can't be read.`;
}

function pdfFormFieldLines(
  annotations: {
    fieldType?: string;
    fieldName?: string;
    fieldValue?: unknown;
    alternativeText?: string;
    radioButton?: boolean;
    hidden?: boolean;
    password?: boolean;
    options?: { exportValue?: unknown; displayValue?: unknown }[];
  }[],
): string[] {
  const fields = new Map<string, string>();
  for (const {
    fieldType,
    fieldName,
    fieldValue,
    alternativeText,
    radioButton,
    hidden,
    password,
    options,
  } of annotations) {
    const value = [fieldValue]
      .flat()
      .filter((part) => typeof part === "string")
      .map((part) => {
        const shown = options?.find((option) => option.exportValue === part);
        return typeof shown?.displayValue === "string"
          ? shown.displayValue
          : part;
      })
      .join(", ");
    const unchecked = fieldType === "Btn" && value === "Off";
    if (!fieldName || !value.trim() || unchecked || hidden || password) {
      continue;
    }
    // The tooltip (/TU) is the human label behind codes like f1_01[0]; a radio
    // widget's tooltip names one option, not the group's selected value.
    const tooltip = radioButton ? "" : alternativeText;
    const label = tooltip?.replace(/\s+/g, " ").trim() || fieldName;
    fields.set(fieldName, `${label}: ${value}`);
  }
  return [...fields.values()];
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
    // getAnnotations re-reads the text under each link, so only forms call it
    const hasFields = await pdf.getFieldObjects().then(Boolean, () => false);
    const pages = await Promise.all(
      text.map(async (pageText, index) => {
        const annotations = hasFields
          ? await pdf
              .getPage(index + 1)
              .then((page) => page.getAnnotations())
              .catch(() => [])
          : [];
        const fields = pdfFormFieldLines(annotations);
        return [pageText, ...fields].filter(Boolean).join("\n");
      }),
    );
    return normalizeExtractedText(pages.join("\n\n"));
  } finally {
    await pdf.destroy();
  }
}

// A text attachment's limit, in UTF-8 bytes. Cells are not capped, so the total must be.
const MAX_OFFICE_TEXT_BYTES = MAX_TEXT_ATTACHMENT_BYTES;

function utf8Within(text: string, limit: number): { bytes: number; end: number } {
  let bytes = 0;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    const pair = code >= 0xd800 && code < 0xdc00 && i + 1 < text.length;
    const size = code < 0x80 ? 1 : code < 0x800 ? 2 : pair ? 4 : 3;
    if (bytes + size > limit) return { bytes, end: i };
    bytes += size;
    if (pair) i++;
  }
  return { bytes, end: text.length };
}

function textBudget(limit: number) {
  let left = limit;
  let cut = false;
  return {
    take(text: string): string {
      if (cut) return "";
      const { bytes, end } = utf8Within(text, Math.max(0, left - 1));
      if (end < text.length) cut = true;
      left -= bytes + 1;
      return text.slice(0, end);
    },
    get cut() {
      return cut;
    },
  };
}

export async function extractOfficeAttachmentText(
  file: File,
  label: "XLSX" | "PPTX",
): Promise<string> {
  assertDocumentAttachmentSize(file, label);
  const [{ MAX_SHEET_COLUMNS, MAX_SHEET_ROWS, readPptx, readXlsx }, buffer] = await Promise.all([
    import("@/components/file-viewer/office"),
    file.arrayBuffer(),
  ]);
  const bytes = new Uint8Array(buffer);
  const budget = textBudget(MAX_OFFICE_TEXT_BYTES);
  const parts: string[] = [];
  if (label === "PPTX") {
    const deck = readPptx(bytes, { images: false });
    for (const [index, slide] of deck.slides.entries()) {
      if (budget.cut) break;
      const lines = [budget.take(`Slide ${index + 1}`)];
      for (const box of slide.boxes) {
        for (const p of box.paragraphs ?? []) lines.push(budget.take(p.text));
        if (box.caption) lines.push(budget.take(box.caption));
        for (const row of box.table ?? []) lines.push(row.map((cell) => budget.take(cell)).join("\t"));
      }
      parts.push(lines.join("\n"));
    }
    if (deck.truncated && !budget.cut) {
      parts.push(`[Truncated: only the first ${deck.slides.length} slides are included.]`);
    }
  } else {
    for (const sheet of readXlsx(bytes)) {
      if (budget.cut) break;
      const lines = [budget.take(`Sheet: ${sheet.name}`)];
      for (const row of sheet.rows) {
        if (!row || budget.cut) continue;
        const line = Array.from(row, (cell) => budget.take(cell?.text ?? ""))
          .filter((_, index) => !sheet.hidden?.columns.has(index))
          .join("\t")
          .trimEnd();
        if (line) lines.push(line);
      }
      if (sheet.truncated) {
        lines.push(
          `[Truncated: only part of the workbook is included, at most ${MAX_SHEET_ROWS} rows and ${MAX_SHEET_COLUMNS} columns per sheet.]`,
        );
      }
      parts.push(lines.join("\n"));
    }
  }
  if (budget.cut) {
    parts.push(`[Truncated: the text stops after ${MAX_OFFICE_TEXT_BYTES.toLocaleString("en-US")} bytes.]`);
  }
  return parts.join("\n\n");
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
  const marked = markDocxNotes(linearizeDocxMath(repacked));
  const { value } = await mammoth.extractRawText({
    arrayBuffer: toArrayBuffer(writeDocxTableRows(writeDocxBreaksAndCheckboxes(writeDocxListNumbers(marked.archive)))),
  });
  return marked.label(value);
}

const HTML_PRESCAN_BYTES = 1024;
// Comment or whole tag, quotes included; an unterminated tag ends the scan, as in browsers.
const HTML_PRESCAN_TAG_RE =
  /<!--[\s\S]*?(?:-->|$)|<([a-z][^\s/>]*)((?:[\s/](?:[^>"']|"[^"]*"|'[^']*')*)?)>|<[!/?][^>]*>|<[a-z!/?][\s\S]*/gi;
const HTML_META_ATTR_RE =
  /([^\s"'/=>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]*)))?/g;
const HTML_META_CONTENT_CHARSET_RE = /charset\s*=\s*["']?\s*([^\s"';]+)/i;

function declaredHtmlEncoding(bytes: Uint8Array): string | null {
  const head = new TextDecoder("windows-1252").decode(
    bytes.subarray(0, HTML_PRESCAN_BYTES),
  );
  for (const [, tag, attributes] of head.matchAll(HTML_PRESCAN_TAG_RE)) {
    if (tag?.toLowerCase() !== "meta") {
      continue;
    }
    const attrs = new Map<string, string>();
    for (const [, name, ...values] of attributes.matchAll(HTML_META_ATTR_RE)) {
      const key = name.toLowerCase();
      if (!attrs.has(key)) {
        attrs.set(key, values.join(""));
      }
    }
    let label = attrs.get("charset");
    if (
      label === undefined &&
      attrs.get("http-equiv")?.toLowerCase() === "content-type"
    ) {
      label = attrs.get("content")?.match(HTML_META_CONTENT_CHARSET_RE)?.[1];
    }
    if (!label) {
      continue;
    }
    let encoding: string;
    try {
      encoding = new TextDecoder(label).encoding;
    } catch {
      continue;
    }
    // WHATWG prescan: a <meta> claiming UTF-16 means UTF-8.
    if (encoding.startsWith("utf-16")) {
      return "utf-8";
    }
    return encoding === "x-user-defined" ? "windows-1252" : encoding;
  }
  return null;
}

export function decodeHtmlAttachmentBytes(
  bytes: Uint8Array,
  truncated = false,
): string {
  const bom =
    bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf
      ? "utf-8"
      : bytes[0] === 0xff && bytes[1] === 0xfe
        ? "utf-16le"
        : bytes[0] === 0xfe && bytes[1] === 0xff
          ? "utf-16be"
          : null;
  if (bom) {
    return new TextDecoder(bom).decode(bytes);
  }
  const declared = declaredHtmlEncoding(bytes);
  // Stale meta after a UTF-8 re-save; short CJK can be valid in both, and ISO-2022-JP is 7-bit.
  if (declared && declared !== "utf-8") {
    const utf8 = strictDecode("utf-8", bytes, truncated);
    if (
      utf8 !== null &&
      utf8.length !== bytes.length &&
      (!MULTIBYTE_HTML_ENCODINGS.has(declared) ||
        strictDecode(declared, bytes, truncated) === null)
    ) {
      return utf8;
    }
  }
  return new TextDecoder(declared ?? "utf-8").decode(bytes);
}

const MULTIBYTE_HTML_ENCODINGS = new Set([
  "big5",
  "euc-jp",
  "euc-kr",
  "gb18030",
  "gbk",
  "iso-2022-jp",
  "shift_jis",
]);

function strictDecode(
  label: string,
  bytes: Uint8Array,
  truncated: boolean,
): string | null {
  try {
    return new TextDecoder(label, { fatal: true }).decode(bytes, {
      stream: truncated,
    });
  } catch {
    return null;
  }
}

export function extractHtmlAttachmentText(html: string): string {
  const doc = new DOMParser().parseFromString(html, "text/html");
  for (const el of doc.querySelectorAll("script, style, noscript, template")) {
    el.remove();
  }
  const preformatted: string[] = [];
  return normalizeExtractedText(collectHtmlBlockText(doc.body, preformatted))
    .split("\u0000")
    .map((text, index) => text + (preformatted[index] ?? ""))
    .join("");
}

/** preserves breaks because `textContent` omits block and `<br>` boundaries. */
function collectHtmlBlockText(
  node: Node | null,
  preformatted?: string[],
  rowSpans?: number[],
): string {
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

  if (tag === "pre" && preformatted) {
    const code = Array.from(element.childNodes)
      .map((child) => collectHtmlBlockText(child))
      .join("")
      .replace(/^(?:[^\S\n]*\n)+/, "")
      .trimEnd();
    if (!code.trim()) {
      return "\n";
    }
    preformatted.push(code);
    return "\n\u0000\n";
  }

  if (tag === "tr" && preformatted) {
    const cells = Array.from(element.childNodes).filter(
      (child): child is Element =>
        child.nodeType === ELEMENT_NODE &&
        ["td", "th"].includes((child as Element).tagName.toLowerCase()),
    );
    const covered = rowSpans ?? [];
    const slots: (Element | null)[] = [];
    const skipCovered = () => {
      while (covered[slots.length] > 0) {
        covered[slots.length]--;
        slots.push(null);
      }
    };
    const span = (value: string | null, max: number) =>
      Math.min(Math.max(Number(value) || 1, 1), max);
    for (const cell of cells) {
      skipCovered();
      const rowspan = cell.getAttribute("rowspan");
      const rows =
        rowspan !== null && /^0+$/.test(rowspan)
          ? Number.POSITIVE_INFINITY
          : span(rowspan, 65534);
      for (let i = 0; i < span(cell.getAttribute("colspan"), 1000); i++) {
        covered[slots.length] = rows - 1;
        slots.push(i ? null : cell);
      }
    }
    for (let i = slots.length; i < covered.length; i++) {
      if (covered[i] > 0) covered[i]--;
    }
    // preformatted code keeps its line breaks on the fallback path.
    if (slots.length > 1 && !cells.some(containsPre)) {
      const row = slots
        .map((cell) => (cell ? collectHtmlBlockText(cell).replace(/\s+/g, " ").trim() : ""))
        .join("\t");
      if (!row.trim()) {
        return "\n";
      }
      preformatted.push(row);
      return "\n\u0000\n";
    }
  }

  const groupSpans = HTML_ROW_GROUP_TAGS.has(tag) ? [] : rowSpans;
  const isItem = (child: Node) =>
    tag === "ol" && child.nodeType === ELEMENT_NODE && (child as Element).tagName.toLowerCase() === "li";
  const reversed = tag === "ol" && element.getAttribute("reversed") !== null;
  const start = tag === "ol" ? Number.parseInt(element.getAttribute("start") ?? "", 10) : Number.NaN;
  let number = Number.isNaN(start) ? (reversed ? Array.from(element.childNodes).filter(isItem).length : 1) : start;
  const format = tag === "ol" ? lookUp(HTML_LIST_FORMATS, element.getAttribute("type") ?? "") : undefined;
  const text = Array.from(element.childNodes)
    .map((child) => {
      const inner = collectHtmlBlockText(child, preformatted, groupSpans);
      if (!isItem(child)) return inner;
      const value = Number.parseInt((child as Element).getAttribute("value") ?? "", 10);
      if (!Number.isNaN(value)) number = value;
      const label = listNumber(number, format);
      number += reversed ? -1 : 1;
      return `\n${label}. ${inner.trimStart()}`;
    })
    .join("");
  return HTML_BLOCK_TAGS.has(tag) ? `\n${text}\n` : text;
}

const HTML_ROW_GROUP_TAGS = new Set(["table", "thead", "tbody", "tfoot"]);
const HTML_LIST_FORMATS: Record<string, string> = {
  a: "lowerAlpha",
  A: "upperAlpha",
  i: "lowerRoman",
  I: "upperRoman",
};

function containsPre(node: Node): boolean {
  return Array.from(node.childNodes).some(
    (child) =>
      child.nodeType === ELEMENT_NODE &&
      ((child as Element).tagName.toLowerCase() === "pre" || containsPre(child)),
  );
}

/** collapses extractor spacing while keeping source line breaks. */
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
    return { label: null, ...(await readBoundedHtml(file)) };
  }
  if (isOpenDocumentAttachment(name, contentType)) {
    const { label, text } = await readOpenDocumentAttachmentContent(
      file,
      name,
      contentType ?? "",
    );
    return { label, text, truncated: false };
  }
  if (isRtfAttachment(name, contentType)) {
    const { label, text } = await readRtfAttachmentContent(file, name);
    return { label, text, truncated: false };
  }
  if (isToolOnlyAttachmentName(name)) {
    return {
      label: null,
      text: `${name} has no preview: only the python tool can read it.`,
      truncated: false,
    };
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
  // Strict decoding belongs to the files the text adapter owns, where refusing is better than
  // sending mojibake to the model. A preview of someone else's file may not be stricter than the
  // adapter that accepted it: .html goes to the HTML adapter, which sends a windows-1252 page
  // happily, and reading the preview through the strict path meant opening one threw where it used
  // to render.
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

async function readBoundedHtml(
  file: File,
): Promise<{ text: string; truncated: boolean }> {
  const truncated = file.size > MAX_PREVIEW_TEXT_BYTES;
  const slice = truncated ? file.slice(0, MAX_PREVIEW_TEXT_BYTES) : file;
  const bytes = new Uint8Array(await slice.arrayBuffer());
  return { text: decodeHtmlAttachmentBytes(bytes, truncated), truncated };
}

// A sent attachment keeps only the text its adapter produced, so the preview unwraps the
// adapter's header rather than showing it. The stored payload has no size limit, so the
// wrapper is matched on a prefix and only the capped body is copied out.
export function parseAttachmentText(raw: string): AttachmentText {
  const { label, start, end } = attachmentBodyRange(raw);
  return { label, ...sliceAttachmentBody(raw, start, end) };
}

export function attachmentBodyText(raw: string): string {
  const { start, end } = attachmentBodyRange(raw);
  return raw.slice(start, Math.max(start, end));
}

function attachmentBodyRange(raw: string): {
  label: AttachmentTextLabel | null;
  start: number;
  end: number;
} {
  const head = raw.slice(0, MAX_ATTACHMENT_WRAPPER_LENGTH);

  const labelled = head.match(LABELLED_ATTACHMENT_TEXT_RE);
  if (labelled) {
    return {
      label: labelled[1] as AttachmentTextLabel,
      start: labelled[0].length,
      end: raw.length,
    };
  }

  const tagOpen = head.match(ATTACHMENT_TAG_OPEN_RE);
  if (tagOpen && raw.endsWith(ATTACHMENT_TAG_CLOSE)) {
    return {
      label: null,
      start: tagOpen[0].length,
      end: raw.length - ATTACHMENT_TAG_CLOSE.length,
    };
  }

  return { label: null, start: 0, end: raw.length };
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

/** The shiki language a plain-text attachment previews as, or null for prose. Only the
 *  filename decides; text pulled out of a PDF or DOCX is prose whatever the document was
 *  called, so callers pass the extracted label instead. */
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
