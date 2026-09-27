// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { attachmentOriginal } from "./attachment-originals";
import { isToolOnlyAttachmentName } from "./open-document-accept";

const NAME_BYTES = 80;
const encoder = new TextEncoder();
const utf8Length = (text: string) => encoder.encode(text).length;

/** sandbox_attachment_path in core/inference/tools.py. Its result is a fixed point there, so the
 *  basename sent back derives this same path on the server. */
export function sandboxAttachmentPath(sha256: string, name: string): string {
  let base =
    name
      // eslint-disable-next-line no-control-regex
      .replace(/[\x00-\x1f\x7f/\\:*?"<>|]/g, "_")
      .replace(/^[ .]+|[ .]+$/g, "") || "attachment";
  if (utf8Length(base) > NAME_BYTES) {
    const dot = base.lastIndexOf(".");
    const stem = dot > 0 ? base.slice(0, dot) : base;
    let ext = dot > 0 ? base.slice(dot) : "";
    if (utf8Length(ext) > 16) ext = "";
    let room = NAME_BYTES - utf8Length(ext);
    let kept = "";
    for (const char of stem) {
      room -= utf8Length(char);
      if (room < 0) break;
      kept += char;
    }
    base = (kept.replace(/[ .]+$/, "") || "attachment") + ext;
  }
  return `.unsloth_attachments/${sha256.slice(0, 12)}/${base}`;
}

// ChatCompletionRequest.sandbox_attachments cap: past it the whole request is refused.
const MAX_SANDBOX_ATTACHMENTS = 64;

type Attachment = { name?: string; content?: readonly unknown[] };

function sandboxCopy(attachment: unknown): { sha256: string; path: string } | null {
  const original = attachmentOriginal(attachment);
  if (!original || !/^[0-9a-f]{64}$/.test(original.sha256)) return null;
  const name = (attachment as Attachment).name ?? "";
  return { sha256: original.sha256, path: sandboxAttachmentPath(original.sha256, name) };
}

/** Notes each kept file's sandbox path for the model, and lists the copies the backend makes. */
export function withSandboxAttachmentPaths<
  M extends { attachments?: readonly unknown[] },
>(messages: readonly M[]) {
  const latest = new Map<string, string>();
  for (const message of messages) {
    for (const attachment of message.attachments ?? []) {
      const copy = sandboxCopy(attachment);
      if (!copy) continue;
      latest.delete(copy.path);
      latest.set(copy.path, copy.sha256);
    }
  }
  const carried = new Map([...latest].slice(-MAX_SANDBOX_ATTACHMENTS));
  const sandboxAttachments = [...carried].map(([path, sha256]) => ({
    sha256,
    name: path.slice(path.lastIndexOf("/") + 1),
  }));
  const annotated = messages.map((message) => {
    let changed = false;
    const attachments = (message.attachments ?? []).map((attachment) => {
      const copy = sandboxCopy(attachment);
      if (!copy || !carried.has(copy.path)) return attachment;
      changed = true;
      const { name = "", content = [] } = attachment as Attachment;
      const reader = sandboxReader(copy.path);
      const path = JSON.stringify(copy.path);
      const note = isToolOnlyAttachmentName(name)
        ? `[${name} is saved at ${copy.path} in the python tool's working directory${reader ? `; open it with ${reader}, where path = ${path}` : ""}]`
        : `[${name}: its text is below, so answer from it. For calculations, the python tool has the file at path = ${path}${reader ? `; ${reader}` : ""}]`;
      return {
        ...(attachment as object),
        content: [{ type: "text", text: note }, ...content],
      };
    });
    return changed ? ({ ...message, attachments } as M) : message;
  });
  return { messages: annotated, sandboxAttachments };
}

const READERS: ReadonlyArray<readonly [string, string]> = [
  [".csv", "pandas.read_csv(path)"],
  [".tsv", 'pandas.read_csv(path, sep="\\t")'],
  [".jsonl,.ndjson", "pandas.read_json(path, lines=True)"],
  [".json", "json.load(open(path))"],
  [".docx", "docx.Document(path)"],
  [
    ".pdf,.pptx,.pptm,.ppsx,.potx,.potm,.ppsm,.docm,.dotx,.dotm,.epub,.mobi,.fb2,.cbz,.xps,.oxps",
    "fitz.open(path)",
  ],
  // No reader for .ods/.odt or Excel: their text is inline, and the raw zip sends small models into XML.
  [".odp,.odg", 'zipfile.ZipFile(path).read("content.xml")'],
  [".zip,.jar,.whl,.apk,.kmz,.3mf,.vsdx", "zipfile.ZipFile(path)"],
  [".tar,.tar.gz,.tgz,.tar.bz2,.tbz2,.tbz,.tar.xz,.txz", "tarfile.open(path)"],
  [".gz", "gzip.open(path)"],
  [".bz2", "bz2.open(path)"],
  [".xz,.lzma", "lzma.open(path)"],
  [".parquet", "pandas.read_parquet(path)"],
  [".feather", "pandas.read_feather(path)"],
  [".arrow", "pyarrow.ipc.open_file(path).read_all()"],
  [".orc", "pyarrow.orc.read_table(path)"],
  // `path` stays a variable: the sandbox scanner treats a literal passed to .connect() as a network host.
  [".sqlite,.sqlite3,.db,.gpkg,.mbtiles", "sqlite3.connect(path)"],
  [".duckdb", "duckdb.connect(path, read_only=True)"],
  [".npy,.npz", "numpy.load(path)"],
  [".dta", "pandas.read_stata(path)"],
  [".sas7bdat", "pandas.read_sas(path)"],
  [".xpt", 'pandas.read_sas(path, format="xport")'],
  [".mat", "scipy.io.loadmat(path)"],
  [".safetensors", 'safetensors.safe_open(path, framework="numpy")'],
  [".stl,.ply,.glb", 'open(path, "rb").read()'],
  [".ttf,.otf,.woff", "fontTools.ttLib.TTFont(path)"],
  [".ttc", "fontTools.ttLib.TTCollection(path)"],
  [
    ".psd,.ico,.icns,.cur,.tga,.dds,.pcx,.ppm,.pgm,.pbm,.pnm,.qoi,.jp2,.j2k,.xbm,.xpm,.sgi,.fits",
    "PIL.Image.open(path)",
  ],
];

export function sandboxReader(name: string): string | null {
  const lower = name.toLowerCase();
  const reader = READERS.find(([extensions]) =>
    extensions.split(",").some((extension) => lower.endsWith(extension)),
  );
  return reader ? reader[1] : null;
}
