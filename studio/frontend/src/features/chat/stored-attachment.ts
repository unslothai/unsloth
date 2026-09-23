// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth/api";

import { isToolOnlyAttachmentName } from "./open-document-accept";

/** What the backend could read of the file without the model opening it. */
export type AttachmentPreview =
  | { kind: "image"; description: string; image: string }
  | { kind: "text"; label: string; text: string }
  | { kind: "outline"; text: string };

export type StoredAttachmentFile = {
  id: string;
  sandboxPath: string;
  /** The preview held the file's own text, which the message now carries. */
  inlineText?: boolean;
};

export type UploadedAttachmentFile = StoredAttachmentFile & {
  preview?: AttachmentPreview;
};

/** What the message keeps: the preview is its content already, and would be stored twice. */
export function storedAttachmentFile({
  id,
  sandboxPath,
  preview,
}: UploadedAttachmentFile): StoredAttachmentFile {
  return preview?.kind === "text"
    ? { id, sandboxPath, inlineText: true }
    : { id, sandboxPath };
}

type AttachmentContent =
  | { type: "text"; text: string }
  | { type: "image"; image: string };

/** What the model sees of such a file: the preview, or the note that the tool has to open it. */
export function toolOnlyAttachmentContent(
  name: string,
  preview: AttachmentPreview | undefined,
  sendsImages: boolean,
): AttachmentContent[] {
  if (!preview) {
    return [
      {
        type: "text",
        text: `[${name}: only the python tool can read this file]`,
      },
    ];
  }
  if (preview.kind === "image") {
    const described: AttachmentContent = {
      type: "text",
      text: `[${name}: ${preview.description}]`,
    };
    return sendsImages
      ? [{ type: "image", image: preview.image }, described]
      : [described];
  }
  const header =
    preview.kind === "text"
      ? `[${preview.label}: ${name}]`
      : `[Outline of ${name}]`;
  return [{ type: "text", text: `${header}\n${preview.text}` }];
}

type Attachment = {
  name: string;
  content?: readonly unknown[];
  storedFile?: StoredAttachmentFile;
};

/** A response's preview, or nothing when it is not one of the three shapes. */
function parsePreview(value: unknown): AttachmentPreview | undefined {
  const preview = value as AttachmentPreview | undefined;
  const has = (...keys: string[]) =>
    keys.every(
      (key) =>
        typeof (preview as unknown as Record<string, unknown>)[key] ===
        "string",
    );
  if (preview?.kind === "image" && has("description", "image")) return preview;
  if (preview?.kind === "text" && has("label", "text")) return preview;
  if (preview?.kind === "outline" && has("text")) return preview;
  return undefined;
}

/** Keeps the original bytes for the python tool. Null when that fails, leaving the inline text. */
export async function uploadAttachmentFile(
  file: File,
): Promise<UploadedAttachmentFile | null> {
  const form = new FormData();
  form.append("file", file);
  try {
    const response = await authFetch("/api/chat/attachment-files", {
      method: "POST",
      body: form,
    });
    const { id, sandboxPath, preview } = (await response.json()) as Record<
      string,
      unknown
    >;
    return typeof id === "string" && typeof sandboxPath === "string"
      ? { id, sandboxPath, preview: parsePreview(preview) }
      : null;
  } catch {
    return null;
  }
}

function isStored(
  attachment: unknown,
): attachment is Attachment & { storedFile: StoredAttachmentFile } {
  const stored = (attachment as Attachment | null)?.storedFile;
  return (
    typeof stored?.id === "string" && typeof stored.sandboxPath === "string"
  );
}

// What ChatCompletionRequest.sandbox_attachments takes. Past it the request is refused outright,
// so a long thread carries its most recent files rather than failing every turn.
const MAX_SANDBOX_ATTACHMENTS = 64;

function carriedSandboxPaths(
  messages: readonly { attachments?: readonly unknown[] }[],
): Set<string> {
  const order = new Set<string>();
  for (const message of messages) {
    for (const attachment of message.attachments ?? []) {
      if (!isStored(attachment)) continue;
      const { sandboxPath } = attachment.storedFile;
      // Deleted first, so a file attached again late in the thread counts as recent, not as old.
      order.delete(sandboxPath);
      order.add(sandboxPath);
    }
  }
  return new Set([...order].slice(-MAX_SANDBOX_ATTACHMENTS));
}

/** Names every stored attachment's sandbox copy, and asks the backend to put the copies there. */
export function withSandboxAttachmentPaths<
  M extends { attachments?: readonly unknown[] },
>(messages: readonly M[]) {
  const sandboxAttachments: Array<{ id: string; name: string }> = [];
  const carried = carriedSandboxPaths(messages);
  const listed = new Set<string>();
  const annotated = messages.map((message) => {
    if (!message.attachments?.some(isStored)) return message;
    const attachments = message.attachments.map((attachment) => {
      if (!isStored(attachment)) return attachment;
      const { id, sandboxPath, inlineText } = attachment.storedFile;
      // Dropped from the request, so it is not told to open a file that was never copied.
      if (!carried.has(sandboxPath)) return attachment;
      if (!listed.has(sandboxPath)) {
        listed.add(sandboxPath);
        // The path's own last segment, so the backend derives this exact path again.
        sandboxAttachments.push({ id, name: sandboxPath.split("/").pop()! });
      }
      const reader = sandboxReader(sandboxPath);
      // A file whose text is inline says so; an outline or an image is not its text.
      const note =
        isToolOnlyAttachmentName(attachment.name) && !inlineText
          ? `[${attachment.name} is saved at ${sandboxPath} in the python tool's working directory${reader ? `; open it with ${reader}, where path = ${JSON.stringify(sandboxPath)}` : ""}]`
          : `[${attachment.name}: its text is below, so answer from it. For calculations, the python tool has the file at path = ${JSON.stringify(sandboxPath)}${reader ? `; ${reader}` : ""}]`;
      return {
        ...attachment,
        content: [{ type: "text", text: note }, ...(attachment.content ?? [])],
      };
    });
    return { ...message, attachments } as M;
  });
  return { messages: annotated, sandboxAttachments };
}

// A library every install ships, since models guess wrong (python-docx refuses .docm; openpyxl,
// python-pptx and odfpy are absent). `path` is a variable because the sandbox's scanner reads a
// literal passed to .connect() as a network host. Compound extensions come before their last part.
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
  // No reader for .ods/.odt, Excel or iWork: their text is inline, and the raw zip sends small models into XML.
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
