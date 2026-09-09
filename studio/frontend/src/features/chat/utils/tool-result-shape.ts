// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a tool call's result becomes ONCE IT IS A PART, and why this lives here instead of inline in the live
// stream: the wire carries one string (`event.result`, plus an image on its own fields), and the card renders an
// OBJECT - text plus images plus the sandbox session they came from. The live path shaped that object at the
// frame; a reopened tab replayed the frames and copied `event.result` VERBATIM, so a `python` turn that plotted a
// chart rendered the literal string `__IMAGES__:["plot_0.png"]`. Both readers now call this one function, which is
// the only way "replay is faithful to live" stops being a per-shape coincidence: whatever shape the wire used to
// carry (sandbox files, an MCP image envelope, web-search images, an inline base64 image), it lands shaped.

import {
  SANDBOX_FILE_TOOLS,
  extractCreatedFiles,
  type SandboxFile,
} from "../../../components/assistant-ui/sandbox-files";
import {
  SEARCH_IMAGE_TOOL,
  extractSearchImages,
  type SearchImageEntry,
} from "../search-images/search-images";

/** A tool result the app wrapped into text + images. The marker an MCP server's image envelope is split on. */
export const MCP_IMAGES_MARKER = "\n__MCP_IMAGES__:";
/** The sandbox's own image envelope, older than the MCP one and still what `python`/`terminal` emit. */
export const SANDBOX_IMAGES_MARKER = "\n__IMAGES__:";

export interface McpImageToolResult {
  text: string;
  images: { data: string; mimeType: string }[];
}

/** An image-bearing result this app wrapped, and not merely a result shaped like one: unwrapping someone else's
 *  MCP result would drop every other field it returned. */
export function isMcpImageToolResult(val: unknown): val is McpImageToolResult {
  if (typeof val !== "object" || val === null) return false;
  const v = val as { text?: unknown; images?: unknown; sessionId?: unknown };
  return (
    typeof v.text === "string" &&
    v.sessionId === undefined &&
    Array.isArray(v.images) &&
    v.images.length > 0 &&
    v.images.every(
      (img: unknown) =>
        typeof img === "object" &&
        img !== null &&
        typeof (img as { data?: unknown }).data === "string" &&
        typeof (img as { mimeType?: unknown }).mimeType === "string",
    )
  );
}

/** What an `image_generation` result becomes: the backend keeps base64 on its own fields to keep logs small. */
export interface ImageGenerationToolResult {
  image_b64: string;
  image_mime: string;
  size?: string;
  quality?: string;
  background?: string;
  prompt?: string;
}

export type ShapedToolResult =
  | string
  | McpImageToolResult
  | SearchImagesToolResultLike
  | SandboxToolResult
  | ImageGenerationToolResult;

type SearchImagesToolResultLike = { text: string; webImages: SearchImageEntry[] };
type SandboxToolResult = {
  text: string;
  images: string[];
  sessionId: string;
  files: SandboxFile[];
};

export type ShapeToolResultInput = {
  /** The tool that ran. Every marker below is content for any OTHER tool, which is why this is required. */
  toolName?: string;
  /** `event.result` exactly as the frame carried it. Anything not a string needs no shaping and is returned. */
  raw: unknown;
  /** The whole frame, for an inline base64 image kept off `result` (image_b64 / image_mime / size / ...). */
  event?: Record<string, unknown>;
  /** The chat's sandbox dir, the only record of WHERE a sandbox call ran. `_default` is the backend's own default. */
  sandboxSessionId?: string;
};

/**
 * Turn a tool frame into the part value the card renders - the live stream's rule, in one place.
 *
 * Order matters and is load-bearing: an inline base64 image wins over any marker; a VALID MCP envelope beats a
 * sandbox suffix (an invalid one falls through so the sandbox suffix still renders); web-search images are their own
 * shape; and a sandbox call stays structured even with neither files nor images, because the session id is the only
 * record of where it ran. Anything else passes through untouched - a marker for a tool that does not emit one is
 * content, and eating it would be its own bug.
 */
export function shapeToolResult(input: ShapeToolResultInput): unknown {
  const raw = input.raw;
  if (typeof raw !== "string") return raw;
  const toolName = input.toolName ?? "";
  const event = input.event ?? {};
  const sessionId = input.sandboxSessionId || "_default";

  // Pulled out first, ahead of the image markers, so the image slice below is unchanged. Only from the tools that
  // emit it: elsewhere that line is content.
  const { text: rawResult, files: createdFiles } = SANDBOX_FILE_TOOLS.has(toolName)
    ? extractCreatedFiles(raw)
    : { text: raw, files: [] as SandboxFile[] };
  // Same rule: only from the tool that emits it.
  const { text: searchText, images: webImages } =
    toolName === SEARCH_IMAGE_TOOL
      ? extractSearchImages(rawResult)
      : { text: rawResult, images: [] as SearchImageEntry[] };

  const imgIdx = rawResult.lastIndexOf(SANDBOX_IMAGES_MARKER);
  const mcpImgIdx = rawResult.lastIndexOf(MCP_IMAGES_MARKER);

  // A valid MCP image envelope wins; an invalid marker falls through so a sandbox __IMAGES__ suffix still renders.
  let mcpImages: McpImageToolResult | null = null;
  if (mcpImgIdx !== -1) {
    try {
      const images = JSON.parse(rawResult.slice(mcpImgIdx + MCP_IMAGES_MARKER.length));
      const candidate = { text: rawResult.slice(0, mcpImgIdx), images };
      if (isMcpImageToolResult(candidate)) mcpImages = candidate;
    } catch {
      // Not a valid envelope; fall through below.
    }
  }

  const imageB64 = typeof event.image_b64 === "string" ? event.image_b64 : undefined;
  if (toolName === "image_generation" && imageB64) {
    return {
      image_b64: imageB64,
      image_mime: (event.image_mime as string | undefined) ?? "image/png",
      size: event.size as string | undefined,
      quality: event.quality as string | undefined,
      background: event.background as string | undefined,
      prompt: event.prompt as string | undefined,
    } satisfies ImageGenerationToolResult;
  }
  if (mcpImages !== null) return mcpImages;
  if (imgIdx !== -1) {
    try {
      const images = JSON.parse(rawResult.slice(imgIdx + SANDBOX_IMAGES_MARKER.length)) as string[];
      return { text: rawResult.slice(0, imgIdx), images, sessionId, files: createdFiles } satisfies SandboxToolResult;
    } catch {
      return rawResult;
    }
  }
  // Structured even with neither files nor images: the session is the only record of WHERE this call ran.
  if (createdFiles.length > 0 || SANDBOX_FILE_TOOLS.has(toolName)) {
    return { text: rawResult, images: [], sessionId, files: createdFiles } satisfies SandboxToolResult;
  }
  if (webImages.length > 0) return { text: searchText, webImages };
  return rawResult;
}

// What a Sources entry is made of, kept next to the shaper for the same reason as everything else in this file:
// live derives these parts at the end of a stream (web_search/web_fetch results, Anthropic citation dicts), and a
// reopened tab must derive the SAME objects from stored frames — the replay cannot import the adapter (react under
// node --test), and both readers must agree on an id shape down to the byte, or one entry lands twice.
/** Return `raw` when it is a safe http(s) URL, else "": rejects CR/LF and javascript:/data:/
 *  vbscript: so provider strings cannot land in an <a href>. */
export function isSafeNavigableSourceUrl(raw: unknown): string {
  if (typeof raw !== "string") return "";
  const value = raw.trim();
  if (!value || /[\r\n]/.test(value)) return "";
  try {
    const parsed = new URL(value);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") {
      return value;
    }
  } catch {
  }
  return "";
}

/** Convert an Anthropic document citation dict into a Sources-panel source. */
export function documentCitationToSource(
  cit: Record<string, unknown>,
  fallbackIdx: number,
): {
  type: "source";
  sourceType: "url";
  id: string;
  url: string;
  title: string;
  metadata?: { description: string };
} | null {
  const source = typeof cit.source === "string" && cit.source ? cit.source : "";
  const docTitle =
    (typeof cit.document_title === "string" && cit.document_title) ||
    (typeof cit.title === "string" && cit.title) ||
    "";
  const docIndex =
    typeof cit.document_index === "number" ? cit.document_index : undefined;
  // search_result_location `source` can be a free-form id or a hostile scheme; fall back to a
  // doc anchor unless it is real http(s).
  const url =
    isSafeNavigableSourceUrl(source) ||
    `#anthropic-doc-${docIndex ?? fallbackIdx}`;
  const title = docTitle || source || `Document ${fallbackIdx + 1}`;
  const cited = typeof cit.cited_text === "string" ? cit.cited_text.trim() : "";
  const description = cited.length > 240 ? `${cited.slice(0, 240)}...` : cited;
  // Anthropic numbers inline [N] per citation, so fold citation type and position into the id.
  const citationType = typeof cit.type === "string" ? String(cit.type) : "";
  const positionParts = [
    cit.search_result_index,
    cit.start_char_index,
    cit.end_char_index,
    cit.start_page_number,
    cit.end_page_number,
    cit.start_block_index,
    cit.end_block_index,
  ]
    .filter((v) => typeof v === "number")
    .map((v) => String(v))
    .join(":");
  const idAnchor = positionParts
    ? `${citationType}:${positionParts}`
    : `${citationType}:${fallbackIdx}`;
  const id = `${url}#${idAnchor}`;
  return {
    type: "source" as const,
    sourceType: "url" as const,
    id,
    url,
    title,
    ...(description ? { metadata: { description } } : {}),
  };
}

/** Parse "Title: ...\nURL: ...\nSnippet: ..." blocks into source content parts. */
export function parseSourcesFromResult(raw: string): {
  type: "source";
  sourceType: "url";
  id: string;
  url: string;
  title: string;
  metadata?: { description: string };
}[] {
  if (!raw) return [];
  const blocks = raw.split(/\n---\n/).filter(Boolean);
  const sources: {
    type: "source";
    sourceType: "url";
    id: string;
    url: string;
    title: string;
    metadata?: { description: string };
  }[] = [];
  for (const block of blocks) {
    const titleMatch = block.match(/Title:\s*(.+)/);
    const urlMatch = block.match(/URL:\s*(.+)/);
    const snippetMatch = block.match(/Snippet:\s*(.+)/);
    if (titleMatch && urlMatch) {
      // Provider output is attacker-controllable: a non-http(s) URL must not reach the Sources panel <a href>.
      const url = isSafeNavigableSourceUrl(urlMatch[1]);
      if (!url) continue;
      const snippet = snippetMatch?.[1]?.trim();
      sources.push({
        type: "source" as const,
        sourceType: "url" as const,
        id: url,
        url,
        title: titleMatch[1].trim(),
        ...(snippet ? { metadata: { description: snippet } } : {}),
      });
    }
  }
  return sources;
}
