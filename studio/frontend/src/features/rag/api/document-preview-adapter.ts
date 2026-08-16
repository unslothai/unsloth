import { fetchDocumentPreview } from "@/integrations/platform-backend";
import type { PreviewTarget } from "../types/rag";
import { getDocumentFileUrl, getPreviewTarget } from "./rag-api";

export interface DocumentPreviewRequest {
  documentId: string;
  chunkId?: string;
  filename?: string | null;
  page?: number | null;
  source?: "platform" | "local" | null;
  signal?: AbortSignal;
}

export interface ResolvedDocumentPreview {
  target: PreviewTarget;
  fileUrl: string | null;
  dispose: () => void;
}

const noDispose = () => undefined;

function isPdf(contentType: string, filename: string): boolean {
  return contentType.toLowerCase().includes("application/pdf") ||
    filename.toLowerCase().endsWith(".pdf");
}

function isText(contentType: string, filename: string): boolean {
  const normalized = contentType.toLowerCase();
  return (
    normalized.startsWith("text/") ||
    normalized.includes("json") ||
    normalized.includes("xml") ||
    /\.(?:txt|md|markdown|html?|csv|json|xml)$/i.test(filename)
  );
}

async function resolvePlatformPreview(
  request: DocumentPreviewRequest,
): Promise<ResolvedDocumentPreview> {
  const asset = await fetchDocumentPreview(request.documentId, request.signal);
  if (request.signal?.aborted) throw request.signal.reason;
  const filename = request.filename?.trim() || "Document";
  const targetBase = {
    documentId: request.documentId,
    filename,
    targetPage: request.page ?? null,
    pdfRegions: [],
  };

  if (isPdf(asset.contentType, filename)) {
    const fileUrl = URL.createObjectURL(asset.blob);
    return {
      target: { ...targetBase, mediaKind: "pdf" },
      fileUrl,
      dispose: () => URL.revokeObjectURL(fileUrl),
    };
  }

  const text = isText(asset.contentType, filename)
    ? await asset.blob.text()
    : null;
  if (request.signal?.aborted) throw request.signal.reason;
  return {
    target: { ...targetBase, mediaKind: "text", text },
    fileUrl: null,
    dispose: noDispose,
  };
}

export async function resolveDocumentPreview(
  request: DocumentPreviewRequest,
): Promise<ResolvedDocumentPreview> {
  if (request.source === "platform") {
    return resolvePlatformPreview(request);
  }

  const target = await getPreviewTarget(
    request.documentId,
    request.chunkId,
    request.signal,
  );
  if (request.signal?.aborted) throw request.signal.reason;
  const fileUrl =
    target.mediaKind === "pdf"
      ? await getDocumentFileUrl(request.documentId, request.signal)
      : null;
  return { target, fileUrl, dispose: noDispose };
}
