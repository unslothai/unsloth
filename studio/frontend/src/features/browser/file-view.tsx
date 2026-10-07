// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CodeSourceView } from "@/components/code-source-view";
import { DocumentView, MAX_DOCUMENT_PREVIEW_BYTES, documentKind } from "@/components/file-viewer";
import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { Spinner } from "@/components/ui/spinner";
import {
  ArtifactHtmlFrame,
  attachmentTextLanguage,
  truncateAttachmentPreviewText,
} from "@/features/chat";
import { useT } from "@/i18n";
import { MAX_HIGHLIGHT_CHARS } from "@/lib/markdown-plugins";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useMemo, useState } from "react";
import { canShowFile, mediaKind, textFileKind } from "./file-kind";
import { stageEditsPrompt } from "./stage-edits";
import { DEFAULT_FILE_VIEW, useBrowserStore } from "./store";

// The preview shows at most 200,000 chars (4 bytes each at most): a 50 MB body is never decoded whole.
const MAX_TEXT_BYTES = 1024 * 1024;

function useObjectUrl(blob: Blob, enabled: boolean, svg: boolean): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled) return;
    if (svg) {
      // A data: URL, so an opened SVG can't run as a page on Studio's origin.
      let live = true;
      const reader = new FileReader();
      reader.onload = () => live && typeof reader.result === "string" && setUrl(reader.result);
      reader.readAsDataURL(new Blob([blob], { type: "image/svg+xml" }));
      return () => {
        live = false;
      };
    }
    const next = URL.createObjectURL(blob);
    // Created and revoked in the effect, so StrictMode never shows a revoked URL.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [blob, enabled, svg]);
  return url;
}

function Unavailable({ message }: { message: string }) {
  return <p className="m-auto max-w-sm px-6 text-center text-sm text-muted-foreground">{message}</p>;
}

const WRAP_CLASS = "[&_pre]:whitespace-pre-wrap! [&_pre]:break-words [&_code]:whitespace-pre-wrap! [&_.min-w-max]:min-w-0!";

function TextFile({
  blob,
  name,
  contentType,
  plainText,
  scale,
  tabId,
  reloadNonce,
}: {
  blob: Blob;
  name: string;
  contentType: string;
  plainText: boolean;
  scale: number;
  tabId: string | undefined;
  reloadNonce: number;
}) {
  const [text, setText] = useState<string | null>(null);
  const view = useBrowserStore((state) => (tabId ? state.fileViews[tabId] : undefined)) ?? DEFAULT_FILE_VIEW;
  const requestEdits = useBrowserStore((state) => state.requestEdits);
  // HTML runs only once previewed, so opening to source does not run it.
  const [previewed, setPreviewed] = useState(view.mode === "preview");
  if (!previewed && view.mode === "preview") setPreviewed(true);
  const kind = textFileKind(name, contentType, plainText);
  useEffect(() => {
    let active = true;
    // HTML runs whole: a cut document breaks its scripts and closing tags.
    const bytes = kind === "html" ? blob.size : MAX_TEXT_BYTES;
    void blob.slice(0, bytes).text().then((value) => active && setText(value));
    return () => {
      active = false;
    };
  }, [blob, kind]);
  // Stable: the frame reports its counts from an effect that depends on these.
  const onConsoleOpenChange = useCallback(
    (consoleOpen: boolean) => tabId && useBrowserStore.getState().setFileView(tabId, { consoleOpen }),
    [tabId],
  );
  const onOutputCountChange = useCallback(
    ({ errors }: { errors: number }) => tabId && useBrowserStore.getState().setFileView(tabId, { errorCount: errors }),
    [tabId],
  );
  const language = useMemo(() => {
    if (!text || text.length > MAX_HIGHLIGHT_CHARS) return null;
    if (kind === "html") return "html";
    if (kind === "markdown") return "markdown";
    return attachmentTextLanguage(name, null);
  }, [text, name, kind]);
  if (text === null) return <Spinner className="m-auto size-6" />;
  const preview = truncateAttachmentPreviewText(text);
  const source = (kind === "html" || kind === "markdown") && view.mode === "source";
  const sourceView = language ? (
    <div className={cn("size-full overflow-auto", view.wrap && WRAP_CLASS)} style={{ zoom: scale }}>
      <CodeSourceView code={preview.text} language={language} className="px-5 py-4" />
    </div>
  ) : (
    <pre
      style={{ zoom: scale }}
      className={cn(
        "size-full overflow-auto px-6 py-4 font-mono text-sm leading-relaxed select-text",
        kind === "text" || view.wrap ? "whitespace-pre-wrap break-words" : "whitespace-pre",
      )}
    >
      {preview.text}
    </pre>
  );
  // Same frame as the attachment preview: network stays off until the user allows it.
  if (kind === "html") {
    return (
      <>
        {/* Kept mounted behind the source, so the console keeps its output. */}
        {previewed ? (
          <div className={cn("size-full overflow-auto", source && "hidden")} style={{ zoom: scale }}>
            <ArtifactHtmlFrame
              code={text}
              title={name}
              fill={true}
              reloadNonce={reloadNonce}
              consoleOpen={view.consoleOpen}
              onConsoleOpenChange={tabId ? onConsoleOpenChange : undefined}
              onOutputCountChange={tabId ? onOutputCountChange : undefined}
              onFixWithModel={requestEdits ?? stageEditsPrompt}
            />
          </div>
        ) : null}
        {source ? sourceView : null}
      </>
    );
  }
  if (kind === "markdown" && !source) {
    return (
      <div className="size-full overflow-auto px-6" style={{ zoom: scale }}>
        <MarkdownPreview
          markdown={preview.text}
          defer={true}
          className="mx-auto max-h-none max-w-3xl select-text overflow-visible border-0 bg-transparent px-2 py-4 text-ui-15p5"
        />
      </div>
    );
  }
  return sourceView;
}

export function FileView({
  blob,
  name,
  contentType,
  plainText = false,
  scale = 1,
  tabId,
  reloadNonce = 0,
}: {
  blob: Blob;
  name: string;
  contentType: string;
  plainText?: boolean;
  scale?: number;
  tabId?: string;
  reloadNonce?: number;
}) {
  const t = useT();
  const media = plainText ? null : mediaKind(name, contentType);
  const docKind = plainText ? null : documentKind(name, contentType);
  const svg = media === "image" && (/^image\/svg\+xml\b/i.test(contentType) || /\.svg$/i.test(name));
  const src = useObjectUrl(blob, media !== null, svg);
  const [failed, setFailed] = useState(false);

  if (docKind) {
    if (blob.size > MAX_DOCUMENT_PREVIEW_BYTES) return <Unavailable message={t("library.preview.cannotPreview")} />;
    return (
      <div className="flex size-full min-h-0 flex-col">
        <DocumentView file={blob} kind={docKind} name={name} contentType={contentType} scale={scale} />
      </div>
    );
  }
  if (media) {
    if (!src) return <Spinner className="m-auto size-6" />;
    if (failed) return <Unavailable message={t("library.preview.cannotPreview")} />;
    if (media === "image") {
      return (
        <div className="flex size-full items-center justify-center overflow-auto bg-muted/20 p-4">
          <img
            src={src}
            alt={name}
            onError={() => setFailed(true)}
            style={{ zoom: scale }}
            className="max-h-full max-w-full object-contain"
          />
        </div>
      );
    }
    if (media === "video") {
      return (
        <video src={src} controls onError={() => setFailed(true)} className="size-full bg-black object-contain" />
      );
    }
    return (
      <div className="m-auto w-full max-w-lg px-6">
        {/* biome-ignore lint/a11y/useMediaCaption: user audio has no captions */}
        <audio src={src} controls onError={() => setFailed(true)} className="w-full" />
      </div>
    );
  }
  if (plainText || canShowFile(name, contentType)) {
    return (
      <TextFile
        blob={blob}
        name={name}
        contentType={contentType}
        plainText={plainText}
        scale={scale}
        tabId={tabId}
        reloadNonce={reloadNonce}
      />
    );
  }
  return <Unavailable message={t("browser.cannotShowFile")} />;
}
