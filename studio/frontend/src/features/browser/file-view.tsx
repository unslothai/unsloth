// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CodeSourceView } from "@/components/code-source-view";
import { DocumentView, MAX_DOCUMENT_PREVIEW_BYTES, documentKind, isMarkdown } from "@/components/file-viewer";
import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { Spinner } from "@/components/ui/spinner";
import { ArtifactHtmlFrame, attachmentTextLanguage, truncateAttachmentPreviewText } from "@/features/chat";
import { useT } from "@/i18n";
import { MAX_HIGHLIGHT_CHARS } from "@/lib/markdown-plugins";
import { useEffect, useMemo, useState } from "react";

const HTML_NAME = /\.(html?|xhtml)$/i;
const HTML_TYPE = /^(text\/html|application\/xhtml\+xml)\b/i;
const TEXT_TYPE = /^(text\/|application\/(json|xml|javascript|x-yaml|yaml|toml|x-sh|sql)\b)/i;
const TEXT_NAME =
  /\.(txt|log|md|markdown|mdx|json|jsonl|ya?ml|toml|ini|cfg|conf|csv|tsv|xml|svg|py|ipynb|js|mjs|cjs|ts|tsx|jsx|css|scss|sh|bash|zsh|rs|go|java|kt|c|cc|cpp|h|hpp|cs|rb|php|swift|sql|r|lua|pl|tex)$/i;

type Media = "image" | "video" | "audio";

function mediaKind(name: string, contentType: string): Media | null {
  if (/^image\//i.test(contentType) || /\.(png|jpe?g|gif|webp|avif|bmp|ico)$/i.test(name)) return "image";
  if (/^video\//i.test(contentType) || /\.(mp4|webm|mov|m4v|ogv)$/i.test(name)) return "video";
  if (/^audio\//i.test(contentType) || /\.(mp3|wav|ogg|oga|flac|m4a|aac|opus)$/i.test(name)) return "audio";
  return null;
}

function useObjectUrl(blob: Blob, enabled: boolean): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled) return;
    const next = URL.createObjectURL(blob);
    // Created and revoked in the effect, so StrictMode never shows a revoked URL.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [blob, enabled]);
  return url;
}

function Unavailable({ message }: { message: string }) {
  return <p className="m-auto max-w-sm px-6 text-center text-sm text-muted-foreground">{message}</p>;
}

function TextFile({
  blob,
  name,
  contentType,
  plainText,
}: {
  blob: Blob;
  name: string;
  contentType: string;
  plainText: boolean;
}) {
  const [text, setText] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    void blob.text().then((value) => active && setText(value));
    return () => {
      active = false;
    };
  }, [blob]);
  const language = useMemo(
    () => (text && text.length <= MAX_HIGHLIGHT_CHARS ? attachmentTextLanguage(name, null) : null),
    [text, name],
  );
  if (text === null) return <Spinner className="m-auto size-6" />;
  const preview = truncateAttachmentPreviewText(text);
  // Same frame as the attachment preview: network stays off until the user allows it.
  if (!plainText && (HTML_NAME.test(name) || HTML_TYPE.test(contentType))) {
    return (
      <div className="size-full overflow-auto">
        <ArtifactHtmlFrame code={preview.text} title={name} fill={true} />
      </div>
    );
  }
  if (!plainText && isMarkdown(name, contentType)) {
    return (
      <div className="size-full overflow-auto px-6">
        <MarkdownPreview
          markdown={preview.text}
          defer={true}
          className="mx-auto max-h-none max-w-3xl select-text overflow-visible border-0 bg-transparent px-2 py-4 text-ui-15p5"
        />
      </div>
    );
  }
  if (language) return <CodeSourceView code={preview.text} language={language} className="px-5 py-4" />;
  return (
    <pre className="size-full overflow-auto whitespace-pre-wrap break-words px-6 py-4 font-mono text-sm leading-relaxed select-text">
      {preview.text}
    </pre>
  );
}

/** A document, image, media file or text, using the attachment viewers. */
export function FileView({
  blob,
  name,
  contentType,
  plainText = false,
}: {
  blob: Blob;
  name: string;
  contentType: string;
  plainText?: boolean;
}) {
  const t = useT();
  const media = plainText ? null : mediaKind(name, contentType);
  const docKind = plainText ? null : documentKind(name, contentType);
  const src = useObjectUrl(blob, media !== null);
  const [failed, setFailed] = useState(false);

  if (docKind) {
    if (blob.size > MAX_DOCUMENT_PREVIEW_BYTES) return <Unavailable message={t("library.preview.cannotPreview")} />;
    return (
      <div className="flex size-full min-h-0 flex-col bg-muted/20">
        <DocumentView file={blob} kind={docKind} name={name} contentType={contentType} />
      </div>
    );
  }
  if (media) {
    if (!src) return <Spinner className="m-auto size-6" />;
    if (failed) return <Unavailable message={t("library.preview.cannotPreview")} />;
    if (media === "image") {
      return (
        <div className="flex size-full items-center justify-center overflow-auto bg-muted/20 p-4">
          <img src={src} alt={name} onError={() => setFailed(true)} className="max-h-full max-w-full object-contain" />
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
  if (plainText || TEXT_TYPE.test(contentType) || TEXT_NAME.test(name) || HTML_NAME.test(name) || !contentType) {
    return (
      <TextFile blob={blob} name={name} contentType={contentType} plainText={plainText} />
    );
  }
  return <Unavailable message={t("browser.cannotShowFile")} />;
}
