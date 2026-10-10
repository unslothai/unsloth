// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The libraries and the page builder load on first use (preview-document.ts); this part is small.

import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import { useEffect, useState } from "react";
import { ArtifactHtmlFrame } from "../html-frame";
import { type UnavailableReason, compileReactPreview } from "./compile-api";
import { needsNode, noteNodeAvailability } from "./node-availability";

export { needsNode, noteNodeAvailability };

export type PreparedReactPreview =
  | { status: "ready"; html: string }
  | { status: "compile-error"; html: string }
  | { status: "unavailable"; reason: UnavailableReason };

/** Compile, load the libraries it imports and build the page. Never runs the code. */
export async function prepareReactPreview({
  source,
  lang,
  title,
  signal,
}: {
  source: string;
  lang: "jsx" | "tsx";
  title: string;
  signal?: AbortSignal;
}): Promise<PreparedReactPreview> {
  const result = await compileReactPreview(source, lang, signal);
  if (result.status === "unavailable") return result;
  try {
    const document = await import("./preview-document");
    if (result.status === "error") {
      return { status: "compile-error", html: document.buildCompileErrorDocument(result.diagnostics, title) };
    }
    const html = await document.buildPreviewDocument({ code: result.code, deps: result.deps, source, lang, title });
    return { status: "ready", html };
  } catch {
    return { status: "unavailable", reason: "failed" };
  }
}

type PreviewState = { key: string; prepared: PreparedReactPreview } | null;

export function ReactPreview({
  source,
  lang,
  title,
  reloadNonce = 0,
  consoleOpen = false,
  onConsoleOpenChange,
  onOutputCountChange,
  onFixWithModel,
}: {
  source: string;
  lang: "jsx" | "tsx";
  title: string;
  reloadNonce?: number;
  consoleOpen?: boolean;
  onConsoleOpenChange?: (open: boolean) => void;
  onOutputCountChange?: (counts: { errors: number; total: number }) => void;
  onFixWithModel?: (prompt: string) => void;
}) {
  const t = useT();
  const key = `${lang}\u0000${source}`;
  const [state, setState] = useState<PreviewState>(null);

  // Run again prepares again too, so a missing library or Node gets retried; a page already
  // showing stays up meanwhile and the frame re-runs it from reloadNonce.
  useEffect(() => {
    const controller = new AbortController();
    void prepareReactPreview({ source, lang, title, signal: controller.signal })
      .then((prepared) => {
        if (controller.signal.aborted) return;
        noteNodeAvailability(prepared);
        setState({ key: `${lang}\u0000${source}`, prepared });
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setState({ key: `${lang}\u0000${source}`, prepared: { status: "unavailable", reason: "failed" } });
        }
      });
    return () => controller.abort();
  }, [source, lang, title, reloadNonce]);

  const prepared = state?.key === key ? state.prepared : null;
  if (!prepared) {
    return (
      <div className="flex size-full">
        <Spinner className="m-auto size-6" />
      </div>
    );
  }
  if (prepared.status === "unavailable") {
    return (
      <div className="flex size-full">
        <p className="m-auto max-w-sm px-6 text-center text-sm text-muted-foreground">
          {needsNode(prepared.reason) ? t("browser.file.reactNeedsNode") : t("browser.file.reactFailed")}
        </p>
      </div>
    );
  }
  return (
    <ArtifactHtmlFrame
      code={prepared.html}
      title={title}
      kind="react"
      fill={true}
      reloadNonce={reloadNonce}
      consoleOpen={consoleOpen}
      onConsoleOpenChange={onConsoleOpenChange}
      onOutputCountChange={onOutputCountChange}
      onFixWithModel={onFixWithModel}
    />
  );
}
