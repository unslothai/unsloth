// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { Suspense, lazy } from "react";
import type { DocumentKind } from "./kind";

// Split out: pdf.js and the OOXML readers load only when a document is opened.
const PdfView = lazy(() => import("./pdf-view"));
const OfficeView = lazy(() => import("./office-view"));

/** A PDF, Word document, spreadsheet or deck, with `scale` 1 fitting the page to the pane. */
export function DocumentView({
  file,
  kind,
  name,
  scale = 1,
}: {
  file: Blob;
  kind: DocumentKind;
  name: string;
  scale?: number;
}) {
  return (
    <Suspense fallback={<Spinner className="m-auto size-6" />}>
      {kind === "pdf" ? (
        <PdfView file={file} scale={scale} />
      ) : (
        <OfficeView file={file} kind={kind} name={name} scale={scale} />
      )}
    </Suspense>
  );
}
