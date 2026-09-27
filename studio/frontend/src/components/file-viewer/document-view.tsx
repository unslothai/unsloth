// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { Suspense, lazy } from "react";
import type { DocumentKind } from "./kind";

const PdfView = lazy(() => import("./pdf-view"));
const OfficeView = lazy(() => import("./office-view"));

export function DocumentView({
  file,
  kind,
  name,
  contentType = "",
  scale = 1,
}: {
  file: Blob;
  kind: DocumentKind;
  name: string;
  contentType?: string;
  scale?: number;
}) {
  return (
    <Suspense fallback={<Spinner className="m-auto size-6" />}>
      {kind === "pdf" ? (
        <PdfView file={file} scale={scale} />
      ) : (
        <OfficeView file={file} kind={kind} name={name} contentType={contentType} scale={scale} />
      )}
    </Suspense>
  );
}
