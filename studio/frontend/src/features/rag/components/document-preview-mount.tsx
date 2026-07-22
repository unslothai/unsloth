// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Suspense, lazy, useEffect, useState } from "react";

import { useDocumentPreviewStore } from "./preview-store";

// pdf.js / react-pdf are heavy (~0.5 MB gzip): defer until the first citation
// click, then keep mounted for open/close anims.
const DocumentPreviewSheet = lazy(() =>
  import("./document-preview-sheet").then((m) => ({
    default: m.DocumentPreviewSheet,
  })),
);

export function DocumentPreviewMount() {
  const open = useDocumentPreviewStore((s) => s.open);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    if (open) setMounted(true);
  }, [open]);

  if (!mounted) return null;
  return (
    <Suspense fallback={null}>
      <DocumentPreviewSheet />
    </Suspense>
  );
}
