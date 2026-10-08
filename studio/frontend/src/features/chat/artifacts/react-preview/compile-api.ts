// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";
import type { CompileDiagnostic } from "./build-html";

export type UnavailableReason = "node_missing" | "transform_missing" | "timeout" | "failed";

export type CompileResult =
  | { status: "ok"; code: string; deps: string[] }
  | { status: "error"; diagnostics: CompileDiagnostic[] }
  | { status: "unavailable"; reason: UnavailableReason };

// The backend bounds a compile at 10 s; this leaves room for the queue in front of it.
const COMPILE_TIMEOUT_MS = 15_000;

export async function compileReactPreview(
  source: string,
  lang: "jsx" | "tsx",
  signal?: AbortSignal,
): Promise<CompileResult> {
  // Linked by hand rather than with AbortSignal.any, which Safari only got in 17.4.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), COMPILE_TIMEOUT_MS);
  const onAbort = () => controller.abort();
  signal?.addEventListener("abort", onAbort, { once: true });
  const response = await authFetch(apiUrl("/api/inference/artifact-react-compile"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, lang }),
    signal: controller.signal,
  })
    .catch((error: unknown) => {
      if (signal?.aborted) throw error;
      return null;
    })
    .finally(() => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    });
  if (response?.status === 413) {
    return {
      status: "error",
      diagnostics: [{ message: "This component is too large to preview (256 KB limit).", line: 0, column: 0 }],
    };
  }
  if (!response?.ok) return { status: "unavailable", reason: "failed" };
  const body = (await response.json().catch(() => null)) as CompileResult | null;
  if (body?.status === "ok" && typeof body.code === "string" && Array.isArray(body.deps)) return body;
  if (body?.status === "error" && Array.isArray(body.diagnostics)) return body;
  if (body?.status === "unavailable") return body;
  return { status: "unavailable", reason: "failed" };
}
