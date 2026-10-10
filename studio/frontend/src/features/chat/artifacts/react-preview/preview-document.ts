// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Loaded only when a React preview opens. It imports nothing the app shares, so the startup
// chunks stay as they were.

import { type CompileDiagnostic, buildCompileErrorHtml, buildReactPreviewHtml, usesTailwind } from "./build-html";
import { runtimeLoader } from "./runtime-manifest";

/** The page for compiled code, with only the libraries it imports. */
export async function buildPreviewDocument({
  code,
  deps,
  source,
  lang,
  title,
}: {
  code: string;
  deps: readonly string[];
  source: string;
  lang: "jsx" | "tsx";
  title: string;
}): Promise<string> {
  const { names } = runtimeLoader.resolve(deps);
  const [runtimes, tailwind] = await Promise.all([
    Promise.all(names.map(async (name) => ({ name, code: await runtimeLoader.load(name) }))),
    usesTailwind(source) ? runtimeLoader.load(runtimeLoader.tailwind) : Promise.resolve(null),
  ]);
  return buildReactPreviewHtml({
    code,
    runtimes,
    tailwind,
    available: runtimeLoader.available,
    title,
    file: `App.${lang}`,
  });
}

export function buildCompileErrorDocument(diagnostics: readonly CompileDiagnostic[], title: string): string {
  return buildCompileErrorHtml(diagnostics, title);
}
