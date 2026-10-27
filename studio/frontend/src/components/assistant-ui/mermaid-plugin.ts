// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createMermaidPlugin } from "@streamdown/mermaid";

// Mermaid's own secure keys, plus the three CSS strings a `%%{init}%%` directive or front matter
// would otherwise write into the diagram's <style>, where a url() would load from elsewhere.
const SECURE_KEYS = [
  "secure",
  "securityLevel",
  "startOnLoad",
  "maxTextSize",
  "suppressErrorRendering",
  "maxEdges",
  "themeCSS",
  "fontFamily",
  "altFontFamily",
];

const stock = createMermaidPlugin({ config: { secure: SECURE_KEYS } });

/**
 * Replaces the `mermaid` export of @streamdown/mermaid, one instance for every surface that draws
 * diagrams, so they share one mermaid config as they did. Every render goes through
 * `stripRemoteDiagramImages`, which keeps a diagram to the chat's image policy. Package imports
 * only: the chat barrel reaches this module while it loads.
 */
export const mermaid: typeof stock = {
  ...stock,
  getMermaid(config) {
    const instance = stock.getMermaid(config);
    return {
      ...instance,
      async render(id, source) {
        // Loaded with the first diagram, not with the app: diagrams already render lazily.
        const { stripRemoteDiagramImages } = await import("./mermaid-guard.ts");
        return instance.render(id, stripRemoteDiagramImages(source));
      },
    };
  },
};
