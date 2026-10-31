// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  DiagramPlugin,
  MermaidConfig,
  MermaidInstance,
} from "@streamdown/mermaid";

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

// Mermaid loads with the first diagram; until then the last config handed in is kept and applied on load.
let stock: DiagramPlugin | null = null;
let loading: Promise<DiagramPlugin> | null = null;
let pendingConfig: MermaidConfig | undefined;

function loadStock(): Promise<DiagramPlugin> {
  if (!loading) {
    loading = import("@streamdown/mermaid").then(({ createMermaidPlugin }) => {
      stock = createMermaidPlugin({ config: { secure: SECURE_KEYS } });
      if (pendingConfig) {
        stock.getMermaid(pendingConfig);
      }
      pendingConfig = undefined;
      return stock;
    });
    // A failed chunk fetch retries on the next diagram.
    loading.catch(() => {
      loading = null;
    });
  }
  return loading;
}

const lazyInstance: MermaidInstance = {
  initialize(config) {
    if (stock) {
      stock.getMermaid().initialize(config);
    } else {
      pendingConfig = config;
    }
  },
  async render(id, source) {
    const [instance, { stripRemoteDiagramImages }] = await Promise.all([
      loadStock().then((plugin) => plugin.getMermaid()),
      import("./mermaid-guard.ts"),
    ]);
    return instance.render(id, stripRemoteDiagramImages(source));
  },
};

/**
 * Replaces the `mermaid` export of @streamdown/mermaid, one instance for every surface that draws
 * diagrams, so they share one mermaid config as they did. Every render goes through
 * `stripRemoteDiagramImages`, which keeps a diagram to the chat's image policy. Package imports
 * only: the chat barrel reaches this module while it loads.
 */
export const mermaid: DiagramPlugin = {
  name: "mermaid",
  type: "diagram",
  language: "mermaid",
  getMermaid(config) {
    if (config) {
      lazyInstance.initialize(config);
    }
    return lazyInstance;
  },
};
