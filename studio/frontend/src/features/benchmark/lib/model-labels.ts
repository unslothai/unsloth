// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function modelLabels(models: string[]): Record<string, string> {
  const parts = (model: string) => model.split(/[\\/]/).filter(Boolean);
  const label = (model: string, depth: number) => {
    const p = parts(model);
    return p.length ? p.slice(-depth).join("/") : model;
  };
  const labels: Record<string, string> = {};
  for (const model of models) {
    const deepest = Math.max(1, parts(model).length);
    let depth = 1;
    while (
      depth < deepest &&
      models.some((other) => other !== model && label(other, depth) === label(model, depth))
    )
      depth += 1;
    labels[model] = label(model, depth);
  }
  return labels;
}
