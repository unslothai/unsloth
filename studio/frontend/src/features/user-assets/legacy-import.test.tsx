// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { act } from "react";
import { type Root, createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  bootstrap: vi.fn(),
  importAssets: vi.fn(),
  onImported: vi.fn(),
  toastError: vi.fn(),
  toastSuccess: vi.fn(),
}));

vi.mock("@/i18n", () => ({
  useT: () => (key: string, values?: Record<string, unknown>) => {
    const labels: Record<string, string> = {
      "dataRecipes.import.confirm": "Import localized data",
      "dataRecipes.import.close": "Close localized dialog",
      "dataRecipes.import.rejected": "Rejected localized",
      "dataRecipes.import.reasons.invalidName": "Invalid name localized",
      "dataRecipes.import.recipeCountOther": `${values?.count ?? 0} localized recipes`,
      "dataRecipes.import.executionCountOther": `${values?.count ?? 0} localized executions`,
    };
    return labels[key] ?? `localized:${key}`;
  },
}));
vi.mock("@/shared/toast", () => ({
  toastError: mocks.toastError,
  toastSuccess: mocks.toastSuccess,
}));
vi.mock("./api", () => ({
  bootstrapUserAssets: mocks.bootstrap,
  importLegacyUserAssets: mocks.importAssets,
}));

import { LegacyImportCoordinator } from "./legacy-import";

type LegacyItem = {
  id: string;
  recipeId?: string;
  name?: string;
  payload?: object;
};

function pageReader(items: LegacyItem[]) {
  return vi.fn(async (cursor?: string | null, limit = 100) => {
    const start = cursor
      ? items.findIndex((item) => item.id === cursor) + 1
      : 0;
    const page = items.slice(start, start + limit);
    return {
      items: page,
      nextCursor:
        start + limit < items.length ? (page.at(-1)?.id ?? null) : null,
    };
  });
}

async function flush(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe("LegacyImportCoordinator", () => {
  let root: Root;
  let container: HTMLDivElement;

  beforeEach(() => {
    for (const mock of Object.values(mocks)) mock.mockReset();
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  it("keeps a cancelled migration reopenable in the same session", async () => {
    const readRecipes = pageReader([{ id: "r", name: "Recipe", payload: {} }]);
    mocks.bootstrap.mockResolvedValue({
      subject: "alice",
      importLedger: { source: "recipe-indexeddb-v1", recipes: [], executions: [] },
    });
    await act(async () => {
      root.render(
        <LegacyImportCoordinator
          onImported={mocks.onImported}
          readRecipes={readRecipes}
          readExecutions={pageReader([])}
        />,
      );
    });
    await flush();
    const cancel = [...document.querySelectorAll("button")].find((button) =>
      button.textContent?.includes("cancel"),
    );
    await act(async () => cancel?.click());
    const reopen = [...document.querySelectorAll("button")].find(
      (button) => button.textContent === "localized:dataRecipes.import.title",
    );
    expect(reopen).toBeTruthy();
    await act(async () => reopen?.click());
    expect(document.body.textContent).toContain("Import localized data");
  });

  it("rejects one unsplittable Unicode record locally and continues later records", async () => {
    const huge = "é".repeat(4_200_000);
    const readRecipes = pageReader([
      { id: "too-large", name: "Large", payload: { huge } },
      { id: "valid", name: "Valid", payload: {} },
    ]);
    mocks.bootstrap.mockResolvedValue({
      subject: "alice",
      importLedger: { source: "recipe-indexeddb-v1", recipes: [], executions: [] },
    });
    mocks.importAssets.mockImplementation(async (input: { recipes: LegacyItem[] }) => ({
      recipes: input.recipes.map((item) => ({ id: item.id, outcome: "imported" })),
      executions: [],
      summary: { imported: input.recipes.length },
    }));
    await act(async () => {
      root.render(
        <LegacyImportCoordinator
          onImported={mocks.onImported}
          readRecipes={readRecipes}
          readExecutions={pageReader([])}
        />,
      );
    });
    await flush();
    const confirm = [...document.querySelectorAll("button")].find(
      (button) => button.textContent === "Import localized data",
    );
    await act(async () => confirm?.click());
    await flush();
    expect(mocks.importAssets).toHaveBeenCalledTimes(1);
    expect(mocks.importAssets.mock.calls[0][0].recipes).toEqual([
      expect.objectContaining({ id: "valid" }),
    ]);
    expect(document.body.textContent).toContain("too-large");
    expect(document.querySelector('[role="status"]')).not.toBeNull();
  });
});
