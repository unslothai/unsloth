// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactElement, act } from "react";
import { type Root, createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/shared/toast", () => ({
  toastError: vi.fn(),
  toastSuccess: vi.fn(),
}));
vi.mock("../api", () => ({ removeUnstructuredBlock: vi.fn() }));
vi.mock("../stores/recipe-studio", () => ({
  useRecipeStudioStore: {
    getState: () => ({ pendingUploadCleanups: [] }),
    setState: vi.fn(),
  },
}));
vi.mock("../utils/import", () => ({
  importRecipePayload: () => ({ snapshot: { nodes: [] }, errors: [] }),
}));

import { useRecipePersistence } from "./use-recipe-persistence";

type HookResult = ReturnType<typeof useRecipePersistence>;
type Payload = Parameters<typeof useRecipePersistence>[0]["initialPayload"];
type PersistRecipeFn = Parameters<
  typeof useRecipePersistence
>[0]["onPersistRecipe"];

const initialPayload = { recipe: { processors: [] } } as unknown as Payload;
const credentialPayload = {
  recipe: { provider: { api_key: "session-only" } },
} as unknown as Payload;
const onPersistRecipe = vi.fn<PersistRecipeFn>();
const onReloadRecipe = vi.fn();
const resetRecipe = vi.fn();
const loadRecipe = vi.fn();
const getCurrentPayloadFromStore = vi.fn(() => initialPayload);
let latest: HookResult;
let currentPayload = initialPayload;

function Harness(): ReactElement | null {
  // eslint-disable-next-line react-hooks/globals -- test harness exposes hook state
  latest = useRecipePersistence({
    recipeId: "recipe-1",
    initialRecipeName: "Recipe",
    initialPayload,
    initialSavedAt: 1,
    initialRevision: 1,
    payloadResult: { payload: currentPayload, errors: [] } as never,
    onPersistRecipe,
    onReloadRecipe,
    resetRecipe,
    loadRecipe,
    getCurrentPayloadFromStore,
  });
  return null;
}

describe("authoritative recipe projection", () => {
  let root: Root;
  let container: HTMLDivElement;

  beforeEach(async () => {
    vi.useFakeTimers();
    currentPayload = initialPayload;
    onPersistRecipe.mockReset().mockResolvedValue({
      id: "recipe-1",
      updatedAt: 90,
      revision: 2,
      payload: initialPayload,
      removedCredentialPaths: ["recipe.provider.api_key"],
    });
    container = document.createElement("div");
    root = createRoot(container);
    await act(async () => root.render(<Harness />));
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    vi.useRealTimers();
  });

  it("signs the server projection and requests stripped credentials after reload", async () => {
    currentPayload = credentialPayload;
    await act(async () => root.render(<Harness />));
    await act(async () => vi.advanceTimersByTimeAsync(800));
    expect(latest.saveTone).toBe("warning");
    expect(latest.savedAtLabel).toContain("Re-enter after reload");
    await act(async () => vi.advanceTimersByTimeAsync(2400));
    expect(onPersistRecipe).toHaveBeenCalledTimes(1);
  });
});
