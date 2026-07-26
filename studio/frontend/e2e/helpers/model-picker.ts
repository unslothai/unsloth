// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { expect, type APIRequestContext, type Page } from "@playwright/test";

const backendUrl =
  process.env.E2E_BACKEND_URL ?? "http://127.0.0.1:8888";

export async function unloadInferenceModel(
  request: APIRequestContext,
): Promise<void> {
  const token = process.env.E2E_ACCESS_TOKEN;
  if (!token) {
    throw new Error("E2E_ACCESS_TOKEN is required to unload models.");
  }
  await request.post(`${backendUrl}/api/inference/unload`, {
    headers: { Authorization: `Bearer ${token}` },
    data: {},
  });
}

function modelSelectorTrigger(page: Page) {
  return page.locator(
    '[data-testid="chat-model-selector"], [data-tour="chat-model-selector"]',
  );
}

function contextUsageBar(page: Page) {
  return page
    .getByTestId("context-usage-bar")
    .or(page.getByRole("button", { name: /Context usage:/ }));
}

export async function isInferenceLoaded(page: Page): Promise<boolean> {
  return page.evaluate(async () => {
    const token = localStorage.getItem("unsloth_auth_token");
    if (!token) return false;
    const response = await fetch("/api/inference/status", {
      headers: { Authorization: `Bearer ${token}` },
    });
    const data = await response.json();
    const loaded = data.loaded;
    return Array.isArray(loaded) ? loaded.length > 0 : Boolean(loaded);
  });
}

export async function openModelPicker(page: Page): Promise<void> {
  await modelSelectorTrigger(page).click();
  await expect(page.locator("[data-model-picker-search-input]")).toBeVisible();
}

export async function selectLocalModelByName(
  page: Page,
  displayName: string,
): Promise<void> {
  await openModelPicker(page);
  await page.getByRole("tab", { name: "On Device" }).click();
  await page.locator("[data-model-picker-search-input]").fill(displayName);
  const option = page
    .locator("[data-model-picker-option]")
    .filter({ hasText: displayName })
    .first();
  await expect(option).toBeVisible({ timeout: 60_000 });

  const loadPromise = page.waitForResponse(
    (response) =>
      response.url().includes("/api/inference/load") &&
      response.status() === 200,
    { timeout: 180_000 },
  );
  await option.click();
  await loadPromise;
  await waitForInferenceLoaded(page);
}

export async function waitForInferenceLoaded(page: Page): Promise<void> {
  await expect.poll(() => isInferenceLoaded(page), {
    timeout: 180_000,
  }).toBe(true);
}

export async function waitForContextTokenCount(page: Page): Promise<void> {
  await page.waitForResponse(
    (response) =>
      response.url().includes("/api/inference/chat/count_tokens") &&
      response.status() === 200,
    { timeout: 180_000 },
  );
}

export async function waitForContextUsageBar(
  page: Page,
  options?: { timeout?: number },
): Promise<void> {
  await expect(contextUsageBar(page)).toBeVisible({
    timeout: options?.timeout ?? 180_000,
  });
}

export async function ensureLocalModelLoaded(
  page: Page,
  displayName: string,
  options?: { force?: boolean },
): Promise<void> {
  if (!options?.force) {
    const inferenceLoaded = await isInferenceLoaded(page);
    const triggerText =
      (await modelSelectorTrigger(page).textContent()) ?? "";
    if (
      inferenceLoaded &&
      triggerText.includes(displayName) &&
      (await contextUsageBar(page).isVisible().catch(() => false))
    ) {
      return;
    }
  }

  await selectLocalModelByName(page, displayName);
  await waitForContextTokenCount(page).catch(() => undefined);
  await waitForContextUsageBar(page);
}

export async function ejectLoadedModel(page: Page): Promise<void> {
  await openModelPicker(page);
  await page.getByRole("button", { name: "Eject model" }).click();
  await expect.poll(() => isInferenceLoaded(page), {
    timeout: 60_000,
  }).toBe(false);
}
