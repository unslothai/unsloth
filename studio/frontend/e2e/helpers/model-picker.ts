// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { expect, type APIRequestContext, type Page } from "@playwright/test";

// Same derivation the Playwright config uses for its backendUrl metadata, so an
// E2E_BACKEND_PORT override reaches the API helpers too and not just the config.
// The page's own /api calls still go through the Vite dev proxy, which targets
// 127.0.0.1:8888 unconditionally.
const backendUrl =
  process.env.E2E_BACKEND_URL ??
  `http://127.0.0.1:${process.env.E2E_BACKEND_PORT ?? 8888}`;

/** How long to let a chat finish decoding before unloading anyway. */
const GENERATION_IDLE_TIMEOUT_MS = 5 * 60 * 1000;
const GENERATION_IDLE_POLL_MS = 500;

/**
 * Block until no conversation is decoding.
 *
 * An unforced /api/inference/unload refuses with 409 while any generation is
 * registered, and the UI signal a test can reach first (its own user bubble)
 * renders the moment the message is sent, long before the run ends. Poll the
 * counter the unload gate itself reads instead. A timeout falls through to the
 * unload so the 409 is still reported rather than swallowed here.
 */
export async function waitForGenerationsIdle(
  request: APIRequestContext,
  headers: Record<string, string>,
): Promise<number> {
  const deadline = Date.now() + GENERATION_IDLE_TIMEOUT_MS;
  let active = 0;
  for (;;) {
    const response = await request.get(
      `${backendUrl}/api/inference/active-generations`,
      { headers },
    );
    if (!response.ok()) return active;
    const payload = (await response.json()) as {
      count?: number;
      active?: unknown[];
    };
    active =
      typeof payload.count === "number"
        ? payload.count
        : (payload.active?.length ?? 0);
    if (active === 0) return 0;
    if (Date.now() >= deadline) return active;
    await new Promise((resolve) =>
      setTimeout(resolve, GENERATION_IDLE_POLL_MS),
    );
  }
}

export async function unloadInferenceModel(
  request: APIRequestContext,
): Promise<void> {
  const token = process.env.E2E_ACCESS_TOKEN;
  if (!token) {
    throw new Error("E2E_ACCESS_TOKEN is required to unload models.");
  }
  const headers = { Authorization: `Bearer ${token}` };
  // UnloadRequest.model_path is required (an empty body 422s), so ask the backend which
  // model is loaded, and skip when none is.
  const statusResponse = await request.get(
    `${backendUrl}/api/inference/status`,
    { headers },
  );
  expect(statusResponse.ok()).toBeTruthy();
  const status = (await statusResponse.json()) as {
    model_identifier?: string | null;
    active_model?: string | null;
    loaded?: string[];
  };
  const modelPath =
    status.model_identifier ?? status.active_model ?? status.loaded?.[0];
  if (!modelPath) return;

  // Unforced, so a chat still decoding would 409 the request below.
  await waitForGenerationsIdle(request, headers);

  const response = await request.post(`${backendUrl}/api/inference/unload`, {
    headers,
    data: { model_path: modelPath },
  });
  expect(response.ok()).toBeTruthy();
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

  // Arm the waiter before the load: waitForResponse only sees later responses and the
  // recount fires at load end. A brand-new chat never calls count_tokens, so race the bar.
  const tokenCount = waitForContextTokenCount(page).catch(() => undefined);
  await selectLocalModelByName(page, displayName);
  await Promise.race([tokenCount, waitForContextUsageBar(page)]);
  await waitForContextUsageBar(page);
}

export async function ejectLoadedModel(page: Page): Promise<void> {
  await openModelPicker(page);
  await page.getByRole("button", { name: "Eject model" }).click();
  await expect.poll(() => isInferenceLoaded(page), {
    timeout: 60_000,
  }).toBe(false);
}
