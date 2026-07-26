// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { expect, test } from "@playwright/test";
import { gotoChat, seedAuthenticatedSession } from "./helpers/auth";
import {
  expectContextUsageBarVisible,
} from "./helpers/context-usage-bar";
import {
  ensureLocalModelLoaded,
  selectLocalModelByName,
  unloadInferenceModel,
} from "./helpers/model-picker";

const LOCAL_MODEL_NAME =
  process.env.E2E_LOCAL_GGUF_NAME ?? "test-load-config";

test.describe("context usage bar", () => {
  test.beforeEach(async ({ context, request }) => {
    await seedAuthenticatedSession(context);
    await unloadInferenceModel(request);
  });

  test("shows token bar after model load without sending a message", async ({
    page,
  }) => {
    await gotoChat(page);
    await ensureLocalModelLoaded(page, LOCAL_MODEL_NAME);
    await expectContextUsageBarVisible(page);
  });

  test("shows token bar after model switch without sending again", async ({
    page,
    request,
  }) => {
    test.setTimeout(12 * 60 * 1000);

    await gotoChat(page);
    await ensureLocalModelLoaded(page, LOCAL_MODEL_NAME);

    const composer = page.getByRole("textbox", { name: "Message input" });
    await expect(composer).toBeVisible({ timeout: 120_000 });
    await composer.fill("Hi");
    const sendButton = page.getByRole("button", { name: "Send message" });
    await expect(sendButton).toBeEnabled({ timeout: 120_000 });
    await sendButton.click();

    await expect(page.getByText("Hi", { exact: true }).first()).toBeVisible({
      timeout: 60_000,
    });

    await unloadInferenceModel(request);
    await selectLocalModelByName(page, LOCAL_MODEL_NAME);
    await expectContextUsageBarVisible(page, { minUsed: 1, poll: true });
  });
});
