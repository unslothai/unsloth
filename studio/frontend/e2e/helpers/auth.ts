// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { BrowserContext, Page } from "@playwright/test";

const AUTH_TOKEN_KEY = "unsloth_auth_token";
const AUTH_REFRESH_TOKEN_KEY = "unsloth_auth_refresh_token";
const ONBOARDING_DONE_KEY = "unsloth_onboarding_done";

export async function seedAuthenticatedSession(
  context: BrowserContext,
): Promise<void> {
  const accessToken = process.env.E2E_ACCESS_TOKEN;
  const refreshToken = process.env.E2E_REFRESH_TOKEN;
  if (!accessToken || !refreshToken) {
    throw new Error(
      "Missing E2E tokens. global-setup should populate E2E_ACCESS_TOKEN and E2E_REFRESH_TOKEN.",
    );
  }

  await context.addInitScript(
    ({ access, refresh, onboardingKey, tokenKey, refreshKey }) => {
      localStorage.setItem(tokenKey, access);
      localStorage.setItem(refreshKey, refresh);
      localStorage.setItem(onboardingKey, "1");
    },
    {
      access: accessToken,
      refresh: refreshToken,
      onboardingKey: ONBOARDING_DONE_KEY,
      tokenKey: AUTH_TOKEN_KEY,
      refreshKey: AUTH_REFRESH_TOKEN_KEY,
    },
  );
}

export async function gotoChat(page: Page): Promise<void> {
  await page.goto("/chat");
  await page.waitForURL(/\/chat/);
  const snooze = page.getByTestId("llama-update-snooze-button");
  if (await snooze.isVisible().catch(() => false)) {
    await snooze.click({ force: true });
  }
}
