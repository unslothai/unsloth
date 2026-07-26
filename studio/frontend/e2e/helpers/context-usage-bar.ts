// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { expect, type Page } from "@playwright/test";

const TOKEN_BAR_RATIO = /\d[\d.]*k? \/ \d[\d.]*k?/;

function parseUsedTokensFromLabel(label: string | null): number {
  const match = label?.match(/Context usage:\s*([\d.]+)(k?)/);
  if (!match) return 0;
  const value = Number(match[1]);
  return match[2] === "k" ? Math.round(value * 1000) : value;
}

export async function readContextUsageTokenCount(page: Page): Promise<number> {
  const bar = page
    .getByTestId("context-usage-bar")
    .or(page.getByRole("button", { name: /Context usage:/ }));
  const label = await bar.getAttribute("aria-label");
  return parseUsedTokensFromLabel(label);
}

export async function expectContextUsageBarVisible(
  page: Page,
  options?: { minUsed?: number; poll?: boolean },
): Promise<void> {
  const bar = page
    .getByTestId("context-usage-bar")
    .or(page.getByRole("button", { name: /Context usage:/ }));
  await expect(bar).toBeVisible({ timeout: 120_000 });
  await expect(bar).toHaveText(TOKEN_BAR_RATIO);

  if (options?.minUsed != null && options.minUsed > 0) {
    const assertMinUsed = async () => {
      const used = await readContextUsageTokenCount(page);
      expect(used).toBeGreaterThanOrEqual(options.minUsed!);
    };

    if (options.poll) {
      await expect
        .poll(async () => readContextUsageTokenCount(page), {
          timeout: 180_000,
        })
        .toBeGreaterThanOrEqual(options.minUsed);
    } else {
      await assertMinUsed();
    }
  }
}
