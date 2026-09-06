// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Run against a disposable integrated Studio server. No public signup or owner bootstrap.
// Set STUDIO_E2E_URL and STUDIO_E2E_OWNER_PASSWORD; the owner must already have a password.
import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

const baseURL = process.env.STUDIO_E2E_URL ?? "http://127.0.0.1:8000";
const ownerPassword = process.env.STUDIO_E2E_OWNER_PASSWORD;

async function login(page: Page, username: string, password: string) {
  await page.goto(`${baseURL}/login`);
  const usernameField = page.getByRole("textbox", {
    name: "Username",
    exact: true,
  });
  if (username !== "unsloth") await expect(usernameField).toBeVisible();
  if (await usernameField.isVisible()) await usernameField.fill(username);
  await page.locator("#password").fill(password);
  await page.getByRole("button", { name: "Login", exact: true }).click();
}

async function logOut(page: Page) {
  await page.evaluate(async () => {
    await fetch("/api/auth/logout", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("unsloth_auth_token")}`,
      },
    });
    for (const key of [
      "unsloth_auth_token",
      "unsloth_auth_refresh_token",
      "unsloth_auth_must_change_password",
      "unsloth_auth_session_mark",
    ]) {
      localStorage.removeItem(key);
    }
  });
}

test("owner creates an account, setup is private, and browser switching clears account data", async ({
  page,
  context,
  request,
}) => {
  test.skip(
    !ownerPassword,
    "Requires a disposable Studio server and owner password.",
  );
  const username = `e2e_${Date.now()}`;
  const managedPassword = `Managed-${Date.now()}-password`;
  const ownerLogin = await request.post(`${baseURL}/api/auth/login`, {
    data: { username: "unsloth", password: ownerPassword },
  });
  expect(ownerLogin.ok()).toBeTruthy();
  const { access_token: ownerToken } = await ownerLogin.json();
  const ownerHeaders = { Authorization: `Bearer ${ownerToken}` };
  let created = false;
  let accountId = "";
  try {
    const initialStatus = await request.get(`${baseURL}/api/auth/status`);
    const initialMode = (await initialStatus.json()).login_mode;
    // The owner password-only browser still keeps its existing local data.
    await page.goto(`${baseURL}/login`);
    await page.evaluate(() => {
      localStorage.setItem("unsloth_locale", "en");
      localStorage.setItem("theme", "dark");
      localStorage.setItem("unsloth_e2e_private", "owner-private");
      localStorage.setItem("chat-draft:e2e", "owner-draft");
      localStorage.setItem("unsloth_chat_permission_mode", "full");
    });
    if (initialMode === "single")
      await expect(page.locator("#username")).toHaveCount(0);
    await login(page, "unsloth", ownerPassword!);
    await expect(page).toHaveURL(/\/chat/);
    expect(
      await page.evaluate(() => localStorage.getItem("unsloth_e2e_private")),
    ).toBe("owner-private");

    // Use the documented admin API to avoid depending on a platform-specific Settings shortcut.
    const response = await request.post(`${baseURL}/api/accounts`, {
      headers: ownerHeaders,
      data: { username },
    });
    expect(response.ok()).toBeTruthy();
    created = true;
    const { account, setup_code, setup_code_expires_at } =
      await response.json();
    accountId = account.account_id;
    expect(setup_code).toBeTruthy();
    expect(Date.parse(setup_code_expires_at)).toBeGreaterThan(Date.now());
    const status = await request.get(`${baseURL}/api/auth/status`);
    expect((await status.json()).login_mode).toBe("multi");
    await logOut(page);
    await page.goto(`${baseURL}/login`);
    const secondTab = await context.newPage();
    await secondTab.goto(`${baseURL}/login`);
    await expect(secondTab.locator("#username")).toBeVisible();
    await secondTab.evaluate(() => {
      (window as Window & { oldAccountDocument?: boolean }).oldAccountDocument =
        true;
    });

    await login(page, username.toUpperCase(), setup_code);
    await expect(page).toHaveURL(/\/change-password/);
    expect(
      await page.evaluate(() => localStorage.getItem("unsloth_e2e_private")),
    ).toBeNull();
    expect(
      await page.evaluate(() => localStorage.getItem("chat-draft:e2e")),
    ).toBeNull();
    expect(await page.evaluate(() => localStorage.getItem("theme"))).toBe(
      "dark",
    );
    expect(
      await page.evaluate(() =>
        localStorage.getItem("unsloth_chat_permission_mode"),
      ),
    ).not.toBe("full");
    await expect
      .poll(() =>
        secondTab.evaluate(
          () =>
            (window as Window & { oldAccountDocument?: boolean })
              .oldAccountDocument,
        ),
      )
      .toBeUndefined();

    await page.locator("#current-password").fill(setup_code);
    await page.locator("#new-password").fill(managedPassword);
    await page.locator("#confirm-password").fill(managedPassword);
    await page
      .getByRole("button", { name: "Change password", exact: true })
      .click();
    await expect(page).toHaveURL(/\/chat/);
    const managedToken = await page.evaluate(() =>
      localStorage.getItem("unsloth_auth_token"),
    );
    const denied = await request.get(`${baseURL}/api/accounts`, {
      headers: { Authorization: `Bearer ${managedToken}` },
    });
    expect(denied.status()).toBe(403);
    // A persisted Accounts selection cannot mount or expose the owner-only panel.
    await page.evaluate(() =>
      localStorage.setItem("unsloth_settings_active_tab", "accounts"),
    );
    await page.reload();
    await expect(page.getByTestId("settings-tab-accounts")).toHaveCount(0);
    await page.evaluate(() =>
      localStorage.setItem("unsloth_e2e_private", "managed-private"),
    );
    await logOut(page);
    await login(page, "unsloth", ownerPassword!);
    await expect(page).toHaveURL(/\/chat/);
    expect(
      await page.evaluate(() => localStorage.getItem("unsloth_e2e_private")),
    ).toBeNull();
    await secondTab.close();
  } finally {
    if (created) {
      const response = await request.delete(
        `${baseURL}/api/accounts/${encodeURIComponent(accountId)}`,
        { headers: ownerHeaders },
      );
      expect(response.ok()).toBeTruthy();
    }
  }
});
