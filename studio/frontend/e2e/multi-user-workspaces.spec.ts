// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { expect, test, type APIRequestContext, type Browser } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

type Tokens = {
  access_token: string;
  refresh_token: string;
  must_change_password: boolean;
};

const baseURL = process.env.STUDIO_E2E_URL ?? "http://127.0.0.1:8767";
const evidenceDir = process.env.STUDIO_EVIDENCE_DIR;
const ownerBootstrapPassword = process.env.STUDIO_E2E_ADMIN_PASSWORD;
const ownerPassword = "owner-e2e-password";
const alicePassword = "alice-e2e-password";
const bobPassword = "bob-e2e-password";

async function login(request: APIRequestContext, username: string, password: string) {
  const response = await request.post(`${baseURL}/api/auth/login`, {
    data: { username, password },
  });
  expect(response.ok(), `login ${username}: ${await response.text()}`).toBeTruthy();
  return (await response.json()) as Tokens;
}

function bearer(tokens: Tokens) {
  return { Authorization: `Bearer ${tokens.access_token}` };
}

async function replaceTemporaryPassword(
  request: APIRequestContext,
  username: string,
  temporaryPassword: string,
  password: string,
) {
  const tokens = await login(request, username, temporaryPassword);
  const response = await request.post(`${baseURL}/api/auth/change-password`, {
    headers: bearer(tokens),
    data: { current_password: temporaryPassword, new_password: password },
  });
  expect(response.ok(), `change password for ${username}: ${await response.text()}`).toBeTruthy();
  return (await response.json()) as Tokens;
}

async function loginInBrowser(browser: Browser, username: string, password: string) {
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.goto(`${baseURL}/login`);
  await page.getByLabel("Username").fill(username);
  await page.getByLabel("Password", { exact: true }).fill(password);
  await page.getByRole("button", { name: "Login", exact: true }).click();
  await expect(page).not.toHaveURL(/\/(login|change-password)(?:\?|$)/);
  return { context, page };
}

test("owner account UX and per-account chat workspaces are isolated", async ({
  browser,
  page,
  request,
}) => {
  test.skip(!ownerBootstrapPassword, "STUDIO_E2E_ADMIN_PASSWORD is required");
  if (evidenceDir) await mkdir(evidenceDir, { recursive: true });

  const initialOwner = await login(request, "unsloth", ownerBootstrapPassword!);
  const owner = initialOwner.must_change_password
    ? await replaceTemporaryPassword(
        request,
        "unsloth",
        ownerBootstrapPassword!,
        ownerPassword,
      )
    : await login(request, "unsloth", ownerPassword);

  for (const username of ["alice", "bob"]) {
    const existing = await request.delete(`${baseURL}/api/auth/users/${username}`, {
      headers: bearer(owner),
    });
    expect([204, 404]).toContain(existing.status());
  }

  const aliceTemporaryPassword = "alice-temporary-password";
  const bobTemporaryPassword = "bob-temporary-password";
  const accountCreator = await loginInBrowser(browser, "unsloth", ownerPassword);
  await accountCreator.page.getByRole("button", { name: "Settings", exact: true }).last().click();
  await accountCreator.page.getByTestId("settings-tab-accounts").click();
  const accountsDialog = accountCreator.page.getByRole("dialog");
  await accountsDialog.getByLabel("Username").fill("alice");
  await accountsDialog.getByLabel("Temporary password").fill(aliceTemporaryPassword);
  expect(await accountsDialog.locator("form").evaluate((form) => form.checkValidity())).toBe(true);
  const createAliceResponse = accountCreator.page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/auth/users") && response.request().method() === "POST",
  );
  await accountsDialog.getByRole("button", { name: "Create account", exact: true }).click();
  expect((await createAliceResponse).status()).toBe(201);
  await expect(accountsDialog.getByText("alice", { exact: true })).toBeVisible();
  await expect(accountsDialog.getByText("Password change required", { exact: true })).toBeVisible();
  await accountCreator.context.close();

  const firstLoginContext = await browser.newContext();
  const firstLoginPage = await firstLoginContext.newPage();
  await firstLoginPage.goto(`${baseURL}/login`);
  await firstLoginPage.getByLabel("Username").fill("alice");
  await firstLoginPage.getByLabel("Password", { exact: true }).fill(aliceTemporaryPassword);
  await firstLoginPage.getByRole("button", { name: "Login", exact: true }).click();
  await expect(firstLoginPage).toHaveURL(/\/change-password(?:\?|$)/);
  await expect(
    firstLoginPage.getByRole("heading", { name: "Setup your account", exact: true }),
  ).toBeVisible();
  await firstLoginContext.close();

  const createdBob = await request.post(`${baseURL}/api/auth/users`, {
    headers: bearer(owner),
    data: { username: "bob", password: bobTemporaryPassword },
  });
  expect(createdBob.status(), `create bob: ${await createdBob.text()}`).toBe(201);

  const alice = await replaceTemporaryPassword(
    request,
    "alice",
    aliceTemporaryPassword,
    alicePassword,
  );
  const bob = await replaceTemporaryPassword(request, "bob", bobTemporaryPassword, bobPassword);

  const aliceThread = {
    id: "shared-browser-proof-id",
    title: "Alice private launch plan",
    modelType: "base",
    modelId: "",
    createdAt: 1_787_851_200_000,
  };
  const saved = await request.post(`${baseURL}/api/chat/threads`, {
    headers: bearer(alice),
    data: aliceThread,
  });
  expect(saved.ok(), await saved.text()).toBeTruthy();

  const aliceThreadsResponse = await request.get(`${baseURL}/api/chat/threads`, {
    headers: bearer(alice),
  });
  const bobThreadsResponse = await request.get(`${baseURL}/api/chat/threads`, {
    headers: bearer(bob),
  });
  expect(aliceThreadsResponse.ok()).toBeTruthy();
  expect(bobThreadsResponse.ok()).toBeTruthy();
  const aliceThreads = (await aliceThreadsResponse.json()) as { threads: Array<{ title: string }> };
  const bobThreads = (await bobThreadsResponse.json()) as { threads: Array<{ title: string }> };
  expect(aliceThreads.threads.map((thread) => thread.title)).toContain(aliceThread.title);
  expect(bobThreads.threads).toEqual([]);

  await page.goto(`${baseURL}/login`);
  await expect(page.getByLabel("Username")).toBeVisible();
  await expect(page.getByLabel("Password", { exact: true })).toBeVisible();
  if (evidenceDir) {
    await page.screenshot({ path: path.join(evidenceDir, "01-login-user-selection.png") });
  }

  const ownerBrowser = await loginInBrowser(browser, "unsloth", ownerPassword);
  await ownerBrowser.page.getByRole("button", { name: "Settings", exact: true }).last().click();
  await ownerBrowser.page.getByTestId("settings-tab-accounts").click();
  await expect(ownerBrowser.page.getByRole("heading", { name: "Accounts", exact: true })).toBeVisible();
  await expect(ownerBrowser.page.getByText("alice", { exact: true })).toBeVisible();
  await expect(ownerBrowser.page.getByText("bob", { exact: true })).toBeVisible();
  if (evidenceDir) {
    await ownerBrowser.page.screenshot({ path: path.join(evidenceDir, "02-owner-accounts.png") });
  }
  await ownerBrowser.context.close();

  const aliceBrowser = await loginInBrowser(browser, "alice", alicePassword);
  await expect(aliceBrowser.page.getByTestId("current-account-username")).toHaveText("@alice");
  await expect(aliceBrowser.page.getByText(aliceThread.title, { exact: true })).toBeVisible();
  if (evidenceDir) {
    await aliceBrowser.page.screenshot({ path: path.join(evidenceDir, "03-alice-workspace.png") });
  }
  await aliceBrowser.context.close();

  const bobBrowser = await loginInBrowser(browser, "bob", bobPassword);
  await expect(bobBrowser.page.getByTestId("current-account-username")).toHaveText("@bob");
  await expect(bobBrowser.page.getByText(aliceThread.title, { exact: true })).toHaveCount(0);
  await expect(bobBrowser.page.getByText("No chats yet", { exact: true })).toBeVisible();
  if (evidenceDir) {
    await bobBrowser.page.screenshot({ path: path.join(evidenceDir, "04-bob-workspace.png") });
    await writeFile(
      path.join(evidenceDir, "facts.json"),
      `${JSON.stringify(
        {
          baseURL,
          loginHasUsernameField: true,
          managedAccounts: ["unsloth", "alice", "bob"],
          aliceThreadTitles: aliceThreads.threads.map((thread) => thread.title),
          bobThreadTitles: bobThreads.threads.map((thread) => thread.title),
          sameServer: true,
        },
        null,
        2,
      )}\n`,
    );
  }
  await bobBrowser.context.close();
});
