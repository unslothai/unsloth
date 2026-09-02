// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  type APIRequestContext,
  type Browser,
  expect,
  test,
} from "@playwright/test";

type Tokens = {
  access_token: string;
  refresh_token: string;
  must_change_password: boolean;
};

type CreatedAccount = {
  username: string;
  setup_code: string;
  setup_code_expires_at: string;
};

const baseURL = process.env.STUDIO_E2E_URL ?? "http://127.0.0.1:8767";
const evidenceDir = process.env.STUDIO_EVIDENCE_DIR;
const ownerBootstrapPassword = process.env.STUDIO_E2E_ADMIN_PASSWORD;
const ownerPassword = "owner-e2e-password";
const alicePassword = "alice-e2e-password";
const bobPassword = "bob-e2e-password";

async function login(
  request: APIRequestContext,
  username: string,
  password: string,
) {
  const response = await request.post(`${baseURL}/api/auth/login`, {
    data: { username, password },
  });
  expect(
    response.ok(),
    `login ${username}: ${await response.text()}`,
  ).toBeTruthy();
  return (await response.json()) as Tokens;
}

function bearer(tokens: Tokens) {
  return { Authorization: `Bearer ${tokens.access_token}` };
}

async function replaceSetupCode(
  request: APIRequestContext,
  username: string,
  setupCode: string,
  password: string,
) {
  const tokens = await login(request, username, setupCode);
  const response = await request.post(`${baseURL}/api/auth/change-password`, {
    headers: bearer(tokens),
    data: { current_password: setupCode, new_password: password },
  });
  expect(
    response.ok(),
    `change password for ${username}: ${await response.text()}`,
  ).toBeTruthy();
  return (await response.json()) as Tokens;
}

async function loginInBrowser(
  browser: Browser,
  username: string,
  password: string,
) {
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
  test.setTimeout(60_000);
  test.skip(!ownerBootstrapPassword, "STUDIO_E2E_ADMIN_PASSWORD is required");
  if (evidenceDir) await mkdir(evidenceDir, { recursive: true });

  const initialOwner = await login(request, "unsloth", ownerBootstrapPassword!);
  const owner = initialOwner.must_change_password
    ? await replaceSetupCode(
        request,
        "unsloth",
        ownerBootstrapPassword!,
        ownerPassword,
      )
    : initialOwner;
  const ownerLoginPassword = initialOwner.must_change_password
    ? ownerPassword
    : ownerBootstrapPassword!;

  for (const username of ["alice", "bob"]) {
    const existing = await request.delete(
      `${baseURL}/api/auth/users/${username}`,
      {
        headers: bearer(owner),
      },
    );
    expect([204, 404]).toContain(existing.status());
  }

  const accountCreator = await loginInBrowser(
    browser,
    "unsloth",
    ownerLoginPassword,
  );
  await accountCreator.page
    .getByRole("button", { name: "Settings", exact: true })
    .last()
    .click();
  await accountCreator.page.getByTestId("settings-tab-accounts").click();
  const accountsDialog = accountCreator.page.getByRole("dialog");
  await accountsDialog.getByLabel("Username").fill("alice");
  expect(
    await accountsDialog
      .locator("form")
      .evaluate((form) => form.checkValidity()),
  ).toBe(true);
  const createAliceResponse = accountCreator.page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/auth/users") &&
      response.request().method() === "POST",
  );
  await accountsDialog
    .getByRole("button", { name: "Create account", exact: true })
    .click();
  const aliceResponse = await createAliceResponse;
  expect(aliceResponse.status()).toBe(201);
  const aliceCreated = (await aliceResponse.json()) as CreatedAccount;
  let aliceSetupCode = aliceCreated.setup_code;
  await expect(accountsDialog.getByTestId("setup-code")).toHaveText(
    aliceSetupCode,
  );
  await expect(accountsDialog.getByTestId("managed-account-alice")).toBeVisible();
  await expect(
    accountsDialog.getByText("Awaiting first sign-in", { exact: true }),
  ).toBeVisible();
  if (evidenceDir) {
    await accountsDialog.screenshot({
      path: path.join(evidenceDir, "02-setup-code-shown-once.png"),
    });
  }
  const regenerateAliceResponse = accountCreator.page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/auth/users/alice/setup-code") &&
      response.request().method() === "POST",
  );
  await accountsDialog
    .getByRole("button", { name: "Account actions for alice" })
    .click();
  if (evidenceDir) {
    await accountCreator.page.screenshot({
      path: path.join(evidenceDir, "02a-account-actions-menu.png"),
    });
  }
  await accountCreator.page
    .getByRole("menuitem", { name: "Generate new setup code" })
    .click();
  const regenerateDialog = accountCreator.page.getByRole("alertdialog");
  await expect(regenerateDialog).toContainText("alice");
  if (evidenceDir) {
    await regenerateDialog.screenshot({
      path: path.join(evidenceDir, "02b-regenerate-code-confirmation.png"),
    });
  }
  await regenerateDialog.getByRole("button", { name: "Generate code" }).click();
  const regeneratedAlice = (await (
    await regenerateAliceResponse
  ).json()) as CreatedAccount;
  expect(regeneratedAlice.setup_code).not.toBe(aliceSetupCode);
  const rejectedOldCode = await request.post(`${baseURL}/api/auth/login`, {
    data: { username: "alice", password: aliceSetupCode },
  });
  expect(rejectedOldCode.status()).toBe(401);
  aliceSetupCode = regeneratedAlice.setup_code;
  await expect(accountsDialog.getByTestId("setup-code")).toHaveText(
    aliceSetupCode,
  );
  await accountCreator.context.close();

  const firstLoginContext = await browser.newContext();
  const firstLoginPage = await firstLoginContext.newPage();
  await firstLoginPage.goto(`${baseURL}/login`);
  await firstLoginPage.getByLabel("Username").fill("alice");
  await firstLoginPage
    .getByLabel("Password", { exact: true })
    .fill(aliceSetupCode);
  await firstLoginPage
    .getByRole("button", { name: "Login", exact: true })
    .click();
  await expect(firstLoginPage).toHaveURL(/\/change-password(?:\?|$)/);
  await expect(
    firstLoginPage.getByRole("heading", {
      name: "Setup your account",
      exact: true,
    }),
  ).toBeVisible();
  if (evidenceDir) {
    await firstLoginPage.screenshot({
      path: path.join(evidenceDir, "03-first-login-password-reset.png"),
    });
  }
  await firstLoginContext.close();

  const createdBob = await request.post(`${baseURL}/api/auth/users`, {
    headers: bearer(owner),
    data: { username: "bob" },
  });
  expect(createdBob.status(), `create bob: ${await createdBob.text()}`).toBe(
    201,
  );
  const bobSetupCode = ((await createdBob.json()) as CreatedAccount).setup_code;

  const alice = await replaceSetupCode(
    request,
    "alice",
    aliceSetupCode,
    alicePassword,
  );
  const bob = await replaceSetupCode(request, "bob", bobSetupCode, bobPassword);

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

  const aliceThreadsResponse = await request.get(
    `${baseURL}/api/chat/threads`,
    {
      headers: bearer(alice),
    },
  );
  const bobThreadsResponse = await request.get(`${baseURL}/api/chat/threads`, {
    headers: bearer(bob),
  });
  expect(aliceThreadsResponse.ok()).toBeTruthy();
  expect(bobThreadsResponse.ok()).toBeTruthy();
  const aliceThreads = (await aliceThreadsResponse.json()) as {
    threads: Array<{ title: string }>;
  };
  const bobThreads = (await bobThreadsResponse.json()) as {
    threads: Array<{ title: string }>;
  };
  expect(aliceThreads.threads.map((thread) => thread.title)).toContain(
    aliceThread.title,
  );
  expect(bobThreads.threads).toEqual([]);

  await page.goto(`${baseURL}/login`);
  await expect(page.getByLabel("Username")).toBeVisible();
  await expect(page.getByLabel("Password", { exact: true })).toBeVisible();
  if (evidenceDir) {
    await page.screenshot({
      path: path.join(evidenceDir, "01-login-user-selection.png"),
    });
  }

  const ownerBrowser = await loginInBrowser(
    browser,
    "unsloth",
    ownerLoginPassword,
  );
  await ownerBrowser.page
    .getByRole("button", { name: "Settings", exact: true })
    .last()
    .click();
  await ownerBrowser.page.getByTestId("settings-tab-accounts").click();
  await expect(
    ownerBrowser.page.getByRole("heading", { name: "Accounts", exact: true }),
  ).toBeVisible();
  await expect(
    ownerBrowser.page.getByText("alice", { exact: true }),
  ).toBeVisible();
  await expect(
    ownerBrowser.page.getByText("bob", { exact: true }),
  ).toBeVisible();
  if (evidenceDir) {
    await ownerBrowser.page.getByRole("dialog").screenshot({
      path: path.join(evidenceDir, "04-owner-accounts.png"),
    });
  }

  const aliceBrowser = await loginInBrowser(browser, "alice", alicePassword);
  await expect(
    aliceBrowser.page.getByTestId("current-account-username"),
  ).toHaveText("@alice");
  await expect(
    aliceBrowser.page.getByText(aliceThread.title, { exact: true }),
  ).toBeVisible();
  if (evidenceDir) {
    await aliceBrowser.page.screenshot({
      path: path.join(evidenceDir, "05-alice-workspace.png"),
    });
  }
  await aliceBrowser.context.close();

  const bobBrowser = await loginInBrowser(browser, "bob", bobPassword);
  await expect(
    bobBrowser.page.getByTestId("current-account-username"),
  ).toHaveText("@bob");
  await expect(
    bobBrowser.page.getByText(aliceThread.title, { exact: true }),
  ).toHaveCount(0);
  await expect(
    bobBrowser.page.getByText("No chats yet", { exact: true }),
  ).toBeVisible();
  if (evidenceDir) {
    await bobBrowser.page.screenshot({
      path: path.join(evidenceDir, "06-bob-workspace.png"),
    });
    await writeFile(
      path.join(evidenceDir, "facts.json"),
      `${JSON.stringify(
        {
          baseURL,
          loginHasUsernameField: true,
          managedAccounts: ["unsloth", "alice", "bob"],
          accountSetupUsesNormalPasswordField: true,
          setupCodeShownOnce: true,
          setupCodeExpiryMinutes: 60,
          setupCodeRegenerationInvalidatesPrevious: true,
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

  await ownerBrowser.page.bringToFront();
  const finalAccountsDialog = ownerBrowser.page.getByRole("dialog");
  await finalAccountsDialog
    .getByRole("button", { name: "Account actions for bob" })
    .click();
  await ownerBrowser.page.getByRole("menuitem", { name: "Delete account" }).click();
  const deleteDialog = ownerBrowser.page.getByRole("alertdialog");
  await expect(deleteDialog).toContainText("bob");
  if (evidenceDir) {
    await deleteDialog.screenshot({
      path: path.join(evidenceDir, "07-delete-account-confirmation.png"),
    });
  }
  await deleteDialog.getByRole("button", { name: "Cancel" }).click();
  await expect(deleteDialog).toHaveCount(0);
  await ownerBrowser.context.close();
});
