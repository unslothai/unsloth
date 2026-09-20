// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const playwright = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const browserName = process.env.PLAYWRIGHT_BROWSER || "chromium";
assert.ok(["chromium", "firefox", "webkit"].includes(browserName));
const base = process.env.SHARE_RUN_BASE_URL || "http://127.0.0.1:5198";
assert.ok(["localhost", "127.0.0.1", "[::1]"].includes(new URL(base).hostname));
const browser = await playwright[browserName].launch({
  headless: true,
  ...(browserName === "chromium" && process.env.PLAYWRIGHT_CHROMIUM_PATH
    ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH }
    : {}),
  ...(browserName === "chromium" ? { args: ["--no-sandbox"] } : {}),
  ...(browserName === "firefox"
    ? { firefoxUserPrefs: { "dom.events.testing.asyncClipboard": true } }
    : {}),
});
const model = "unsloth/Test-GGUF";
const params = (extra = {}) =>
  new URLSearchParams({ model, ggufVariant: "Q4_K_M", ...extra });

async function fixture({
  authenticated = true,
  delay = 100,
  models = [],
  cachedGguf = [],
  cachedModels = [],
  variants = [],
} = {}) {
  const context = await browser.newContext({
    viewport: { width: 1400, height: 1000 },
  });
  const page = await context.newPage();
  page.setDefaultTimeout(15_000);
  const writes = [];
  const errors = [];
  const requests = [];
  const loads = [];
  const validations = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.addInitScript((authenticated) => {
    if (authenticated) localStorage.setItem("unsloth_auth_token", "local-test");
    localStorage.setItem("unsloth_model_configs_migrated", "1");
    localStorage.setItem("unsloth_model_overrides_backfilled_v2", "1");
    localStorage.setItem(
      "unsloth_model_configs",
      JSON.stringify({
        'v2:["unsloth/test-gguf","q4_k_m"]': {
          version: 2,
          nParallel: 8,
          nBatch: 1024,
        },
      }),
    );
  }, authenticated);
  await page.route("**/*", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    requests.push(request.url());
    if (url.origin !== new URL(base).origin) return route.abort();
    if (!url.pathname.startsWith("/api/") && !url.pathname.startsWith("/v1/"))
      return route.continue();
    if (request.method() !== "GET") writes.push(url.pathname);
    let body = {};
    if (url.pathname === "/api/auth/status")
      body = { initialized: true, requires_password_change: false };
    if (url.pathname === "/api/auth/login")
      body = {
        access_token: "local-test",
        refresh_token: "local-refresh",
        must_change_password: false,
      };
    if (url.pathname === "/api/health")
      body = { device_type: "linux", chat_only: false };
    if (url.pathname.includes("llama-flags"))
      body = {
        flags: {},
        managed: [],
        switches: [],
        probe_ok: false,
        max_bytes: 32768,
      };
    if (url.pathname.includes("openai-auto-switch/overrides")) {
      await new Promise((resolve) => setTimeout(resolve, delay));
      body = {
        resolved: {
          n_parallel: 8,
          n_batch: 1024,
          llama_extra_args: ["--metrics"],
        },
      };
    }
    if (url.pathname === "/api/inference/monitor") body = { entries: [] };
    if (url.pathname.includes("/models"))
      body = {
        models,
        loras: [],
        folders: [],
        default_models: [],
        local_models: [],
      };
    if (url.pathname === "/api/hub/local") {
      body = { models: [] };
    }
    if (
      ["/api/hub/cached-gguf", "/api/hub/cached-models"].includes(url.pathname)
    ) {
      body = {
        cached: url.pathname.endsWith("cached-gguf")
          ? cachedGguf
          : cachedModels,
        scan_confirmed: true,
      };
    }
    if (url.pathname === "/api/hub/gguf-variants") {
      body = {
        repo_id: url.searchParams.get("repo_id"),
        variants,
        default_variant: variants[0]?.quant ?? null,
        has_vision: false,
      };
    }
    if (url.pathname === "/api/inference/validate") {
      validations.push(request.postDataJSON());
      body = { valid: true };
    }
    if (url.pathname === "/api/inference/load") {
      loads.push(request.postDataJSON());
      body = {
        status: "success",
        is_gguf: request.postDataJSON().gguf_variant != null,
        context_length: 8192,
      };
    }
    if (url.pathname === "/api/inference/unload") body = { status: "unloaded" };
    if (url.pathname.includes("/threads")) body = { threads: [] };
    if (url.pathname.includes("/projects")) body = { projects: [] };
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });
  return { context, page, writes, errors, requests, loads, validations };
}

async function value(page, label, expected, scope = "body") {
  await page.waitForFunction(
    ({ label, expected, scope }) =>
      document.querySelector(`${scope} [aria-label="${label}"]`)?.value ===
      expected,
    { label, expected, scope },
    { timeout: 10000 },
  );
}

async function nativeLink(page, query) {
  await page.evaluate(async (query) => {
    const { receiveSharedRunConfigUrls } = await import(
      "/src/features/share-run-configs/receive-link.ts"
    );
    receiveSharedRunConfigUrls([`unsloth://run?${query}`]);
  }, String(query));
}

async function waitForSettings(page) {
  await page.waitForFunction(
    async () =>
      (
        await import("/src/features/chat/stores/chat-runtime-store.ts")
      ).useChatRuntimeStore.getState().settingsHydrated,
  );
}

try {
  for (const id of [
    "/models/model-Q4_K_M.gguf",
    "C:\\Models\\model-Q4_K_M.gguf",
    "model-Q4_K_M.gguf",
  ]) {
    const { context, page, errors } = await fixture();
    await page.goto(`${base}/chat`);
    await waitForSettings(page);
    await page.evaluate(async (id) => {
      const { savePerModelConfig, DEFAULT_PER_MODEL_CONFIG } = await import(
        "/src/features/model-picker/model-config/per-model-config.ts"
      );
      savePerModelConfig(id, null, {
        ...DEFAULT_PER_MODEL_CONFIG,
        nParallel: 8,
      });
      const { useChatRuntimeStore: store } = await import(
        "/src/features/chat/stores/chat-runtime-store.ts"
      );
      store.setState({
        params: { ...store.getState().params, checkpoint: id },
        loadedIsGguf: true,
        activeGgufVariant: "Q4_K_M",
        activeLoadId: id,
        activeNativePathToken:
          id.includes("/") || id.includes("\\") ? null : "native-test-token",
      });
    }, id);
    await value(page, "Parallel decode slots", "8");
    await nativeLink(page, "nParallel=3");
    const editor = page.getByRole("dialog", { name: /^Run settings for/ });
    await editor.waitFor();
    await value(
      page,
      "Parallel decode slots",
      "3",
      '[data-slot="popover-content"]',
    );
    const slots = await page
      .getByLabel("Parallel decode slots", { exact: true })
      .evaluateAll((inputs) => inputs.map((input) => input.value));
    assert.deepEqual(slots, ["3", "3"]);
    assert.equal(
      await editor
        .getByRole("checkbox", { name: "Remember for this model" })
        .isChecked(),
      true,
    );
    await editor.getByRole("button", { name: "Share", exact: true }).focus();
    await page
      .getByRole("textbox", { name: "Message input", exact: true })
      .focus();
    await editor.waitFor({ state: "hidden" });
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      `PASS: standalone identity, remembered settings, sidebar draft and focus dismissal: ${id}`,
    );
  }
  {
    const { context, page, errors } = await fixture({
      cachedGguf: [
        { repo_id: model, load_id: "/cache/pinned", size_bytes: 1024 },
      ],
    });
    let release;
    let entered;
    const held = new Promise((resolve) => {
      release = resolve;
    });
    const started = new Promise((resolve) => {
      entered = resolve;
    });
    await page.route("**/api/hub/gguf-variants?**", async (route) => {
      entered();
      await held;
      await route.fulfill({
        json: {
          variants: [
            { quant: "Q4_K_M", filename: "model.gguf", downloaded: true },
          ],
        },
      });
    });
    await page.goto(`${base}/chat#run?${params({ nParallel: "3" })}`);
    await started;
    await page.evaluate(async () => {
      const { router } = await import("/src/app/router.tsx");
      await router.navigate({ to: "/hub" });
    });
    await page.waitForFunction(
      async () =>
        (
          await import("/src/features/share-run-configs/inbox.ts")
        ).runConfigInbox.getSnapshot() === null,
    );
    release();
    assert.equal(new URL(page.url()).pathname, "/hub");
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      "PASS: navigating away cancels a delayed handoff without redirecting back",
    );
  }
  {
    const { context, page, errors } = await fixture();
    const query = String(params({ nParallel: "3" }));
    await page.goto(`${base}/chat#run?${query}`);
    await value(page, "Parallel decode slots", "3");
    const first = new URL(page.url()).searchParams.get("new");
    await page.evaluate(async () => {
      const { router } = await import("/src/app/router.tsx");
      await router.navigate({ to: "/chat", search: {} });
    });
    await page.waitForURL(`${base}/chat`);
    await page.evaluate((query) => {
      window.location.hash = `run?${query}`;
    }, query);
    await page.waitForFunction((first) => {
      const id = new URL(window.location.href).searchParams.get("new");
      return id && id !== first;
    }, first);
    await value(page, "Parallel decode slots", "3");
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      "PASS: the identical browser link can reopen after router navigation to bare chat",
    );
  }
  {
    const { context, page, errors } = await fixture();
    await page.goto(`${base}/hub`);
    await waitForSettings(page);
    await page.evaluate(async () => {
      const { useChatRuntimeStore: store } = await import(
        "/src/features/chat/stores/chat-runtime-store.ts"
      );
      store.setState({
        incognito: true,
        activeProjectId: "previous-project",
        activeThreadId: "previous-thread",
      });
    });
    await nativeLink(page, params({ nParallel: "3" }));
    await value(page, "Parallel decode slots", "3");
    assert.deepEqual(
      await page.evaluate(async () => {
        const state = (
          await import("/src/features/chat/stores/chat-runtime-store.ts")
        ).useChatRuntimeStore.getState();
        return [state.incognito, state.activeProjectId, state.activeThreadId];
      }),
      [false, null, null],
    );
    await page.goBack();
    assert.equal(new URL(page.url()).pathname, "/hub");
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      "PASS: native handoffs reset temporary/project context and preserve the previous history entry",
    );
  }
  for (const previousAccount of [null, "account:old:previous"]) {
    const { context, page, errors } = await fixture({ authenticated: false });
    if (previousAccount)
      await page.addInitScript((marker) => {
        if (!localStorage.getItem("unsloth.browser-account.v1")) {
          localStorage.setItem("unsloth.browser-account.v1", marker);
        }
      }, previousAccount);
    await page.route("**/api/auth/status", (route) =>
      route.fulfill({
        json: {
          initialized: true,
          requires_password_change: false,
          login_mode: "multi",
        },
      }),
    );
    const token = `header.${Buffer.from(JSON.stringify({ sub: "alice", role: "member" })).toString("base64url")}.signature`;
    await page.route("**/api/auth/login", (route) =>
      route.fulfill({
        json: {
          access_token: token,
          refresh_token: "refresh",
          must_change_password: false,
          account_id: "alice-id",
        },
      }),
    );
    await page.goto(`${base}/chat#run?${params({ nParallel: "6" })}`);
    await page.waitForURL("**/login");
    await page.getByLabel("Username", { exact: true }).fill("alice");
    await page
      .getByLabel("Password", { exact: true })
      .fill("local-test-password");
    const navigated = page.waitForEvent("domcontentloaded");
    await page.getByRole("button", { name: "Login", exact: true }).click();
    await navigated;
    await value(page, "Parallel decode slots", "6");
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      `PASS: a link survives ${previousAccount ? "account-switch" : "first managed sign-in"} document replacement`,
    );
  }
  for (const { loadId, isGguf } of [
    {
      loadId: "/secondary/cache/models--unsloth--Test-GGUF/snapshots/pinned",
      isGguf: true,
    },
    {
      loadId: "C:\\Models\\models--unsloth--Test-GGUF\\snapshots\\pinned",
      isGguf: true,
    },
    {
      loadId: "/secondary/cache/models--owner--native/snapshots/pinned",
      isGguf: false,
    },
  ]) {
    const repoId = isGguf ? model : "owner/native";
    for (const active of [false, true]) {
      const { context, page, writes, errors, requests, loads, validations } =
        await fixture({
          [isGguf ? "cachedGguf" : "cachedModels"]: [
            {
              repo_id: repoId,
              load_id: active ? "/other/snapshot" : loadId,
              cache_path: active ? "/other/cache" : loadId,
              size_bytes: 1024,
            },
          ],
          variants: [
            {
              quant: "Q4_K_M",
              filename: "Test-Q4_K_M.gguf",
              size_bytes: 1024,
              downloaded: true,
            },
          ],
        });
      await page.goto(`${base}/chat`);
      await waitForSettings(page);
      if (!active && !isGguf) {
        await page.evaluate(async () => {
          const { fetchInventorySource, useDeviceInventoryStore } =
            await import("/src/features/hub/inventory/use-device-inventory.ts");
          const { getInventoryVersion } = await import(
            "/src/features/hub/stores/inventory-events.ts"
          );
          await fetchInventorySource("cachedModels", {
            inventoryVersion: getInventoryVersion(),
          });
          useDeviceInventoryStore.setState((state) => ({
            cachedModels: {
              ...state.cachedModels,
              rows: [],
              refreshedAt: Date.now() - 60_000,
            },
          }));
        });
      }
      if (active) {
        await page.evaluate(
          async ({ model, loadId, isGguf }) => {
            const { useChatRuntimeStore: store } = await import(
              "/src/features/chat/stores/chat-runtime-store.ts"
            );
            store.setState({
              params: { ...store.getState().params, checkpoint: model },
              loadedIsGguf: isGguf,
              activeGgufVariant: isGguf ? "Q4_K_M" : null,
              activeLoadId: loadId,
            });
          },
          { model: repoId, loadId, isGguf },
        );
      }
      const config = isGguf ? { nParallel: "3" } : { maxSeqLength: "8192" };
      await nativeLink(
        page,
        new URLSearchParams({
          ...(active
            ? {}
            : {
                model: repoId,
                isGguf: String(isGguf),
                ...(isGguf ? { ggufVariant: "Q4_K_M" } : {}),
              }),
          ...config,
        }),
      );
      const editor = page.locator('[data-slot="popover-content"]');
      await value(
        page,
        isGguf ? "Parallel decode slots" : "Max Seq Length",
        isGguf ? "3" : "8192",
        '[data-slot="popover-content"]',
      );
      await editor.getByRole("button", { name: "Share", exact: true }).click();
      const dialog = page.getByRole("dialog", { name: "Share run settings" });
      const shared = await dialog.getByLabel("Shareable link").inputValue();
      const sharedParams = new URLSearchParams(new URL(shared).hash.slice(5));
      assert.equal(sharedParams.get("model"), repoId);
      assert.equal(shared.includes("pinned"), false);
      assert.equal(sharedParams.has("loadId"), false);
      await page.keyboard.press("Escape");
      assert.equal(loads.length, 0);
      assert.ok(
        !writes.some((path) => path.includes("download")),
        writes.join(","),
      );
      await Promise.all([
        page.waitForResponse((response) =>
          response.url().endsWith("/api/inference/load"),
        ),
        editor.getByRole("button", { name: "Load model", exact: true }).click(),
      ]);
      assert.equal(loads.length, 1);
      assert.equal(loads[0].model_path, loadId);
      assert.equal(loads[0].gguf_variant, isGguf ? "Q4_K_M" : null);
      if (isGguf) assert.equal(loads[0].n_parallel, 3);
      else assert.equal(loads[0].max_seq_length, 8192);
      await page.waitForFunction(
        async ({ model, loadId }) => {
          const { useChatRuntimeStore } = await import(
            "/src/features/chat/stores/chat-runtime-store.ts"
          );
          const state = useChatRuntimeStore.getState();
          return (
            state.params.checkpoint === model &&
            state.activeLoadId === loadId &&
            !state.modelLoading
          );
        },
        { model: repoId, loadId },
      );
      assert.ok(validations.length > 0);
      assert.ok(validations.some((request) => request.model_path === loadId));
      const loadValidations = validations.filter(
        (request) => "max_seq_length" in request,
      );
      assert.ok(loadValidations.length > 0);
      assert.ok(
        loadValidations.every((request) => request.model_path === loadId),
      );
      assert.ok(
        !writes.some((path) => path.includes("download")),
        writes.join(","),
      );
      const listings = requests
        .map((url) => new URL(url))
        .filter((url) => url.pathname === "/api/hub/gguf-variants");
      assert.equal(listings.length, active || !isGguf ? 0 : 1);
      for (const url of listings) {
        assert.equal(url.searchParams.get("offline"), "true");
        assert.equal(url.searchParams.get("repo_id"), loadId);
        assert.equal(url.searchParams.get("local_path"), loadId);
      }
      assert.deepEqual(errors, []);
      await context.close();
      console.log(
        `PASS: ${active ? "active" : "cached"} snapshot imports load locally: ${loadId}`,
      );
    }
  }
  for (const newer of ["valid", "invalid"]) {
    const { context, page, writes, errors } = await fixture({
      cachedGguf: [
        { repo_id: model, load_id: "/cache/pinned", size_bytes: 1024 },
      ],
    });
    let release;
    let entered;
    const held = new Promise((resolve) => {
      release = resolve;
    });
    const listingStarted = new Promise((resolve) => {
      entered = resolve;
    });
    await page.route("**/api/hub/gguf-variants?**", async (route) => {
      entered();
      await held;
      await route.fulfill({
        json: {
          variants: [
            { quant: "Q4_K_M", filename: "model.gguf", downloaded: true },
          ],
        },
      });
    });
    await page.goto(`${base}/chat#run?${params({ nParallel: "3" })}`);
    await listingStarted;
    await nativeLink(
      page,
      newer === "valid"
        ? "model=owner/native&isGguf=false&maxSeqLength=8192"
        : "model=owner/native&unknown=true",
    );
    release();
    if (newer === "valid") {
      await value(page, "Max Seq Length", "8192");
    } else {
      await page
        .getByText("Could not open shared run settings", { exact: true })
        .waitFor();
      assert.equal(
        await page.getByRole("button", { name: "Share", exact: true }).count(),
        0,
      );
    }
    assert.ok(
      !writes.some((path) => /\/load$|download/.test(path)),
      writes.join(","),
    );
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      `PASS: a newer ${newer} link supersedes a delayed inventory lookup`,
    );
  }
  {
    const { context, page, writes, errors } = await fixture();
    await page.route("**/api/hub/cached-gguf", (route) =>
      route.fulfill({ status: 503, json: { detail: "Unavailable" } }),
    );
    await page.goto(`${base}/chat#run?${params({ nParallel: "3" })}`);
    await page
      .getByText(
        "Could not check local model availability. Reopen the link to try again.",
        { exact: true },
      )
      .waitFor();
    assert.equal(
      await page.getByRole("button", { name: "Share", exact: true }).count(),
      0,
    );
    assert.ok(
      !writes.some((path) => /\/load$|download/.test(path)),
      writes.join(","),
    );
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      "PASS: an inventory failure cancels the import without staging a download",
    );
  }
  {
    const { context, page, writes, errors } = await fixture({ delay: 400 });
    const args = [
      "--rope-scaling",
      "yarn",
      "--yarn-orig-ctx",
      "32768",
      "--flash-attn",
      "on",
    ];
    await page.goto(
      `${base}/chat#run?${params({ nParallel: "3", llamaExtraArgs: JSON.stringify(args) })}`,
    );
    await value(page, "Parallel decode slots", "3");
    await value(page, "Prompt batch size", "1024");
    const share = page.getByRole("button", { name: "Share", exact: true });
    assert.equal(
      await page
        .getByRole("button", { name: "Load model", exact: true })
        .count(),
      1,
    );
    assert.equal(
      await page.getByRole("button", { name: "Reset", exact: true }).count(),
      1,
    );
    assert.equal(await share.count(), 1);
    assert.deepEqual(
      await share.locator("..").getByRole("button").allTextContents(),
      ["Load model", "Reset", "Share"],
    );
    const draft = await page.evaluate(async () => {
      const { readModelConfigDraft, modelConfigDraftKey } = await import(
        "/src/features/model-picker/model-config/model-config-draft.ts"
      );
      return readModelConfigDraft(
        modelConfigDraftKey("unsloth/Test-GGUF", "Q4_K_M"),
      )?.config;
    });
    assert.deepEqual(draft.llamaExtraArgs, args);
    await page.getByRole("button", { name: "Share", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Share run settings" });
    await dialog.waitFor();
    const link = await dialog.getByLabel("Shareable link").inputValue();
    if (browserName === "chromium") {
      await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    } else if (browserName === "webkit") {
      await page.evaluate(() => {
        let text = "";
        Object.defineProperty(navigator, "clipboard", {
          value: {
            writeText: async (value) => {
              text = value;
            },
            readText: async () => text,
          },
        });
      });
      console.log(
        "NOTE: WebKit's headless system clipboard is unavailable; clipboard calls use a test double.",
      );
    }
    await dialog
      .getByRole("button", { name: "Copy link", exact: true })
      .click();
    assert.equal(
      await page.evaluate(() => navigator.clipboard.readText()),
      link,
    );
    assert.equal(
      new URLSearchParams(new URL(link).hash.slice(5)).get("nParallel"),
      "3",
    );
    await dialog.getByRole("checkbox", { name: /^Parallel slots/ }).uncheck();
    await dialog.getByRole("checkbox", { name: /^Model unsloth/ }).uncheck();
    const partial = new URLSearchParams(
      new URL(
        await dialog.getByLabel("Shareable link").inputValue(),
      ).hash.slice(5),
    );
    assert.equal(partial.has("nParallel"), false);
    assert.equal(partial.has("model"), false);
    assert.deepEqual(JSON.parse(partial.get("llamaExtraArgs")), args);
    await dialog
      .getByRole("combobox", { name: "Open in", exact: true })
      .click();
    await page
      .getByRole("option", { name: "Unsloth desktop app", exact: true })
      .click();
    assert.ok(
      (await dialog.getByLabel("Shareable link").inputValue()).startsWith(
        "unsloth://run?",
      ),
    );
    if (process.env.SHARE_RUN_SCREENSHOT)
      await page.screenshot({ path: process.env.SHARE_RUN_SCREENSHOT });
    await page.keyboard.press("Escape");
    await nativeLink(page, params({ nParallel: "5", llamaExtraArgs: "[]" }));
    await value(page, "Parallel decode slots", "5");
    await value(page, "Extra llama-server arguments", "");
    await page.getByLabel("Parallel decode slots").fill("7");
    await nativeLink(page, params({ nParallel: "5", llamaExtraArgs: "[]" }));
    await value(page, "Parallel decode slots", "7");
    await page.waitForTimeout(200);
    assert.ok(
      !writes.some((path) =>
        /\/load$|download|openai-auto-switch\/overrides/.test(path),
      ),
      writes.join(","),
    );
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      "PASS: editor, delayed defaults, exact argv, sharing, omitted fields, native intents and deduplication",
    );
  }
  {
    const { context, page, errors } = await fixture();
    await page.goto(`${base}/chat#run?nParallel=2`);
    const dialog = page.getByRole("dialog", { name: "Choose a model" });
    await dialog.getByLabel("Hugging Face model ID").fill(model);
    await dialog.getByRole("button", { name: "Open run settings" }).click();
    await value(page, "Parallel decode slots", "2");
    await value(page, "Extra llama-server arguments", "--metrics");
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      "PASS: missing model prompts for a model and omitted arguments retain saved defaults",
    );
  }
  {
    const { context, page, errors } = await fixture({ authenticated: false });
    await page.goto(`${base}/chat#run?${params({ nParallel: "6" })}`);
    await page.waitForURL("**/login");
    await page
      .getByLabel("Password", { exact: true })
      .fill("local-test-password");
    await page.getByRole("button", { name: "Login", exact: true }).click();
    await value(page, "Parallel decode slots", "6");
    assert.deepEqual(errors, []);
    await context.close();
    console.log("PASS: a cold link survives login");
  }
  {
    const { context, page, writes, errors, requests } = await fixture();
    const message =
      '</textarea><img data-share-attack src="/share-attack" onerror="globalThis.shareAttack=true"><script>globalThis.shareAttack=true</script>';
    await page.goto(
      `${base}/chat#run?${params({ nParallel: "3", llamaExtraArgs: "[]", reasoningBudgetMessage: JSON.stringify(message) })}`,
    );
    await value(page, "Parallel decode slots", "3");
    await value(page, "Reasoning Budget Message", message);
    await page.getByRole("button", { name: "Share", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Share run settings" });
    const preview = new URL(
      await dialog.getByLabel("Shareable link").inputValue(),
    );
    assert.equal(
      JSON.parse(
        new URLSearchParams(preview.hash.slice(5)).get(
          "reasoningBudgetMessage",
        ),
      ),
      message,
    );
    assert.equal(await page.locator("[data-share-attack]").count(), 0);
    assert.equal(await page.evaluate(() => globalThis.shareAttack), undefined);
    await page.keyboard.press("Escape");
    for (const hostile of [
      'llamaExtraArgs=["--agent"]',
      `llamaExtraArgs=${encodeURIComponent('["\\u002d\\u002dmcp-servers-json","{}"]')}`,
      `llamaExtraArgs=${encodeURIComponent(encodeURIComponent('["--rpc","evil:5000"]'))}`,
      new URLSearchParams({ ggufVariant: "C:/model.gguf" }).toString(),
      new URLSearchParams({
        chatTemplateOverride: "{{ cycler.__init__.__globals__ }}",
      }).toString(),
      "%5f%5fproto%5f%5f=true",
      "%6eParallel=9",
    ]) {
      await nativeLink(page, `model=${model}&nParallel=6&${hostile}`);
      await value(page, "Parallel decode slots", "3");
      assert.equal(
        await page.evaluate(async () =>
          (
            await import("/src/features/share-run-configs/inbox.ts")
          ).runConfigInbox.getSnapshot(),
        ),
        null,
      );
    }
    await page.evaluate(async () => {
      const { patchModelConfigDraft, modelConfigDraftKey } = await import(
        "/src/features/model-picker/model-config/model-config-draft.ts"
      );
      patchModelConfigDraft(
        modelConfigDraftKey("unsloth/Test-GGUF", "Q4_K_M"),
        (current) => ({
          ...current,
          llamaExtraArgs: ["--lora", "C:\\private\\adapter.gguf"],
          chatTemplateOverride: "{{ messages[0]['content'] }}",
        }),
      );
    });
    await page.getByRole("button", { name: "Share", exact: true }).click();
    await dialog.waitFor();
    assert.equal(
      await dialog
        .getByRole("checkbox", { name: /^Extra arguments/ })
        .isDisabled(),
      true,
    );
    assert.equal(
      await dialog
        .getByRole("checkbox", { name: /^Chat template/ })
        .isDisabled(),
      true,
    );
    const excluded = new URLSearchParams(
      new URL(
        await dialog.getByLabel("Shareable link").inputValue(),
      ).hash.slice(5),
    );
    assert.equal(excluded.has("llamaExtraArgs"), false);
    assert.equal(excluded.has("chatTemplateOverride"), false);
    assert.equal(
      requests.some((url) => url.includes("/share-attack")),
      false,
    );
    assert.ok(
      writes.every((path) =>
        [
          "/api/inference/validate",
          "/api/inference/estimate-memory",
          "/api/settings/chat-preferences/migrate",
        ].includes(path),
      ),
      writes.join(","),
    );
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      "PASS: inert HTML, encoded attack rejection, unchanged drafts, restricted sharing and no side effects",
    );
  }
  {
    const { context, page, writes, errors } = await fixture();
    let releaseRead;
    const heldRead = new Promise((resolve) => {
      releaseRead = resolve;
    });
    await page.route(
      "**/api/settings/openai-auto-switch/overrides?**",
      async (route) => {
        await heldRead;
        await route.fallback();
      },
    );
    await page.goto(`${base}/chat#run?${params({ nParallel: "6" })}`);
    await page.getByRole("button", { name: "Share", exact: true }).waitFor();
    await nativeLink(
      page,
      params({ nParallel: "9", llamaExtraArgs: '["--agent"]' }),
    );
    await page.waitForFunction(
      async () =>
        (
          await import("/src/features/share-run-configs/inbox.ts")
        ).runConfigInbox.getSnapshot() === null,
    );
    releaseRead();
    await value(page, "Parallel decode slots", "8");
    await page.waitForFunction(() =>
      [...document.querySelectorAll("button")].some(
        (button) => button.textContent === "Share" && !button.disabled,
      ),
    );
    assert.ok(
      writes.every((path) =>
        [
          "/api/inference/validate",
          "/api/inference/estimate-memory",
          "/api/settings/chat-preferences/migrate",
        ].includes(path),
      ),
      writes.join(","),
    );
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      "PASS: a rejected newer link cancels a pending import during hydration",
    );
  }
  {
    const { context, page, writes, errors } = await fixture();
    await page.goto(
      `${base}/chat#run?${params({ nParallel: "6", llamaExtraArgs: '["--agent"]' })}`,
    );
    await page
      .getByText("Could not open shared run settings", { exact: true })
      .first()
      .waitFor();
    assert.equal(
      await page.getByRole("button", { name: "Share", exact: true }).count(),
      0,
    );
    assert.equal(
      await page.evaluate(async () =>
        (
          await import("/src/features/share-run-configs/inbox.ts")
        ).runConfigInbox.getSnapshot(),
      ),
      null,
    );
    assert.ok(
      writes.every((path) => path === "/api/settings/chat-preferences/migrate"),
      writes.join(","),
    );
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      "PASS: a rejected browser startup link opens no editor and performs no configuration writes",
    );
  }
  for (const intent of ["valid", "invalid", "hub"]) {
    const { context, page, errors } = await fixture();
    let release;
    const credentials = new Promise((resolve) => {
      release = resolve;
    });
    await page.route("**/api/settings/hugging-face-token", async (route) => {
      await credentials;
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ token: null, has_token: false }),
      });
    });
    await page.goto(`${base}/chat#run?${params({ nParallel: "3" })}`);
    await page.evaluate(async () => {
      await import("/src/features/share-run-configs/link-handler.tsx");
    });
    if (intent === "hub") {
      assert.equal(
        await page.evaluate(async () =>
          (
            await import("/src/features/share-run-configs/receive-link.ts")
          ).receiveSharedRunConfigUrls([
            "unsloth://open_from_hf?model=owner/other",
          ]),
        ),
        false,
      );
    } else {
      await nativeLink(
        page,
        params({
          nParallel: "7",
          ...(intent === "invalid" ? { llamaExtraArgs: '["--agent"]' } : {}),
        }),
      );
    }
    release();
    await waitForSettings(page);
    if (intent === "valid") {
      await value(page, "Parallel decode slots", "7");
    } else {
      await page.waitForTimeout(300);
      assert.equal(
        await page.getByRole("button", { name: "Share", exact: true }).count(),
        0,
      );
    }
    assert.deepEqual(errors, []);
    await context.close();
  }
  console.log(
    "PASS: newer native, rejected and Hub intents supersede a startup link during credential loading",
  );
  {
    const { context, page, errors } = await fixture();
    await page.goto(`${base}/chat#run?${params()}`);
    await value(page, "Parallel decode slots", "8");
    assert.equal(
      await page.evaluate(async () => {
        const { isModelConfigDraftEdited, modelConfigDraftKey } = await import(
          "/src/features/model-picker/model-config/model-config-draft.ts"
        );
        return isModelConfigDraftEdited(
          modelConfigDraftKey("unsloth/Test-GGUF", "Q4_K_M"),
        );
      }),
      false,
    );
    await page.evaluate(
      (query) => {
        window.location.hash = `run?${query}`;
      },
      String(params({ nParallel: "4" })),
    );
    await value(page, "Parallel decode slots", "4");
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      "PASS: model-only links preserve unedited defaults and browser hash changes apply once",
    );
  }
  {
    const { context, page, errors } = await fixture({
      models: [{ id: "owner/weights", name: "Weights", is_gguf: true }],
    });
    await page.goto(`${base}/chat`);
    await waitForSettings(page);
    await page.evaluate(async () => {
      const { useChatRuntimeStore: store } = await import(
        "/src/features/chat/stores/chat-runtime-store.ts"
      );
      store.setState({
        params: { ...store.getState().params, checkpoint: "unsloth/Test-GGUF" },
        loadedIsGguf: true,
        activeGgufVariant: "Q4_K_M",
      });
    });
    await nativeLink(page, "isGguf=false&maxSeqLength=8192");
    await page.waitForFunction(async () => {
      const { readModelConfigDraft, modelConfigDraftKey } = await import(
        "/src/features/model-picker/model-config/model-config-draft.ts"
      );
      return (
        readModelConfigDraft(modelConfigDraftKey("unsloth/Test-GGUF", null))
          ?.config.maxSeqLength === 8192
      );
    });
    await page.keyboard.press("Escape");
    await page.waitForFunction(async () => {
      const { useChatRuntimeStore: store } = await import(
        "/src/features/chat/stores/chat-runtime-store.ts"
      );
      return store
        .getState()
        .models.some((model) => model.id === "owner/weights");
    });
    await nativeLink(page, "model=owner/weights&nParallel=5");
    await value(
      page,
      "Parallel decode slots",
      "5",
      '[data-slot="popover-content"]',
    );
    await value(page, "Parallel decode slots", "8");
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      "PASS: explicit native formats clear GGUF identity and omitted formats use known model metadata",
    );
  }
  {
    const { context, page, errors } = await fixture();
    await page.goto(`${base}/chat?new=ordinary-editor`);
    await waitForSettings(page);
    await page.waitForFunction(
      () =>
        document.activeElement?.getAttribute("aria-label") === "Message input",
    );
    await page
      .getByRole("button", { name: "Select model", exact: true })
      .click();
    await page.evaluate(async () => {
      const { requestModelConfigHandoff } = await import(
        "/src/features/model-picker/model-config/model-config-handoff.ts"
      );
      requestModelConfigHandoff({
        requestId: "ordinary-editor",
        id: "unsloth/Test-GGUF",
        meta: {
          source: "hub",
          isLora: false,
          isGguf: true,
          ggufVariant: "Q4_K_M",
        },
      });
    });
    await page.getByRole("button", { name: "Share", exact: true }).click();
    await page.getByRole("dialog", { name: "Share run settings" }).waitFor();
    await page.keyboard.press("Escape");
    await page
      .getByRole("dialog", { name: "Share run settings" })
      .waitFor({ state: "hidden" });
    await page.getByRole("button", { name: "Share", exact: true }).waitFor();
    await page.getByRole("switch", { name: "Show advanced settings" }).check();
    await page.getByLabel("Parallel decode slots", { exact: true }).fill("6");
    await page
      .getByLabel("Parallel decode slots", { exact: true })
      .press("Tab");
    await value(page, "Parallel decode slots", "6");
    await page
      .getByRole("checkbox", { name: "Remember for this model" })
      .check();
    await page.getByRole("button", { name: "Reset", exact: true }).click();
    const ordinaryDraft = await page.evaluate(async () => {
      const { readModelConfigDraft, modelConfigDraftKey } = await import(
        "/src/features/model-picker/model-config/model-config-draft.ts"
      );
      return readModelConfigDraft(
        modelConfigDraftKey("unsloth/Test-GGUF", "Q4_K_M"),
      );
    });
    assert.equal(ordinaryDraft.config.nParallel, null);
    assert.equal(ordinaryDraft.config.llamaExtraArgs, null);
    assert.equal(ordinaryDraft.remember, true);
    assert.equal(
      await page
        .getByText("Run settings import cancelled", { exact: true })
        .count(),
      0,
    );
    await page.getByRole("button", { name: "Share", exact: true }).focus();
    await page
      .getByRole("textbox", { name: "Message input", exact: true })
      .focus();
    await page
      .getByRole("button", { name: "Share", exact: true })
      .waitFor({ state: "hidden" });
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      "PASS: ordinary editing, Remember, Reset and focus dismissal remain unchanged around the Share dialog",
    );
  }
  {
    const { context, page, errors } = await fixture({ delay: 2000 });
    await page.goto(`${base}/chat#run?${params({ nParallel: "6" })}`);
    await page.getByRole("button", { name: "Share", exact: true }).waitFor();
    await page.keyboard.press("Escape");
    await page.waitForFunction(
      async () =>
        (
          await import("/src/features/share-run-configs/inbox.ts")
        ).runConfigInbox.getSnapshot() === null,
    );
    await page.waitForTimeout(2100);
    assert.equal(
      await page.getByRole("button", { name: "Share", exact: true }).count(),
      0,
    );
    assert.deepEqual(errors, []);
    await context.close();
    console.log("PASS: closing the editor cancels a delayed import");
  }
  {
    const { context, page, errors } = await fixture();
    await page.goto(`${base}/chat#run?${params({ nParallel: "3" })}`);
    await value(page, "Parallel decode slots", "3");
    for (const interruptedBy of ["valid", "invalid"]) {
      const result = await page.evaluate(
        async ({ query, interruptedBy }) => {
          const { receiveRunConfigUrl, receiveSharedRunConfigUrls } =
            await import("/src/features/share-run-configs/receive-link.ts");
          const { runConfigInbox } = await import(
            "/src/features/share-run-configs/inbox.ts"
          );
          receiveRunConfigUrl("http://localhost/chat#run?nParallel=5");
          const native = `unsloth://run?${query}`;
          receiveSharedRunConfigUrls([native]);
          const first = runConfigInbox.getSnapshot()?.id;
          receiveSharedRunConfigUrls([native]);
          const duplicate = runConfigInbox.getSnapshot()?.id;
          receiveRunConfigUrl(
            interruptedBy === "valid"
              ? "http://localhost/chat#run?nParallel=5"
              : "http://localhost/chat#run?llamaExtraArgs=[%22--agent%22]",
          );
          receiveSharedRunConfigUrls([native]);
          return { first, duplicate, last: runConfigInbox.getSnapshot() };
        },
        { query: String(params({ nParallel: "4" })), interruptedBy },
      );
      assert.equal(result.duplicate, result.first);
      assert.notEqual(result.last.id, result.first);
      assert.equal(result.last.value.config.nParallel, 4);
      await value(page, "Parallel decode slots", "4");
    }
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      "PASS: browser links retire native deduplication without accepting consecutive duplicates",
    );
  }
  for (const unavailable of ["timeout", "failure"]) {
    const { context, page, errors } = await fixture();
    await page.clock.install();
    let releaseRead;
    const heldRead = new Promise((resolve) => {
      releaseRead = resolve;
    });
    let fail = unavailable === "failure";
    await page.route(
      "**/api/settings/openai-auto-switch/overrides?**",
      async (route) => {
        if (unavailable === "timeout") {
          await heldRead;
        }
        await route.fulfill({
          status: fail ? 503 : 200,
          contentType: "application/json",
          body: JSON.stringify(
            fail
              ? {}
              : {
                  resolved: {
                    n_parallel: 8,
                    n_batch: 1024,
                    llama_extra_args: ["--metrics"],
                  },
                },
          ),
        });
      },
    );
    await Promise.all([
      page.waitForRequest("**/api/settings/openai-auto-switch/overrides?**"),
      page.goto(`${base}/chat#run?${params({ nParallel: "3" })}`),
    ]);
    await page.getByRole("button", { name: "Share", exact: true }).waitFor();
    if (unavailable === "timeout") {
      await page.clock.fastForward(15_001);
    }
    await page
      .getByText("Could not import run settings", { exact: true })
      .waitFor();
    const untouched = await page.evaluate(async () => {
      const {
        isModelConfigDraftEdited,
        readModelConfigDraft,
        modelConfigDraftKey,
      } = await import(
        "/src/features/model-picker/model-config/model-config-draft.ts"
      );
      const key = modelConfigDraftKey("unsloth/Test-GGUF", "Q4_K_M");
      const { runConfigInbox } = await import(
        "/src/features/share-run-configs/inbox.ts"
      );
      return {
        config: readModelConfigDraft(key)?.config,
        edited: isModelConfigDraftEdited(key),
        pending: runConfigInbox.getSnapshot(),
      };
    });
    assert.notEqual(untouched.config.nParallel, 3);
    assert.equal(untouched.edited, false);
    assert.equal(untouched.pending, null);
    fail = false;
    releaseRead();
    if (unavailable === "timeout") {
      await value(page, "Parallel decode slots", "8");
      await value(page, "Prompt batch size", "1024");
    }
    await page.keyboard.press("Escape");
    await page
      .getByRole("button", { name: "Share", exact: true })
      .waitFor({ state: "hidden" });
    await nativeLink(page, params({ nParallel: "3" }));
    await value(page, "Parallel decode slots", "3");
    await value(page, "Prompt batch size", "1024");
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      `PASS: saved-settings ${unavailable} leaves drafts untouched and permits retry after recovery`,
    );
  }
  for (const scenario of ["recovery", "cancellation"]) {
    const { context, page, errors } = await fixture();
    let fail = scenario === "recovery";
    let holding = false;
    let releaseRead;
    const heldRead = new Promise((resolve) => {
      releaseRead = resolve;
    });
    await page.route(
      "**/api/settings/openai-auto-switch/overrides?**",
      async (route) => {
        if (holding) {
          await heldRead;
        }
        await route.fulfill({
          status: fail ? 503 : 200,
          contentType: "application/json",
          body: JSON.stringify(
            fail
              ? {}
              : {
                  resolved: {
                    n_parallel: 8,
                    n_batch: 1024,
                    llama_extra_args: ["--metrics"],
                  },
                },
          ),
        });
      },
    );
    await page.goto(`${base}/chat`);
    await waitForSettings(page);
    await page.evaluate(async () => {
      const { useChatRuntimeStore: store } = await import(
        "/src/features/chat/stores/chat-runtime-store.ts"
      );
      store.setState({
        params: { ...store.getState().params, checkpoint: "unsloth/Test-GGUF" },
        loadedIsGguf: true,
        activeGgufVariant: "Q4_K_M",
      });
    });
    await page.waitForFunction(() =>
      [...document.querySelectorAll("button")].some(
        (button) => button.textContent === "Share" && !button.disabled,
      ),
    );
    fail = false;
    holding = true;
    await Promise.all([
      page.waitForRequest("**/api/settings/openai-auto-switch/overrides?**"),
      nativeLink(page, params({ nParallel: "3" })),
    ]);
    const editor = page.getByRole("dialog", { name: /^Run settings for/ });
    await editor.waitFor();
    if (scenario === "cancellation") {
      await page.keyboard.press("Escape");
      await editor.waitFor({ state: "hidden" });
      await page.waitForFunction(
        async () =>
          (
            await import("/src/features/share-run-configs/inbox.ts")
          ).runConfigInbox.getSnapshot() === null,
      );
    }
    releaseRead();
    await value(
      page,
      "Parallel decode slots",
      scenario === "recovery" ? "3" : "8",
    );
    await value(page, "Prompt batch size", "1024");
    assert.equal(
      await page
        .getByText("Could not import run settings", { exact: true })
        .count(),
      0,
    );
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      `PASS: import ${scenario} is owned by the opened editor while the resident sidebar stays mounted`,
    );
  }
  for (const edit of ["context", "slots", "arguments", "remember", "reset"]) {
    const { context, page, errors } = await fixture();
    let releaseRead;
    const heldRead = new Promise((resolve) => {
      releaseRead = resolve;
    });
    await page.route(
      "**/api/settings/openai-auto-switch/overrides?**",
      async (route) => {
        await heldRead;
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ resolved: { n_parallel: 8, n_batch: 1024 } }),
        });
      },
    );
    await page.goto(`${base}/chat`);
    await waitForSettings(page);
    await page.evaluate(async () => {
      const { savePerModelConfig, DEFAULT_PER_MODEL_CONFIG } = await import(
        "/src/features/model-picker/model-config/per-model-config.ts"
      );
      savePerModelConfig("unsloth/Test-GGUF", "Q4_K_M", {
        ...DEFAULT_PER_MODEL_CONFIG,
        nParallel: 7,
      });
    });
    await nativeLink(
      page,
      params({ nParallel: "3", llamaExtraArgs: '["--threads","4"]' }),
    );
    await page.getByRole("button", { name: "Share", exact: true }).waitFor();
    await page.getByRole("switch", { name: "Show advanced settings" }).check();
    if (edit === "context") {
      await page.getByLabel("Context Length", { exact: true }).focus();
      await page.waitForFunction(() => {
        const input = document.querySelector('[aria-label="Context Length"]');
        return (
          input.selectionStart === 0 &&
          input.selectionEnd === input.value.length
        );
      });
      await page.getByLabel("Context Length", { exact: true }).fill("4096");
    } else if (edit === "slots") {
      await page.getByLabel("Parallel decode slots", { exact: true }).fill("6");
      await page
        .getByLabel("Parallel decode slots", { exact: true })
        .press("Tab");
    } else if (edit === "arguments") {
      await page
        .getByLabel("Extra llama-server arguments", { exact: true })
        .fill("--threads 2");
    } else if (edit === "remember") {
      await page
        .getByRole("checkbox", { name: "Remember for this model" })
        .uncheck();
    } else {
      await page.getByRole("button", { name: "Reset", exact: true }).click();
    }
    await page
      .getByText("Run settings import cancelled", { exact: true })
      .waitFor();
    const response = page.waitForResponse(
      "**/api/settings/openai-auto-switch/overrides?**",
    );
    releaseRead();
    await response;
    await page.waitForFunction(() =>
      [...document.querySelectorAll("button")].some(
        (button) => button.textContent === "Share" && !button.disabled,
      ),
    );
    const draft = await page.evaluate(async () => {
      const { readModelConfigDraft, modelConfigDraftKey } = await import(
        "/src/features/model-picker/model-config/model-config-draft.ts"
      );
      return readModelConfigDraft(
        modelConfigDraftKey("unsloth/Test-GGUF", "Q4_K_M"),
      );
    });
    assert.equal(
      draft.config.nParallel,
      edit === "slots" ? 6 : edit === "reset" ? null : 7,
    );
    assert.deepEqual(
      draft.config.llamaExtraArgs,
      edit === "arguments"
        ? ["--threads", "2"]
        : edit === "reset"
          ? null
          : undefined,
    );
    if (edit === "remember") assert.equal(draft.remember, false);
    if (edit === "context") {
      await value(page, "Context Length", "4096");
      await page.getByLabel("Context Length", { exact: true }).press("Tab");
      await page.waitForFunction(async () => {
        const { readModelConfigDraft, modelConfigDraftKey } = await import(
          "/src/features/model-picker/model-config/model-config-draft.ts"
        );
        return (
          readModelConfigDraft(
            modelConfigDraftKey("unsloth/Test-GGUF", "Q4_K_M"),
          )?.config.customContextLength === 4096
        );
      });
    }
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      `PASS: newer ${edit} edits cancel a delayed import without losing user changes`,
    );
  }
  for (const key of ["maxSeqLength", "customContextLength"]) {
    const other =
      key === "maxSeqLength" ? "customContextLength" : "maxSeqLength";
    const { context, page, errors } = await fixture();
    await page.goto(`${base}/chat`);
    await waitForSettings(page);
    await page.evaluate(async (other) => {
      const { savePerModelConfig, DEFAULT_PER_MODEL_CONFIG } = await import(
        "/src/features/model-picker/model-config/per-model-config.ts"
      );
      savePerModelConfig("owner/native", null, {
        ...DEFAULT_PER_MODEL_CONFIG,
        [other]: 4096,
        nParallel: 8,
      });
    }, other);
    await nativeLink(page, `model=owner/native&isGguf=false&${key}=8192`);
    await value(page, "Max Seq Length", "8192");
    await page.getByRole("button", { name: "Share", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Share run settings" });
    const link = await dialog.getByLabel("Shareable link").inputValue();
    const config = new URLSearchParams(new URL(link).hash.slice(5));
    assert.equal(config.get(key), "8192");
    assert.equal(config.get(other), null);
    assert.equal(config.get("nParallel"), "8");
    assert.deepEqual(errors, []);
    await context.close();
    console.log(
      `PASS: ${key} replaces an older context pin and shares the displayed value`,
    );
  }
} finally {
  await browser.close();
}
