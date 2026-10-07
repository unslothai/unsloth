// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Chat loads no longer notify, so training must grant itself notification permission.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { register } from "node:module";
import test from "node:test";

// Module-scope caches force a fresh import per case via ?bust=N.
register("./helpers/notification-resolver.mjs", import.meta.url);

// file:// URL: import() rejects Windows paths, and ?bust=N needs a URL.
const MODULE = new URL("../src/lib/native-notifications.ts", import.meta.url).href;

const CHAT_RUNTIME = new URL(
  "../src/features/chat/hooks/use-chat-model-runtime.ts",
  import.meta.url,
);
const TRAINING_LIFECYCLE = new URL(
  "../src/features/training/hooks/use-training-runtime-lifecycle.ts",
  import.meta.url,
);
const TRAINING_ENTRY_POINTS = [
  new URL("../src/features/training/lib/start-fresh-training-run.ts", import.meta.url),
  new URL("../src/features/training/lib/resume-training-run.ts", import.meta.url),
];

const PRIME_CALL = /primeNativeNotificationPermission\(\)/;
const NOTIFY_CALL = /notifyNative\(\{/;

type WebviewPermission = "absent" | "default" | "granted" | "denied";

type StubMode = "ok" | "send-fails" | "module-missing";

type EnvOptions = {
  tauri: boolean;
  webview?: WebviewPermission;
  pluginGranted?: boolean;
  answer?: "granted" | "denied";
  stub?: StubMode;
};

type Control = {
  sent: { title: string; body?: string }[];
  granted: boolean;
  mode: StubMode;
  requests: number;
};

let generation = 0;

function define(name: string, value: unknown) {
  Object.defineProperty(globalThis, name, {
    value,
    configurable: true,
    writable: true,
  });
}

async function load(options: EnvOptions) {
  const {
    tauri,
    webview = "absent",
    pluginGranted = false,
    answer = "granted",
    stub = "ok",
  } = options;

  const webviewRequests = { count: 0 };

  const windowStub: Record<string, unknown> = {
    location: { protocol: tauri ? "tauri:" : "https:" },
  };
  if (tauri) {
    windowStub.__TAURI_INTERNALS__ = {};
  }
  if (webview !== "absent") {
    windowStub.Notification = {
      permission: webview,
      async requestPermission() {
        webviewRequests.count += 1;
        (windowStub.Notification as { permission: string }).permission = answer;
        return answer;
      },
    };
  }
  define("window", windowStub);

  generation += 1;
  const control = ((globalThis as Record<string, unknown>).__TAURI_NOTIFICATION_STUB__ ??=
    {}) as Control;
  control.sent = [];
  control.granted = pluginGranted;
  control.mode = stub;
  control.requests = 0;

  const mod = (await import(`${MODULE}?bust=${generation}`)) as {
    notifyNative: (options: {
      key: string;
      title: string;
      body?: string;
      requestPermission?: boolean;
    }) => Promise<void>;
    primeNativeNotificationPermission: () => Promise<void>;
    sanitizeNotificationBody: (input: string | null, fallback: string) => string;
    safeNotificationLabel: (input: string | null, fallback: string) => string;
  };

  const api = (await import(
    `${new URL("../src/lib/api-base.ts", import.meta.url).href}?bust=${generation}`
  )) as { isTauri: boolean };
  assert.equal(api.isTauri, tauri, "isTauri did not match the staged environment");

  return { ...mod, control, webviewRequests };
}

async function trainingFinished(
  mod: Awaited<ReturnType<typeof load>>,
  jobId = "job-1",
) {
  await mod
    .notifyNative({
      key: `training-completed:${jobId}`,
      title: "Training finished",
      body: "Your training run is complete.",
      requestPermission: false,
    })
    .catch(() => undefined);
}

test("the chat model-load path carries no native-notification dependency", async () => {
  const source = await readFile(CHAT_RUNTIME, "utf8");

  assert.ok(
    !source.includes("native-notifications"),
    "use-chat-model-runtime imports the native notification helper again",
  );
  for (const symbol of [
    "notifyNative",
    "primeNativeNotificationPermission",
    "safeNotificationLabel",
  ]) {
    assert.ok(
      !source.includes(symbol),
      `use-chat-model-runtime calls ${symbol} again; the load toast already reports this`,
    );
  }
});

test("training keeps its own permission prime, which nothing else provides", async () => {
  for (const entry of TRAINING_ENTRY_POINTS) {
    const source = await readFile(entry, "utf8");
    // The call, not the import: an unused import would pass a bare name match.
    assert.match(
      source,
      PRIME_CALL,
      `${entry.href} dropped its prime; training would never obtain permission`,
    );
  }

  const lifecycle = await readFile(TRAINING_LIFECYCLE, "utf8");
  assert.match(
    lifecycle,
    NOTIFY_CALL,
    "the training lifecycle stopped sending native notifications",
  );
});

test("a desktop session that only loads chat models never asks for permission", async () => {
  for (const webview of ["absent", "default", "granted"] as const) {
    const mod = await load({ tauri: true, webview, pluginGranted: true });

    assert.equal(
      mod.webviewRequests.count,
      0,
      `loading a chat model prompted for notification permission [webview=${webview}]`,
    );
    assert.deepEqual(mod.control.sent, [], "a chat model load sent a notification");
  }
});

test("training primes and notifies on a fresh install where chat never ran", async () => {
  const prompted = await load({
    tauri: true,
    webview: "default",
    answer: "granted",
  });
  await prompted.primeNativeNotificationPermission();
  await trainingFinished(prompted);
  assert.equal(prompted.webviewRequests.count, 1, "training did not prompt");
  assert.deepEqual(
    prompted.control.sent.map((n) => n.title),
    ["Training finished"],
  );

  const viaPlugin = await load({
    tauri: true,
    webview: "absent",
    pluginGranted: true,
  });
  await viaPlugin.primeNativeNotificationPermission();
  await trainingFinished(viaPlugin);
  assert.deepEqual(
    viaPlugin.control.sent.map((n) => n.title),
    ["Training finished"],
    "training lost its notification on the plugin permission path",
  );

  const already = await load({ tauri: true, webview: "granted" });
  await already.primeNativeNotificationPermission();
  await trainingFinished(already);
  assert.equal(already.webviewRequests.count, 0, "re-prompted an existing grant");
  assert.deepEqual(
    already.control.sent.map((n) => n.title),
    ["Training finished"],
  );
});

test("a training notification arrives even if the prime is still in flight", async () => {
  const mod = await load({ tauri: true, webview: "default", answer: "granted" });

  // The prime is fire-and-forget, so a quick run must wait for the grant.
  const priming = mod.primeNativeNotificationPermission().catch(() => undefined);
  const finishing = trainingFinished(mod);
  await Promise.all([priming, finishing]);

  assert.deepEqual(
    mod.control.sent.map((n) => n.title),
    ["Training finished"],
    "a terminal event during the prime lost its notification",
  );
});

test("a denied grant sends nothing and does not reject", async () => {
  const mod = await load({ tauri: true, webview: "denied", pluginGranted: true });
  await mod.primeNativeNotificationPermission();
  await assert.doesNotReject(() => trainingFinished(mod));
  assert.deepEqual(mod.control.sent, [], "sent a notification after a denial");
});

// The plugin shim reports granted on Linux/macOS and denied on Windows (plugins-workspace#3512).
test("each desktop platform's shim state behaves the same with and without a chat prime", async () => {
  const platforms = [
    { name: "linux/macOS", webview: "granted" as const, expected: ["Training finished"] },
    { name: "windows", webview: "denied" as const, expected: [] },
  ];

  for (const platform of platforms) {
    const primedByChat = await load({ tauri: true, webview: platform.webview });
    await primedByChat.primeNativeNotificationPermission();
    await trainingFinished(primedByChat);

    const trainingOnly = await load({ tauri: true, webview: platform.webview });
    await trainingOnly.primeNativeNotificationPermission();
    await trainingFinished(trainingOnly);

    assert.deepEqual(
      trainingOnly.control.sent.map((n) => n.title),
      primedByChat.control.sent.map((n) => n.title),
      `${platform.name}: an earlier chat prime changed the training outcome`,
    );
    assert.deepEqual(
      trainingOnly.control.sent.map((n) => n.title),
      platform.expected,
      `${platform.name}: unexpected training notification set`,
    );
  }
});

test("a missing notification plugin degrades quietly", async () => {
  const mod = await load({
    tauri: true,
    webview: "granted",
    stub: "module-missing",
  });
  await assert.doesNotReject(() => mod.primeNativeNotificationPermission());
  await assert.doesNotReject(() => trainingFinished(mod));
});

test("a send that throws never reaches the training caller", async () => {
  const mod = await load({ tauri: true, webview: "granted", stub: "send-fails" });
  await assert.doesNotReject(() => trainingFinished(mod));
});

test("browser and LAN sessions send nothing and never prompt", async () => {
  for (const webview of ["absent", "default", "granted"] as const) {
    const mod = await load({ tauri: false, webview, pluginGranted: true });
    await mod.primeNativeNotificationPermission();
    await trainingFinished(mod);

    assert.deepEqual(mod.control.sent, [], `a browser session notified [${webview}]`);
    assert.equal(
      mod.webviewRequests.count,
      0,
      `a browser session prompted for permission [${webview}]`,
    );
  }
});

test("a repeated notification key is sent once", async () => {
  const mod = await load({ tauri: true, webview: "granted" });
  await trainingFinished(mod, "job-7");
  await trainingFinished(mod, "job-7");

  assert.equal(mod.control.sent.length, 1, "the same job notified twice");
});

test("notification bodies still redact tokens and local paths", async () => {
  const mod = await load({ tauri: true, webview: "granted" });

  await mod
    .notifyNative({
      key: "training-error:job-9",
      title: "Training failed",
      body: "run died at /home/ada/models/run.gguf using hf_abcdefghijklmnopqrstuvwxyz012345",
      requestPermission: false,
    })
    .catch(() => undefined);

  const body = mod.control.sent.at(-1)?.body ?? "";
  assert.ok(!body.includes("/home/ada"), `a local path reached the OS: ${body}`);
  assert.ok(
    !body.includes("hf_abcdefghijklmnopqrstuvwxyz012345"),
    `a token reached the OS: ${body}`,
  );
  assert.ok(body.includes("[path]"), body);
  assert.ok(body.includes("hf_[redacted]"), body);
});
