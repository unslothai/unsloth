// CI-only browser proof through production consent controls, routes and native tools.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import net from "node:net";
import { fileURLToPath, pathToFileURL } from "node:url";
const { chromium } = await import(
  process.env.SRT_PLAYWRIGHT_MODULE
    ? pathToFileURL(process.env.SRT_PLAYWRIGHT_MODULE).href
    : "playwright"
);

const here = path.dirname(fileURLToPath(import.meta.url));
const frontend = path.resolve(here, "../..");
const root = path.resolve(frontend, "../..");
const artifacts = path.resolve(
  process.env.SRT_BROWSER_ARTIFACTS || path.join(root, "srt-browser-artifacts"),
);
await mkdir(artifacts, { recursive: true });
const children = [];
let browser;
let runFailure;
const results = [];
const env = {
  ...process.env,
  SRT_BROWSER_ARTIFACTS: artifacts,
  OPENBLAS_NUM_THREADS: "1",
  OMP_NUM_THREADS: "1",
  MKL_NUM_THREADS: "1",
};
function start(command, args, name) {
  const child = spawn(command, args, {
    cwd: frontend,
    env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  const output = [];
  child.stdout.on("data", (b) => output.push(b));
  child.stderr.on("data", (b) => output.push(b));
  child.on("error", (error) => output.push(Buffer.from(String(error))));
  children.push({ child, name, output });
  return child;
}
async function ready(url) {
  const deadline = Date.now() + 90000;
  while (Date.now() < deadline) {
    if (children.some(({ child }) => child.exitCode !== null))
      throw Error("Fixture service exited before readiness");
    try {
      if ((await fetch(url, { signal: AbortSignal.timeout(1000) })).ok) return;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw Error(`Fixture did not become ready: ${url}`);
}
async function stop(child) {
  if (!child.pid || child.exitCode !== null || child.signalCode !== null)
    return;
  child.kill("SIGTERM");
  const deadline = Date.now() + 10000;
  while (
    child.exitCode === null &&
    child.signalCode === null &&
    Date.now() < deadline
  )
    await new Promise((resolve) => setTimeout(resolve, 50));
  if (child.exitCode === null && child.signalCode === null) {
    child.kill("SIGKILL");
    await new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(Error("service did not exit after SIGKILL")),
        5000,
      );
      child.once("exit", () => {
        clearTimeout(timer);
        resolve();
      });
    });
  }
}
try {
  start(
    process.env.SRT_BROWSER_PYTHON || "python",
    [path.join(here, "api.py")],
    "api",
  );
  start(
    process.execPath,
    [
      path.join(frontend, "node_modules/vite/bin/vite.js"),
      "--host",
      "127.0.0.1",
      "--port",
      "5197",
      "--strictPort",
    ],
    "vite",
  );
  await ready("http://127.0.0.1:5198/fixture/health");
  await ready("http://127.0.0.1:5197/scripts/srt-browser/index.html");
  browser = await chromium.launch({
    channel: process.env.SRT_BROWSER_CHANNEL || undefined,
  });
  const page = await browser.newPage({
    viewport: { width: 1280, height: 900 },
  });
  page.setDefaultTimeout(60000);
  await page.goto("http://127.0.0.1:5197/scripts/srt-browser/index.html");
  async function run(kind, name, mode = "os_isolation_required") {
    await page.evaluate(() => {
      window.lastExecution = null;
    });
    await page
      .getByRole("button", { name: `Run fixed ${kind}`, exact: true })
      .click();
    await page.waitForFunction(() => !!window.lastExecution);
    const result = await page.evaluate(() => window.lastExecution);
    results.push({ name, ...result });
    await page.screenshot({
      path: path.join(artifacts, `${name}.png`),
      fullPage: true,
    });
    assert.ok(
      result.result.includes(`SRT_BROWSER_${kind.toUpperCase()}_EXECUTED`),
      JSON.stringify(result),
    );
    assert.equal(result.records.length, 1);
    assert.equal(result.records[0].effective_mode, mode);
    if (mode === "os_isolation_required") {
      assert.equal(result.records[0].backend, "srt");
      assert.equal(result.records[0].os_isolation, true);
    }
  }
  await run("Python", "required-python");
  await run("Terminal", "required-terminal");
  await page
    .getByRole("button", { name: "Permission level for tool calls" })
    .click();
  const menu = await page.getByRole("menu").innerText();
  assert.match(menu, /Preview limitations apply/);
  assert.doesNotMatch(menu, /Required mode remains unavailable/);
  await page
    .getByRole("menuitem", { name: "Sandbox details", exact: true })
    .click();
  const details = await page.getByRole("dialog").innerText();
  assert.match(details, /Limitations/);
  await writeFile(path.join(artifacts, "limitations.txt"), details);
  await page.screenshot({
    path: path.join(artifacts, "limitations.png"),
    fullPage: true,
  });
  await page.keyboard.press("Escape");
  await page
    .getByRole("button", { name: "Permission level for tool calls" })
    .click();
  await page.getByRole("menuitem", { name: /^Full access/ }).click();
  await page
    .getByRole("button", { name: "Enable Full access", exact: true })
    .click();
  await run("Python", "full-python", "full");
  await page.reload();
  await run("Terminal", "reload-required-terminal");
} catch (error) {
  runFailure = error;
  throw error;
} finally {
  const cleanup = {
    steps: [],
    errors: [],
    runError: runFailure ? String(runFailure) : null,
  };
  async function attempt(name, action) {
    try {
      await action();
      cleanup.steps.push({ name, ok: true });
    } catch (error) {
      cleanup.steps.push({ name, ok: false });
      cleanup.errors.push(`${name}: ${String(error)}`);
    }
  }
  if (browser) await attempt("browser.close", () => browser.close());
  for (const { child, name } of children.reverse())
    await attempt(`stop.${name}`, () => stop(child));
  for (const port of [5197, 5198]) {
    await attempt(
      `port.${port}.closed`,
      () =>
        new Promise((resolve, reject) => {
          const socket = net.connect({ host: "127.0.0.1", port });
          socket.setTimeout(1000);
          socket.once("connect", () => {
            socket.destroy();
            reject(Error("still listening"));
          });
          socket.once("error", (error) => {
            socket.destroy();
            error.code === "ECONNREFUSED" ? resolve() : reject(error);
          });
          socket.once("timeout", () => {
            socket.destroy();
            reject(Error("closure check timed out"));
          });
        }),
    );
  }
  for (const { name, output } of children)
    await attempt(`write.${name}.log`, () =>
      writeFile(path.join(artifacts, `${name}.log`), Buffer.concat(output)),
    );
  await attempt("write.results", () =>
    writeFile(
      path.join(artifacts, "results.json"),
      JSON.stringify(results, null, 2),
    ),
  );
  const paths = [
    "studio/frontend/src/features/chat/permission-mode-select.tsx",
    "studio/frontend/src/features/chat/tool-isolation.ts",
    "studio/frontend/src/features/chat/tool-isolation-labels.ts",
    "studio/frontend/src/features/chat/stores/chat-runtime-store.ts",
    "studio/backend/routes/inference.py",
    "studio/backend/core/inference/tools.py",
    "studio/backend/core/inference/os_sandbox.py",
    "studio/backend/core/inference/srt_adapter.py",
    "studio/backend/core/inference/srt_runtime/bridge.mjs",
  ];
  const hashes = {};
  for (const name of paths)
    await attempt(`hash.${name}`, async () => {
      hashes[name] = createHash("sha256")
        .update(await readFile(path.join(root, name)))
        .digest("hex");
    });
  await attempt("write.hashes", () =>
    writeFile(
      path.join(artifacts, "source-sha256.json"),
      JSON.stringify(hashes, null, 2),
    ),
  );
  await writeFile(
    path.join(artifacts, "cleanup.json"),
    JSON.stringify(cleanup, null, 2),
  );
  if (cleanup.errors.length)
    throw new AggregateError(
      [...(runFailure ? [runFailure] : []), ...cleanup.errors],
      "Browser fixture cleanup failed; see cleanup.json",
    );
}
console.log(
  "SRT_BROWSER_NATIVE_PASSED: Required Python/Terminal, Full confirmation, reload Required; real production routes/tools, no model/GPU claim.",
);
