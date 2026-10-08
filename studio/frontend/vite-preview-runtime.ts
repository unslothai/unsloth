// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The libraries a React preview can import, built from Studio's own locked versions.
 *
 * Each library file is an IIFE that registers module namespaces in `globalThis.__unslothModules`,
 * keyed by import specifier. Every file except `react` reads React back from that registry, so a
 * preview page has exactly one React. Tailwind is `@tailwindcss/browser` copied as a classic script.
 *
 * Nothing in the app imports these files, so the eager set is untouched. The lazy preview loader
 * gets their hashed names from `virtual:preview-runtime-manifest` instead of fetching a manifest:
 * the `/assets` mount caches as immutable, so an unhashed manifest there would go stale.
 */

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { rolldown } from "rolldown";
import type { Plugin } from "vite";

export interface PreviewRuntimeManifest {
  /** File name -> where it is served (relative to the app base, no leading slash), its exact UTF-8 size, and the package version it was built from. */
  files: Record<string, { url: string; bytes: number; version: string }>;
  /** Import specifier -> the file name that registers it. */
  modules: Record<string, string>;
  /** The file name of the Tailwind classic script. It is not importable. */
  tailwind: string;
}

export type PreviewRuntimeFile = {
  name: string;
  /** Path inside the build output, e.g. `assets/preview-runtime/react.0123456789.js`. */
  fileName: string;
  code: string;
};

type PreviewRuntimeSpec = {
  /** File stem and manifest key. */
  name: string;
  /** Package whose version goes into the manifest. */
  versionOf: string;
  /** Registered specifier -> the specifier it is bundled from. */
  modules?: Record<string, string>;
  /** A prebuilt classic script copied as is. */
  classic?: string;
};

export const PREVIEW_RUNTIME_REGISTRY = "__unslothModules";
export const PREVIEW_RUNTIME_DIR = "preview-runtime";
export const MANIFEST_MODULE = "virtual:preview-runtime-manifest";

/** Provided by the `react` file and read back from the registry by every other file. */
const SHARED = [
  "react",
  "react/jsx-runtime",
  "react/jsx-dev-runtime",
  "react-dom",
  "react-dom/client",
];

export const PREVIEW_RUNTIME_SPECS: readonly PreviewRuntimeSpec[] = [
  {
    name: "react",
    versionOf: "react",
    modules: Object.fromEntries(SHARED.map((s) => [s, s])),
  },
  {
    name: "lucide-react",
    versionOf: "lucide-react",
    modules: { "lucide-react": "lucide-react" },
  },
  {
    name: "recharts",
    versionOf: "recharts",
    modules: { recharts: "recharts" },
  },
  {
    name: "motion",
    versionOf: "motion",
    modules: {
      motion: "motion",
      "motion/react": "motion/react",
      "framer-motion": "motion/react",
    },
  },
  {
    name: "tailwind",
    versionOf: "@tailwindcss/browser",
    classic: "@tailwindcss/browser",
  },
];

/**
 * A preview runs under a CSP without 'unsafe-eval', so code that evaluates strings would fail there
 * at runtime. The only ones allowed are the exact global-object fallbacks below, never reached in a
 * browser (`globalThis` exists), at exactly the measured count per file: a new one, or one more or
 * fewer, fails the build so somebody looks at it.
 */
const EVAL_PATTERN = /\beval\s*\(|\bnew\s+Function\b|(?<![\w$.])Function\s*\(/;
/** `Function("return this")`, in whichever quotes the minifier picked (it prints backticks). */
const EVAL_ALLOWED = /(?<![\w$.])Function\((["'`])return this\1\)/g;
const EVAL_ALLOWED_COUNT: Record<string, number> = {
  // es-toolkit's and decimal.js-light's global fallbacks.
  recharts: 2,
};

const ENTRY = "\0preview-runtime-entry";
const SHARED_PREFIX = "\0preview-runtime-shared:";
const REACT_INTERNALS =
  /[\\/]node_modules[\\/](react|react-dom|scheduler)[\\/]/;
/** The last node_modules segment of a module id names the package that owns the file. */
const OWNING_PACKAGE =
  /^(.*[\\/]node_modules[\\/])((?:@[^\\/]+[\\/])?[^\\/]+)[\\/]/;

function readJson<T>(file: string): T {
  return JSON.parse(readFileSync(file, "utf8")) as T;
}

function packageVersion(root: string, name: string): string {
  return readJson<{ version: string }>(
    path.join(root, "node_modules", name, "package.json"),
  ).version;
}

/** One comment line naming every bundled package, in place of the per-module legal comments. */
function licenseBanner(packages: Map<string, string>): string {
  const names = [...packages.entries()]
    .map(([name, rest]) => `${name}@${rest}`)
    .sort();
  return `/*! ${names.join(", ")} */`;
}

function bundledPackages(moduleIds: string[]): Map<string, string> {
  const seen = new Map<string, string>();
  for (const id of moduleIds) {
    const m = OWNING_PACKAGE.exec(id);
    if (!m) {
      continue;
    }
    const name = (m[2] as string).replace(/\\/g, "/");
    if (seen.has(name)) {
      continue;
    }
    const pkg = readJson<{ version: string; license?: string }>(
      path.join(m[1] as string, name, "package.json"),
    );
    seen.set(name, `${pkg.version} (${pkg.license ?? "see package"})`);
  }
  return seen;
}

function entrySource(modules: Record<string, string>): string {
  const imports: string[] = [];
  const registers = [
    `const r = (globalThis.${PREVIEW_RUNTIME_REGISTRY} ??= Object.create(null));`,
  ];
  Object.entries(modules).forEach(([specifier, from], i) => {
    // A namespace import keeps every export. For a CommonJS package (react, react-dom) the
    // namespace is its exports object's keys plus `default` set to that object.
    imports.push(`import * as m${i} from ${JSON.stringify(from)};`);
    registers.push(`r[${JSON.stringify(specifier)}] = m${i};`);
  });
  return [...imports, ...registers].join("\n");
}

/** Throws a build error naming the offending text when a file evaluates strings. */
export function checkNoEval(name: string, code: string): void {
  const allowed = code.match(EVAL_ALLOWED)?.length ?? 0;
  const expected = EVAL_ALLOWED_COUNT[name] ?? 0;
  if (allowed !== expected) {
    throw new Error(
      `preview runtime ${name}: expected ${expected} \`Function("return this")\`, found ${allowed}. Check that a new one cannot run in a browser before changing EVAL_ALLOWED_COUNT.`,
    );
  }
  const rest = code.replace(EVAL_ALLOWED, "");
  const m = EVAL_PATTERN.exec(rest);
  if (m) {
    throw new Error(
      `preview runtime ${name} evaluates code from a string, which a preview's CSP blocks: ` +
        `…${rest.slice(Math.max(0, m.index - 60), m.index + 60)}…`,
    );
  }
}

async function bundleOne(
  root: string,
  spec: PreviewRuntimeSpec,
): Promise<string> {
  if (spec.classic) {
    const file = createRequire(path.join(root, "package.json")).resolve(
      spec.classic,
    );
    const version = packageVersion(root, spec.classic);
    const { license } = readJson<{ license?: string }>(
      path.join(root, "node_modules", spec.classic, "package.json"),
    );
    return `/*! ${spec.classic}@${version} (${license ?? "see package"}) */\n${readFileSync(file, "utf8")}`;
  }
  const source = entrySource(spec.modules ?? {});
  const ownsReact = spec.name === "react";
  const bundle = await rolldown({
    input: ENTRY,
    cwd: root,
    platform: "browser",
    transform: { define: { "process.env.NODE_ENV": '"production"' } },
    plugins: [
      {
        name: "preview-runtime-entry",
        resolveId: (id) => (id === ENTRY ? ENTRY : null),
        load: (id) => (id === ENTRY ? source : null),
      },
      {
        // React comes from the registry as a CommonJS shim, so both ESM imports and the
        // `require("react")` inside CommonJS dependencies (use-sync-external-store) reach it.
        name: "preview-runtime-shared",
        resolveId: (id) =>
          !ownsReact && SHARED.includes(id) ? `${SHARED_PREFIX}${id}` : null,
        load(id) {
          if (!id.startsWith(SHARED_PREFIX)) {
            return null;
          }
          const specifier = id.slice(SHARED_PREFIX.length);
          const missing = `Preview runtime: "${specifier}" is not loaded. Load the react runtime file first.`;
          return [
            `var m = globalThis.${PREVIEW_RUNTIME_REGISTRY} && globalThis.${PREVIEW_RUNTIME_REGISTRY}[${JSON.stringify(specifier)}];`,
            `if (!m) throw new Error(${JSON.stringify(missing)});`,
            "module.exports = m;",
          ].join("\n");
        },
      },
    ],
    onLog(level, log, handler) {
      // Something the runtime cannot resolve would be a broken file, not a warning.
      if (log.code === "UNRESOLVED_IMPORT") {
        throw new Error(log.message);
      }
      handler(level, log);
    },
  });
  try {
    const { output } = await bundle.generate({
      format: "iife",
      minify: true,
      comments: { legal: false },
    });
    const chunk = output[0];
    if (output.length !== 1 || chunk?.type !== "chunk") {
      throw new Error(`preview runtime ${spec.name}: expected one chunk`);
    }
    // A second React in a library file breaks hooks ("Invalid hook call"); fail the build instead.
    const bundledReact = chunk.moduleIds.find((id) => REACT_INTERNALS.test(id));
    if (!ownsReact && bundledReact) {
      throw new Error(`preview runtime ${spec.name} bundles ${bundledReact}`);
    }
    return `${licenseBanner(bundledPackages(chunk.moduleIds))}\n${chunk.code}`;
  } finally {
    await bundle.close();
  }
}

/**
 * Builds every runtime file and the manifest that names them. `assetsDir` is Vite's
 * `build.assetsDir`, so the files sit under the `/assets` mount (gzip, immutable caching, a real 404).
 */
export async function buildPreviewRuntime(
  root: string,
  assetsDir = "assets",
): Promise<{ files: PreviewRuntimeFile[]; manifest: PreviewRuntimeManifest }> {
  const built = await Promise.all(
    PREVIEW_RUNTIME_SPECS.map((spec) => bundleOne(root, spec)),
  );
  const files: PreviewRuntimeFile[] = [];
  const manifest: PreviewRuntimeManifest = {
    files: {},
    modules: {},
    tailwind: "",
  };
  PREVIEW_RUNTIME_SPECS.forEach((spec, i) => {
    const code = built[i] as string;
    checkNoEval(spec.name, code);
    const hash = createHash("sha256").update(code).digest("hex").slice(0, 10);
    const fileName = path.posix.join(
      assetsDir,
      PREVIEW_RUNTIME_DIR,
      `${spec.name}.${hash}.js`,
    );
    files.push({ name: spec.name, fileName, code });
    manifest.files[spec.name] = {
      url: fileName,
      bytes: Buffer.byteLength(code, "utf8"),
      version: packageVersion(root, spec.versionOf),
    };
    if (spec.classic) {
      manifest.tailwind = spec.name;
    }
    for (const specifier of Object.keys(spec.modules ?? {})) {
      manifest.modules[specifier] = spec.name;
    }
  });
  return { files, manifest };
}

const RESOLVED_MANIFEST = `\0${MANIFEST_MODULE}`;
/** The file name a dev request asks for, after the middleware's mount prefix is stripped. */
const REQUESTED_NAME = /^\/?([^?#]*)/;

/**
 * Emits the runtime files into the client build, serves them in dev, and provides
 * `virtual:preview-runtime-manifest`. The libraries are only bundled when something needs them:
 * at build start for a production build, and on first request in dev.
 */
export function previewRuntime(): Plugin {
  let root = process.cwd();
  let assetsDir = "assets";
  let base = "/";
  let isBuild = false;
  let pending: ReturnType<typeof buildPreviewRuntime> | undefined;
  const runtime = () => {
    pending ??= buildPreviewRuntime(root, assetsDir).catch((error: unknown) => {
      pending = undefined;
      throw error;
    });
    return pending;
  };
  return {
    name: "unsloth-preview-runtime",
    applyToEnvironment: (environment) => environment.name === "client",
    configResolved(config) {
      root = config.root;
      assetsDir = config.build.assetsDir;
      base = config.base.startsWith("/") ? config.base : "/";
      isBuild = config.command === "build";
    },
    buildStart() {
      if (isBuild) {
        // Start now so it overlaps the app build; generateBundle awaits it and reports errors.
        runtime().catch(() => undefined);
      }
    },
    resolveId: (id) => (id === MANIFEST_MODULE ? RESOLVED_MANIFEST : null),
    async load(id) {
      if (id !== RESOLVED_MANIFEST) {
        return null;
      }
      const { manifest } = await runtime();
      return `export default ${JSON.stringify(manifest)};`;
    },
    async generateBundle() {
      const { files } = await runtime();
      for (const file of files) {
        this.emitFile({
          type: "asset",
          fileName: file.fileName,
          source: file.code,
        });
      }
    },
    configureServer(server) {
      const prefix = `${base}${assetsDir}/${PREVIEW_RUNTIME_DIR}/`;
      server.middlewares.use(prefix, (req, res, next) => {
        const name = REQUESTED_NAME.exec(req.url ?? "")?.[1] ?? "";
        runtime().then(({ files }) => {
          const file = files.find(
            (f) => f.fileName === `${assetsDir}/${PREVIEW_RUNTIME_DIR}/${name}`,
          );
          if (!file) {
            // Not the SPA fallback: a stale hash has to fail loudly, not load index.html.
            res.statusCode = 404;
            res.end();
            return;
          }
          res.setHeader("Content-Type", "text/javascript; charset=utf-8");
          res.setHeader("Cache-Control", "no-cache");
          res.end(file.code);
        }, next);
      });
    },
  };
}
