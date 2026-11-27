// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { ProjectsHomeSettings } from "../src/features/settings/api/projects-home.ts";
import {
  SETTINGS_SEARCH_KEYWORDS,
  createSettingsSearchIndex,
  renderedSearchEntries,
} from "../src/features/settings/settings-search.ts";
import { en } from "../src/i18n/locales/en.ts";
import { readFastApiError } from "../src/lib/format-fastapi-error.ts";
import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

const DEFAULT: ProjectsHomeSettings = {
  projectsHome: "/home/owner/Projects",
  source: "default",
  editable: true,
  isCustom: false,
  environmentVariable: null,
};
const CUSTOM: ProjectsHomeSettings = {
  ...DEFAULT,
  projectsHome: "/Volumes/Projects",
  source: "studio",
  isCustom: true,
};
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status });
const apiBody = (value: ProjectsHomeSettings) => ({
  projects_home: value.projectsHome,
  source: value.source,
  editable: value.editable,
  is_custom: value.isCustom,
  environment_variable: value.environmentVariable,
});

function api(respond: (init?: RequestInit) => Response) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const module = loadWithStubs<
    typeof import("../src/features/settings/api/projects-home.ts")
  >(new URL("../src/features/settings/api/projects-home.ts", import.meta.url), {
    "@/features/auth": {
      authFetch: async (url: string, init?: RequestInit) => {
        calls.push({ url, init });
        return respond(init);
      },
    },
    "@/lib/format-fastapi-error": { readFastApiError },
  });
  return { ...module, calls };
}

test("projects home settings preserve effective path and environment ownership", async () => {
  const expected: ProjectsHomeSettings = {
    ...CUSTOM,
    source: "environment",
    editable: false,
    isCustom: false,
    environmentVariable: "UNSLOTH_STUDIO_PROJECTS_HOME",
  };
  const client = api(() => json(apiBody(expected)));
  assert.deepEqual(
    await client.loadProjectsHomeSettings("load failed"),
    expected,
  );
  assert.deepEqual(client.calls, [
    { url: "/api/settings/projects-home", init: undefined },
  ]);
});

test("choosing a folder and resetting send only the projects_home setting", async () => {
  const client = api((init) => {
    const path = JSON.parse(String(init?.body)).projects_home;
    return json(apiBody(path === null ? DEFAULT : CUSTOM));
  });
  assert.deepEqual(
    await client.updateProjectsHomeSettings(CUSTOM.projectsHome, "failed"),
    CUSTOM,
  );
  assert.deepEqual(
    await client.updateProjectsHomeSettings(null, "failed"),
    DEFAULT,
  );
  assert.deepEqual(
    client.calls.map(({ url, init }) => ({ url, ...init })),
    [
      {
        url: "/api/settings/projects-home",
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ projects_home: CUSTOM.projectsHome }),
      },
      {
        url: "/api/settings/projects-home",
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ projects_home: null }),
      },
    ],
  );
});

test("settings errors preserve backend validation and translated fallbacks", async () => {
  for (const status of [400, 403, 409]) {
    const client = api(() =>
      json({ detail: "Choose an existing writable folder" }, status),
    );
    await assert.rejects(
      client.updateProjectsHomeSettings("/missing", "translated"),
      /Choose an existing writable folder/,
    );
  }
  const client = api(() => new Response("unavailable", { status: 500 }));
  await assert.rejects(
    client.loadProjectsHomeSettings("Chargement impossible"),
    /Chargement impossible \(500\)/,
  );
  await assert.rejects(
    client.updateProjectsHomeSettings(null, "Enregistrement impossible"),
    /Enregistrement impossible \(500\)/,
  );
});

function translate(key: string, values: Record<string, string> = {}): string {
  const message = key
    .split(".")
    .reduce<unknown>(
      (value, part) => (value as Record<string, unknown>)[part],
      en,
    ) as string;
  return message.replace(
    /\{([^}]+)\}/g,
    (_match, name: string) => values[name] ?? "",
  );
}

function nodes(value: unknown): StubElement[] {
  if (Array.isArray(value)) return value.flatMap(nodes);
  if (!value || typeof value !== "object" || !("props" in value)) return [];
  const node = value as StubElement;
  return [node, ...Object.values(node.props).flatMap(nodes)];
}

function text(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(text).join(" ");
  if (!value || typeof value !== "object" || !("props" in value)) return "";
  const props = (value as StubElement).props;
  return [text(props.children), text(props.description)].join(" ");
}

function byId(tree: unknown, id: string): StubElement {
  const node = nodes(tree).find((entry) => entry.props["data-testid"] === id);
  assert.ok(node, id);
  return node;
}

const tick = () => new Promise<void>((resolve) => setImmediate(resolve));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function ui(
  options: {
    load?: () => Promise<ProjectsHomeSettings>;
    update?: (path: string | null) => Promise<ProjectsHomeSettings>;
  } = {},
) {
  const states: unknown[] = [];
  const refs: { current: unknown }[] = [];
  const effects: { deps: unknown[]; cleanup?: () => void }[] = [];
  let stateIndex = 0;
  let refIndex = 0;
  let effectIndex = 0;
  let pending: (() => void)[] = [];
  const updates: (string | null)[] = [];
  const successes: string[] = [];
  let currentTranslate = translate;
  const component = loadWithStubs<{ ProjectsHomeSetting: () => StubElement }>(
    new URL(
      "../src/features/settings/components/projects-home-setting.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        useState(initial: unknown) {
          const index = stateIndex++;
          if (!(index in states)) states[index] = initial;
          return [
            states[index],
            (value: unknown) => {
              states[index] =
                typeof value === "function" ? value(states[index]) : value;
            },
          ];
        },
        useRef(initial: unknown) {
          const index = refIndex++;
          refs[index] ??= { current: initial };
          return refs[index];
        },
        useEffect(effect: () => (() => void) | undefined, deps: unknown[]) {
          const index = effectIndex++;
          const previous = effects[index];
          if (
            !previous ||
            deps.some((dep, i) => !Object.is(dep, previous.deps[i]))
          ) {
            pending.push(() => {
              previous?.cleanup?.();
              effects[index] = { deps, cleanup: effect() };
            });
          }
        },
      },
      "@/components/ui/button": { Button: "Button" },
      "@/features/model-picker": { FolderBrowser: "FolderBrowser" },
      "./storage-overview": { SettingsCardRow: "SettingsCardRow" },
      "@/i18n": { useT: () => currentTranslate },
      "@/lib/toast": {
        toast: { success: (message: string) => successes.push(message) },
      },
      "../api/projects-home": {
        loadProjectsHomeSettings: options.load ?? (async () => DEFAULT),
        updateProjectsHomeSettings: async (path: string | null) => {
          updates.push(path);
          return options.update
            ? options.update(path)
            : path === null
              ? DEFAULT
              : CUSTOM;
        },
      },
    },
  );
  function render() {
    stateIndex = refIndex = effectIndex = 0;
    const tree = component.ProjectsHomeSetting();
    const scheduled = pending;
    pending = [];
    for (const effect of scheduled) effect();
    return tree;
  }
  return {
    render,
    updates,
    successes,
    changeLocale() {
      // useT returns a new callback when another tab changes the locale.
      currentTranslate = (key, values) => translate(key, values);
      return render();
    },
    async initialize() {
      render();
      await tick();
      return render();
    },
    unmount() {
      for (const effect of effects) effect.cleanup?.();
    },
  };
}

test("the row browses the effective server path and applies selection and reset", async () => {
  const view = ui();
  assert.equal(
    byId(view.render(), "projects-home-change").props.disabled,
    true,
  );
  await tick();
  let tree = view.render();
  assert.equal(
    text(byId(tree, "projects-home-path")).trim(),
    DEFAULT.projectsHome,
  );
  assert.match(text(tree), /Existing projects stay where they are/);
  (byId(tree, "projects-home-change").props.onClick as () => void)();
  tree = view.render();
  const picker = nodes(tree).find((node) => node.type === "FolderBrowser");
  assert.ok(picker);
  assert.equal(picker.props.open, true);
  assert.equal(picker.props.initialPath, DEFAULT.projectsHome);
  assert.equal(picker.props.showModelHints, false);
  (picker.props.onSelect as (path: string) => void)(CUSTOM.projectsHome);
  await tick();
  tree = view.render();
  assert.equal(
    text(byId(tree, "projects-home-path")).trim(),
    CUSTOM.projectsHome,
  );
  (byId(tree, "projects-home-reset").props.onClick as () => void)();
  await tick();
  tree = view.render();
  assert.equal(
    text(byId(tree, "projects-home-path")).trim(),
    DEFAULT.projectsHome,
  );
  assert.deepEqual(view.updates, [CUSTOM.projectsHome, null]);
  assert.equal(view.successes.length, 2);
});

test("environment-managed paths cannot be changed or reset", async () => {
  const view = ui({
    load: async () => ({
      ...CUSTOM,
      source: "environment",
      editable: false,
      isCustom: false,
      environmentVariable: "UNSLOTH_STUDIO_PROJECTS_HOME",
    }),
  });
  const tree = await view.initialize();
  assert.match(
    text(tree),
    /Managed by the UNSLOTH_STUDIO_PROJECTS_HOME environment variable/,
  );
  assert.equal(byId(tree, "projects-home-change").props.disabled, true);
  assert.ok(
    !nodes(tree).some(
      (node) => node.props["data-testid"] === "projects-home-reset",
    ),
  );
  const picker = nodes(tree).find((node) => node.type === "FolderBrowser");
  assert.ok(picker);
  (picker.props.onSelect as (path: string) => void)("/ignored");
  await tick();
  assert.deepEqual(view.updates, []);
});

test("failed saves keep the effective path and show the server error", async () => {
  const view = ui({
    load: async () => CUSTOM,
    update: async () => {
      throw new Error("Folder is unavailable");
    },
  });
  const tree = await view.initialize();
  (byId(tree, "projects-home-reset").props.onClick as () => void)();
  await tick();
  const failed = view.render();
  assert.equal(
    text(byId(failed, "projects-home-path")).trim(),
    CUSTOM.projectsHome,
  );
  assert.match(
    text(byId(failed, "projects-home-error")),
    /Folder is unavailable/,
  );
  assert.equal(byId(failed, "projects-home-change").props.disabled, false);
  assert.deepEqual(view.successes, []);
});

test("a failed initial load can be retried without reopening Settings", async () => {
  let attempts = 0;
  const view = ui({
    load: async () => {
      if (attempts++ === 0) throw new Error("Unavailable");
      return DEFAULT;
    },
  });
  const tree = await view.initialize();
  assert.match(text(byId(tree, "projects-home-error")), /Unavailable/);
  (byId(tree, "projects-home-refresh").props.onClick as () => void)();
  view.render();
  await tick();
  assert.equal(
    text(byId(view.render(), "projects-home-path")).trim(),
    DEFAULT.projectsHome,
  );
  assert.equal(attempts, 2);
});

test("an in-flight save disables controls and does not notify after unmount", async () => {
  const save = deferred<ProjectsHomeSettings>();
  const view = ui({
    load: async () => CUSTOM,
    update: () => save.promise,
  });
  const tree = await view.initialize();
  (byId(tree, "projects-home-reset").props.onClick as () => void)();
  assert.equal(
    byId(view.render(), "projects-home-change").props.disabled,
    true,
  );
  assert.equal(byId(view.render(), "projects-home-reset").props.disabled, true);
  view.unmount();
  save.resolve(DEFAULT);
  await tick();
  assert.deepEqual(view.successes, []);
});

test("a successful locale reload clears an earlier load error", async () => {
  let attempts = 0;
  const view = ui({
    load: async () => {
      if (attempts++ === 0) throw new Error("Unavailable");
      return CUSTOM;
    },
  });
  const failed = await view.initialize();
  assert.match(text(byId(failed, "projects-home-error")), /Unavailable/);
  view.changeLocale();
  await tick();
  const loaded = view.render();
  assert.equal(
    text(byId(loaded, "projects-home-path")).trim(),
    CUSTOM.projectsHome,
  );
  assert.ok(
    !nodes(loaded).some(
      (node) => node.props["data-testid"] === "projects-home-error",
    ),
  );
  assert.equal(byId(loaded, "projects-home-change").props.disabled, false);
  assert.equal(attempts, 2);
});

for (const outcome of ["success", "error"] as const) {
  for (const timing of ["before save", "during save"] as const) {
    test(`a locale read started ${timing} cannot replace a saved folder with late ${outcome}`, async () => {
      let loads = 0;
      const read = deferred<ProjectsHomeSettings>();
      const save = deferred<ProjectsHomeSettings>();
      const view = ui({
        load: () => (++loads === 1 ? Promise.resolve(DEFAULT) : read.promise),
        update: () => save.promise,
      });
      let tree = await view.initialize();
      if (timing === "before save") tree = view.changeLocale();
      const picker = nodes(tree).find((node) => node.type === "FolderBrowser");
      assert.ok(picker);
      (picker.props.onSelect as (path: string) => void)(CUSTOM.projectsHome);
      if (timing === "during save") view.changeLocale();
      save.resolve(CUSTOM);
      await tick();
      assert.equal(
        text(byId(view.render(), "projects-home-path")).trim(),
        CUSTOM.projectsHome,
      );
      if (outcome === "success") read.resolve(DEFAULT);
      else read.reject(new Error("Stale read failed"));
      await tick();
      const saved = view.render();
      assert.equal(
        text(byId(saved, "projects-home-path")).trim(),
        CUSTOM.projectsHome,
      );
      assert.ok(
        !nodes(saved).some(
          (node) => node.props["data-testid"] === "projects-home-error",
        ),
      );
      assert.equal(byId(saved, "projects-home-change").props.disabled, false);
      assert.deepEqual(view.updates, [CUSTOM.projectsHome]);
      assert.equal(view.successes.length, 1);
    });
  }
}

for (const outcome of ["success", "error"] as const) {
  test(`a late read ${outcome} cannot replace a failed reset's validation error`, async () => {
    let loads = 0;
    const read = deferred<ProjectsHomeSettings>();
    const view = ui({
      load: () => (++loads === 1 ? Promise.resolve(CUSTOM) : read.promise),
      update: async () => {
        throw new Error("The projects folder is managed by the environment");
      },
    });
    await view.initialize();
    const tree = view.changeLocale();
    (byId(tree, "projects-home-reset").props.onClick as () => void)();
    await tick();
    if (outcome === "success") read.resolve(DEFAULT);
    else read.reject(new Error("Stale read failed"));
    await tick();
    const failed = view.render();
    assert.equal(
      text(byId(failed, "projects-home-path")).trim(),
      CUSTOM.projectsHome,
    );
    assert.match(
      text(byId(failed, "projects-home-error")),
      /managed by the environment/,
    );
    assert.deepEqual(view.successes, []);
  });
}

test("a read that fails during a pending save leaves no error after save success", async () => {
  let loads = 0;
  const read = deferred<ProjectsHomeSettings>();
  const save = deferred<ProjectsHomeSettings>();
  const view = ui({
    load: () => (++loads === 1 ? Promise.resolve(CUSTOM) : read.promise),
    update: () => save.promise,
  });
  const tree = await view.initialize();
  (byId(tree, "projects-home-reset").props.onClick as () => void)();
  view.changeLocale();
  read.reject(new Error("Read failed during save"));
  await tick();
  save.resolve(DEFAULT);
  await tick();
  const saved = view.render();
  assert.equal(
    text(byId(saved, "projects-home-path")).trim(),
    DEFAULT.projectsHome,
  );
  assert.ok(
    !nodes(saved).some(
      (node) => node.props["data-testid"] === "projects-home-error",
    ),
  );
  assert.equal(view.successes.length, 1);
});

test("project folder search routes owners to Resources and hides the row for members", () => {
  const key = "settings.resources.storage.projectsFolder.label";
  for (const desktop of [false, true]) {
    const index = createSettingsSearchIndex({
      desktop,
      closeToTray: false,
      menuBarIcon: false,
    });
    assert.ok(
      renderedSearchEntries(index, "resources", "huggingface", true).includes(
        key,
      ),
    );
    assert.ok(
      !renderedSearchEntries(index, "resources", "huggingface", false).includes(
        key,
      ),
    );
  }
  assert.equal(
    SETTINGS_SEARCH_KEYWORDS[key],
    "settings.resources.storage.projectsFolder.keywords",
  );
  for (const term of ["projects", "workspace", "directory", "default"]) {
    assert.ok(
      en.settings.resources.storage.projectsFolder.keywords.includes(term),
    );
  }
});
