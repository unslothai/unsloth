// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hydration calls setLocale, which can come back "superseded" when a newer
// request took the language over. The rest of personalization has still
// hydrated by then, so the sync has to finish; leaving it unfinished pauses the
// save gate for the whole signed-in session.

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Slot = {
  value?: unknown;
  deps?: readonly unknown[];
  // Whatever the effect returned: a cleanup, or nothing.
  cleanup?: unknown;
  set?: boolean;
};

function runCleanup(cleanup: unknown): void {
  if (typeof cleanup === "function") (cleanup as () => void)();
}

type Profile = {
  displayName: string;
  nickname: string;
  avatarDataUrl: string | null;
  avatarShape: "circle" | "rounded";
  showGreetingSloth: boolean;
};

type SavedPayload = {
  version: number;
  profile: Profile;
  appearance: {
    theme: string;
    palette: string;
    language: string | null;
    customization: Record<string, unknown>;
  };
};

const HOOK_URL = new URL(
  "../src/features/profile/hooks/use-personalization-sync.ts",
  import.meta.url,
);

function sameDeps(a: readonly unknown[], b: readonly unknown[]): boolean {
  return a.length === b.length && a.every((value, i) => Object.is(value, b[i]));
}

/** The four hooks the sync uses, with renders and effects the test drives. */
function createReact() {
  const slots: Slot[] = [];
  const effects: (() => void)[] = [];
  let cursor = 0;
  let dirty = false;

  const slot = (): Slot => {
    const existing = slots[cursor];
    if (existing) {
      cursor += 1;
      return existing;
    }
    const created: Slot = {};
    slots[cursor] = created;
    cursor += 1;
    return created;
  };

  const react = {
    useState<T>(initial: T): [T, (next: T) => void] {
      const self = slot();
      if (!self.set) {
        self.value = initial;
        self.set = true;
      }
      return [
        self.value as T,
        (next: T) => {
          if (Object.is(next, self.value)) return;
          self.value = next;
          dirty = true;
        },
      ];
    },
    useRef<T>(initial: T): { current: T } {
      const self = slot();
      if (!self.set) {
        self.value = { current: initial };
        self.set = true;
      }
      return self.value as { current: T };
    },
    useCallback<T>(fn: T, deps: readonly unknown[]): T {
      const self = slot();
      if (!self.deps || !sameDeps(self.deps, deps)) {
        self.value = fn;
        self.deps = deps;
      }
      return self.value as T;
    },
    useEffect(fn: () => unknown, deps: readonly unknown[]): void {
      const self = slot();
      if (self.deps && sameDeps(self.deps, deps)) return;
      self.deps = deps;
      effects.push(() => {
        runCleanup(self.cleanup);
        self.cleanup = fn();
      });
    },
  };

  return {
    react,
    /** Renders until no state change is left, running effects after each pass. */
    flush(body: () => void): void {
      do {
        cursor = 0;
        dirty = false;
        body();
        while (effects.length) effects.shift()?.();
      } while (dirty);
    },
    unmount(): void {
      for (const self of slots) runCleanup(self.cleanup);
    },
  };
}

/** Timers the test releases by hand, so the debounced push is not a real wait. */
function installWindow() {
  const timers = new Map<number, () => void>();
  let nextId = 1;
  Object.assign(globalThis, {
    window: {
      setTimeout(fn: () => void): number {
        const id = nextId;
        nextId += 1;
        timers.set(id, fn);
        return id;
      },
      clearTimeout(id: number): void {
        timers.delete(id);
      },
    },
  });
  return () => {
    const due = [...timers.values()];
    timers.clear();
    for (const fn of due) fn();
  };
}

/** Lets every pending promise callback run. */
function settle(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}

function remotePersonalization(language: string) {
  return {
    version: 3,
    profile: {
      displayName: "",
      nickname: "",
      avatarDataUrl: null,
      avatarShape: "circle" as const,
      showGreetingSloth: true,
    },
    appearance: {
      theme: "system",
      palette: "standard",
      language,
      customization: {},
    },
    saved: true,
    customizationSaved: true,
    paletteSaved: true,
    greetingSlothSaved: true,
  };
}

function setup(localeResult: "superseded" | "cancelled") {
  const runTimers = installWindow();
  const host = createReact();
  const saves: SavedPayload[] = [];
  const localeCalls: unknown[] = [];
  let profile: Profile = {
    displayName: "",
    nickname: "",
    avatarDataUrl: null,
    avatarShape: "circle",
    showGreetingSloth: true,
  };
  // A newer request already took French, which is the state "superseded"
  // reports: the hydrated language is not the one in effect.
  let preference = "auto";
  let releaseLocale!: () => void;
  const localeSettled = new Promise<void>((resolve) => {
    releaseLocale = resolve;
  });

  const profileStore = Object.assign(
    (select: (state: Profile) => unknown) => select(profile),
    {
      getState: () => profile,
      setState: (next: Partial<Profile>) => {
        profile = { ...profile, ...next };
      },
    },
  );

  const { usePersonalizationSync } = loadWithStubs<{
    usePersonalizationSync: (enabled: boolean) => void;
  }>(HOOK_URL, {
    react: host.react,
    "@/features/settings": {
      isDefaultCustomization: (value: unknown) =>
        Object.keys(value as object).length === 0,
      isPalette: (value: unknown) => typeof value === "string",
      loadPersonalization: () => Promise.resolve(remotePersonalization("de")),
      migrateShippedSidebarNavDefault: (value: unknown) => value,
      sanitizeCustomization: (value: unknown) => value ?? {},
      savePersonalization: (data: SavedPayload) => {
        saves.push(data);
        return Promise.resolve();
      },
      setPalette: () => undefined,
      setTheme: () => undefined,
      useAppearanceCustomStore: (select: (state: unknown) => unknown) =>
        select({ customization: {} }),
      usePalette: () => ({ palette: "standard" }),
      useTheme: () => ({ theme: "system" }),
    },
    "@/i18n": {
      DEFAULT_LOCALE_PREFERENCE: "auto",
      getLocalePreference: () => preference,
      isLocalePreference: (value: unknown) => typeof value === "string",
      setLocale: async (value: unknown) => {
        localeCalls.push(value);
        await localeSettled;
        if (localeResult === "superseded") preference = "fr";
        return localeResult;
      },
      useLocalePreference: () => preference,
    },
    "../stores/user-profile-store": {
      PROFILE_TEXT_MAX_LENGTH: 200,
      useUserProfileStore: profileStore,
    },
  });

  return {
    host,
    localeCalls,
    releaseLocale,
    runTimers,
    saves,
    rename(displayName: string) {
      profile = { ...profile, displayName };
    },
    render() {
      host.flush(() => usePersonalizationSync(true));
    },
  };
}

test("a superseded locale hydration does not pause personalization saves", async () => {
  const app = setup("superseded");

  app.render();
  await settle();
  assert.deepEqual(app.localeCalls, ["de"]);

  app.releaseLocale();
  await settle();
  app.render();

  // Pre-fix the sync returned here without finishing hydration, so the save
  // effect's generation gate never opened again and nothing the user changed
  // for the rest of the session was written.
  app.rename("Ada");
  app.render();
  app.runTimers();
  await settle();

  assert.equal(app.saves.length, 1);
  assert.equal(app.saves[0]?.profile.displayName, "Ada");
  // The push carries the language actually in effect, not the superseded one.
  assert.equal(app.saves[0]?.appearance.language, "fr");
});

test("a cancelled locale hydration stays paused, because it never finished", async () => {
  const app = setup("cancelled");

  app.render();
  await settle();
  app.host.unmount();

  app.releaseLocale();
  await settle();
  app.render();
  app.rename("Ada");
  app.render();
  app.runTimers();
  await settle();

  assert.equal(app.saves.length, 0);
});
