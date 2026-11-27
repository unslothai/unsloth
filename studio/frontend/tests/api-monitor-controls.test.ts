// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

type Settings = {
  enabled: boolean;
  effective_enabled: boolean;
  disabled_by_env: boolean;
};

/** Walk the rendered component tree to find accessible controls. */
function nodes(value: unknown): StubElement[] {
  if (Array.isArray(value)) return value.flatMap(nodes);
  if (!value || typeof value !== "object" || !("props" in value)) return [];
  const node = value as StubElement;
  return [node, ...nodes(node.props.children)];
}

/** Mount the real settings component with controlled network and account boundaries. */
async function mount(owner: boolean, initial: Settings) {
  const states: unknown[] = [];
  let cursor = 0;
  let effect: (() => void) | undefined;
  const writes: boolean[] = [];
  const store = Object.assign(
    (selector: (state: object) => unknown) =>
      selector({
        autoOpen: false,
        setAutoOpen: () => undefined,
      }),
    { getState: () => ({ closeDialog: () => undefined }) },
  );
  const translate = (key: string) => key;
  const { MonitorLink } = loadWithStubs<{ MonitorLink(): StubElement }>(
    new URL(
      "../src/features/settings/components/monitor-link.tsx",
      import.meta.url,
    ),
    {
      "react/jsx-runtime": stubJsxRuntime(),
      react: {
        useState(initialValue: unknown) {
          const i = cursor++;
          if (!(i in states)) states[i] = initialValue;
          return [
            states[i],
            (value: unknown) => {
              states[i] = value;
            },
          ];
        },
        useEffect(callback: () => void) {
          effect = callback;
        },
      },
      "@/components/ui/switch": { Switch: "switch" },
      "@/features/auth": { useIsAccountOwner: () => owner },
      "@/features/api-monitor/overlay-store": {
        useApiMonitorOverlayStore: store,
      },
      "@/features/chat/api/chat-api": {
        getApiMonitor: async () => ({ entries: [] }),
      },
      "@/i18n": { useLocale: () => "en", useT: () => translate },
      "@/lib/utils": { cn: () => "" },
      "@hugeicons/core-free-icons": { ActivityIcon: "activity" },
      "lucide-react": { ArrowRightIcon: "arrow" },
      "@hugeicons/react": { HugeiconsIcon: "icon" },
      "@tanstack/react-router": { useNavigate: () => () => undefined },
      "../stores/settings-dialog-store": { useSettingsDialogStore: store },
      "../api/api-monitor": {
        loadApiMonitorSettings: async () => initial,
        updateApiMonitorSettings: async (enabled: boolean) => {
          writes.push(enabled);
          return {
            enabled,
            effective_enabled: enabled,
            disabled_by_env: false,
          };
        },
      },
      "./settings-row": { SettingsRow: "row" },
      "./settings-section": { SettingsSection: "section" },
    },
  );
  /** Render from the current hook state. */
  function render() {
    cursor = 0;
    return nodes(MonitorLink()).find(
      (node) =>
        node.props["aria-label"] === "settings.resources.liveMonitor.recording",
    );
  }
  render();
  effect?.();
  await new Promise((resolve) => setImmediate(resolve));
  return { render, writes };
}

test("only the owner can change recording, and saving uses the chosen value", async () => {
  const initial = {
    enabled: true,
    effective_enabled: true,
    disabled_by_env: false,
  };
  const owner = await mount(true, initial);
  const toggle = owner.render();
  assert.equal(toggle?.props.checked, true);
  assert.equal(toggle?.props.disabled, false);
  (toggle?.props.onCheckedChange as (enabled: boolean) => void)(false);
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(owner.writes, [false]);
  assert.equal(owner.render()?.props.checked, false);
  const managed = await mount(false, initial);
  assert.equal(managed.render()?.props.disabled, true);
});

test("environment override locks the recording control", async () => {
  const page = await mount(true, {
    enabled: true,
    effective_enabled: false,
    disabled_by_env: true,
  });
  assert.equal(page.render()?.props.disabled, true);
  assert.equal(page.render()?.props.checked, false);
});
