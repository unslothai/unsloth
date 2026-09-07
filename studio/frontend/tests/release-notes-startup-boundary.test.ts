// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
const read = (name: string) => readFileSync(new URL("../src/" + name, import.meta.url), "utf8");

test("web and desktop update banners defer the same notes renderer", () => {
  for (const name of ["web", "tauri"]) {
    assert.ok(read(`components/${name}/update-banner.tsx`).includes('from "@/components/update/release-notes-panel-mount"'));
  }
  const source = read("components/update/release-notes-panel-mount.tsx");
  assert.ok(source.includes('const Notes = lazy(() => import("./release-notes-panel")'));
  assert.ok(source.includes("type Props = ComponentProps<typeof import"));
  assert.ok(source.indexOf("const Notes = lazy") < source.indexOf("export function"));
});

test("collapsed previews still request notes and errors do not hide update controls", () => {
  const source = read("components/update/release-notes-panel-mount.tsx");
  assert.ok(source.includes("<Notes {...props} />"));
  assert.ok(!source.includes("if (!props.open)"));
  assert.ok(source.includes("<LazyImportBoundary"));
  assert.ok(source.includes('role="alert"'));
  assert.ok(source.includes('role="status"'));
  assert.ok(!source.includes("useTauriUpdate"));
  assert.ok(!source.includes("useWebUpdate"));
});
