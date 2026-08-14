import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL(
    "../src/features/studio/sections/checkpoint-backup-section.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("disabled backups cannot expose or render the disclosure", () => {
  assert.match(source, /open=\{backupEnabled && backupExpanded\}/);
  assert.match(
    source,
    /if \(backupEnabled\) \{\s*setBackupExpanded\(expanded\);/,
  );
  assert.match(source, /disabled=\{!backupEnabled\}/);
  assert.match(source, /aria-disabled=\{!backupEnabled\}/);
  assert.match(source, /aria-expanded=\{backupEnabled && backupExpanded\}/);
  assert.match(source, /\{backupEnabled && \(\s*<CollapsibleContent/);
});

test("the enable switch owns the disclosure transition", () => {
  assert.match(
    source,
    /const setBackupEnabled = \(enabled: boolean\) => \{\s*update\(\{ enabled \}\);\s*setBackupExpanded\(enabled\);/,
  );
  assert.match(source, /onCheckedChange=\{setBackupEnabled\}/);
  assert.match(source, /onClick=\{\(event\) => event\.stopPropagation\(\)\}/);
  assert.match(source, /onKeyDown=\{\(event\) => event\.stopPropagation\(\)\}/);
});

test("the hint remains adjacent, accessible, and separate from the switch", () => {
  const title = source.indexOf("Automatic Hugging Face backups</span>");
  const hint = source.indexOf('label="About automatic Hugging Face backups"');
  const switchControl = source.indexOf("<Switch", hint);

  assert.ok(title >= 0 && hint > title && switchControl > hint);
  assert.match(source, /className="shrink-0"[\s\S]*?onPointerDown=/);
  assert.match(source, /className="flex shrink-0 items-center"/);
  assert.match(source, /className="truncate">Automatic Hugging Face backups/);
  assert.match(source, /className="cursor-pointer"/);
});
