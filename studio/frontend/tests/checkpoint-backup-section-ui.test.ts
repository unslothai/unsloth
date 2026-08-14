import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL(
    "../src/features/studio/sections/checkpoint-backup-section.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("backup settings expose consistent accessible hint controls", () => {
  for (const name of [
    "More information about automatic Hugging Face backups",
    "More information about Repository ID",
    "More information about backup frequency",
    "More information about remote checkpoint retention",
    "More information about uploading when training stops",
    "More information about uploading when training completes",
  ]) {
    assert.match(source, new RegExp(name));
  }
  assert.match(source, /<FieldHint/g);
});

test("backup form keeps responsive containment and zero-save behavior", () => {
  assert.match(source, /grid-cols-\[minmax\(0,1fr\)\]/);
  assert.match(source, /box-border w-full min-w-0 max-w-full/);
  assert.match(source, /disabled=\{store\.saveSteps <= 0\}/);
  assert.match(source, /Periodic backups are off because Save Steps is 0\./);
  assert.match(source, /Configure Save Steps/);
  assert.doesNotMatch(source, /value=\{store\.saveSteps\}/);
});

test("backup labels, access states, and validation help remain explicit", () => {
  for (const text of [
    "Destination",
    "Backup schedule",
    "Retention and final uploads",
    "Remote checkpoints to keep",
    "Authentication required",
    "No write permission",
    "Enter a repository ID before testing access.",
    "It does not change local checkpoint retention",
  ]) {
    assert.match(source, new RegExp(text));
  }
});
