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

const row = (label: string) => {
  const start = source.indexOf(
    `<BackupSettingRow\n              label="${label}"`,
  );
  assert.notEqual(start, -1, `${label} uses BackupSettingRow`);
  return source.slice(start, source.indexOf("</BackupSettingRow>", start));
};

test("backup settings use one responsive, contained control grid", () => {
  assert.match(
    source,
    /grid-cols-1 items-center gap-x-6 gap-y-2 md:grid-cols-\[minmax\(0,1fr\)_12rem\]/,
  );
  assert.match(source, /className="col-span-full text-xs font-medium/g);
  assert.match(
    source,
    /className="flex w-full min-w-0 justify-self-end justify-end"/,
  );
  assert.match(source, /box-border w-full min-w-0 max-w-full/);
  assert.doesNotMatch(source, /(?:absolute|translate-x|margin-left|ml-)\b/);
});

test("repository, interval, and retention controls share their label rows", () => {
  assert.match(row("Repository ID"), /id="checkpoint-backup-repo"/);
  assert.match(row("Upload backup every"), /<Select/);
  assert.match(
    row("Remote checkpoints to keep"),
    /id="checkpoint-backup-retention"/,
  );
  assert.match(source, /hidden md:block/);
  assert.match(source, /Example: username\/my-training-backups/);
  assert.doesNotMatch(source, /Test access/);
  assert.match(source, /className="box-border w-full min-w-0 max-w-full pr-9"/);
  assert.match(source, /LoaderCircle/);
  assert.match(source, /CircleCheck/);
  assert.match(source, /CircleAlert/);
});

test("final-upload switches use the shared right-aligned control column", () => {
  assert.match(row("Upload when training stops"), /<Switch/);
  assert.match(row("Upload when training completes"), /<Switch/);
});

test("hints remain adjacent to labels", () => {
  assert.match(
    source,
    /inline-flex min-w-0 items-center gap-1\.5[\s\S]*?\{children\}[\s\S]*?<FieldHint/,
  );
  for (const name of [
    "More information about Repository ID",
    "More information about backup frequency",
    "More information about remote checkpoint retention",
    "More information about uploading when training stops",
    "More information about uploading when training completes",
  ]) {
    assert.match(source, new RegExp(name));
  }
});

test("disabled cadence uses aligned guidance instead of loose status text", () => {
  assert.match(source, /disabled=\{store\.saveSteps <= 0\}/);
  assert.match(source, /aria-disabled=\{store\.saveSteps <= 0\}/);
  assert.match(
    source,
    /Periodic uploads are unavailable because Save Steps is 0\./,
  );
  assert.match(source, /Configure Save Steps/);
  assert.match(source, /text-primary underline-offset-4 hover:underline/);
  assert.doesNotMatch(source, />\s*Disabled\s*</);
  assert.doesNotMatch(source, /value=\{store\.saveSteps\}/);
});
