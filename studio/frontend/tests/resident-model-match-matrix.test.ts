// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A wrong TRUE here keeps weights the user did not ask for; a wrong FALSE reloads for
 * nothing. Both answers depend on strings the BACKEND chose, which differ on two axes:
 *
 *   host      a snapshot path is POSIX on Linux and macOS, a drive path or UNC share on
 *             Windows, /mnt/<letter> under WSL; separators and case fold differently
 *   version   older backends published no model_identifier, or put the RAW path in
 *             active_model; a native lease withholds it on every version
 *
 * The table is those two crossed with the model kinds Unsloth loads. Every row states the
 * answer the USER needs, not the one the code happens to give.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { residentModelMatchesPick } = await import(
  "../src/features/chat/lib/resident-model-match.ts"
);

type Row = {
  host: string;
  what: string;
  status: {
    active_model?: string | null;
    model_identifier?: string | null;
    gguf_variant?: string | null;
  };
  pick: { id: string; loadPath?: string | null; ggufVariant?: string | null };
  resident: boolean;
};

const REPO = "unsloth/Qwen3-0.6B-GGUF";
const OTHER = "unsloth/gemma-4-12b-GGUF";
const Q = "Q4_K_M";

// One snapshot of one repo, spelled the way each host spells it.
const SNAP = {
  linux:
    "/home/dev/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/a1b2c3",
  mac: "/Users/dev/.cache/huggingface/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/a1b2c3",
  win: "D:\\models\\hub\\models--unsloth--Qwen3-0.6B-GGUF\\snapshots\\a1b2c3",
  unc: "\\\\nas\\models\\hub\\models--unsloth--Qwen3-0.6B-GGUF\\snapshots\\a1b2c3",
  wsl: "/mnt/c/Users/dev/models/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/a1b2c3",
};

const FILE = {
  linux: "/home/dev/models/Qwen3-0.6B-Q4_K_M.gguf",
  mac: "/Users/dev/models/Qwen3-0.6B-Q4_K_M.gguf",
  win: "C:\\Users\\dev\\models\\Qwen3-0.6B-Q4_K_M.gguf",
  wsl: "/mnt/c/Users/dev/models/Qwen3-0.6B-Q4_K_M.gguf",
};

const ROWS: Row[] = [];

// ── The pinned cached row, the shape #8893 was reported on ───────────────────
for (const [host, snap] of Object.entries(SNAP)) {
  ROWS.push({
    host,
    what: "the picker row's repo id names the resident snapshot it loaded from",
    status: { active_model: REPO, model_identifier: snap, gguf_variant: Q },
    pick: { id: REPO, loadPath: snap, ggufVariant: Q },
    resident: true,
  });
  ROWS.push({
    host,
    what: "a different quant of that same snapshot is a real reload",
    status: { active_model: REPO, model_identifier: snap, gguf_variant: Q },
    pick: { id: REPO, loadPath: snap, ggufVariant: "Q8_0" },
    resident: false,
  });
  ROWS.push({
    host,
    what: "a newer snapshot of the resident repo is a real reload",
    status: { active_model: REPO, model_identifier: snap, gguf_variant: Q },
    pick: {
      id: REPO,
      loadPath: snap.replace("a1b2c3", "d4e5f6"),
      ggufVariant: Q,
    },
    resident: false,
  });
  ROWS.push({
    host,
    what: "another repo entirely is a real reload",
    status: { active_model: REPO, model_identifier: snap, gguf_variant: Q },
    pick: {
      id: OTHER,
      loadPath: snap.replace("Qwen3-0.6B", "gemma-4-12b"),
      ggufVariant: Q,
    },
    resident: false,
  });
  ROWS.push({
    host,
    what: "quant labels differing only in case name the same weights",
    status: { active_model: REPO, model_identifier: snap, gguf_variant: Q },
    pick: { id: REPO, loadPath: snap, ggufVariant: Q.toLowerCase() },
    resident: true,
  });
  ROWS.push({
    host,
    what: "a quant label the user's browser padded still names the same weights",
    status: {
      active_model: REPO,
      model_identifier: snap,
      gguf_variant: ` ${Q} `,
    },
    pick: { id: REPO, loadPath: snap, ggufVariant: Q },
    resident: true,
  });
}

// ── Windows spells one directory several ways, and means one directory ───────
ROWS.push({
  host: "win",
  what: "forward slashes name the same directory as backslashes",
  status: { active_model: REPO, model_identifier: SNAP.win, gguf_variant: Q },
  pick: { id: REPO, loadPath: SNAP.win.replace(/\\/g, "/"), ggufVariant: Q },
  resident: true,
});
ROWS.push({
  host: "win",
  what: "a drive letter's case does not change the directory",
  status: { active_model: REPO, model_identifier: SNAP.win, gguf_variant: Q },
  pick: { id: REPO, loadPath: SNAP.win.toLowerCase(), ggufVariant: Q },
  resident: true,
});
ROWS.push({
  host: "win",
  what: "a trailing separator does not change the directory",
  status: {
    active_model: REPO,
    model_identifier: `${SNAP.win}\\`,
    gguf_variant: Q,
  },
  pick: { id: REPO, loadPath: SNAP.win, ggufVariant: Q },
  resident: true,
});
ROWS.push({
  host: "unc",
  what: "a UNC share compares case-insensitively like the rest of Windows",
  status: { active_model: REPO, model_identifier: SNAP.unc, gguf_variant: Q },
  pick: { id: REPO, loadPath: SNAP.unc.toUpperCase(), ggufVariant: Q },
  resident: true,
});

// ── POSIX is case-SENSITIVE, and two files can differ by case alone ──────────
ROWS.push({
  host: "linux",
  what: "two snapshot dirs differing only in case are different directories",
  status: { active_model: REPO, model_identifier: SNAP.linux, gguf_variant: Q },
  pick: { id: REPO, loadPath: SNAP.linux.toUpperCase(), ggufVariant: Q },
  resident: false,
});
ROWS.push({
  host: "mac",
  what: "the same holds for a mac home directory",
  status: { active_model: REPO, model_identifier: SNAP.mac, gguf_variant: Q },
  pick: {
    id: REPO,
    loadPath: SNAP.mac.replace("/Users/dev", "/Users/DEV"),
    ggufVariant: Q,
  },
  resident: false,
});

// ── The unpinned cached row: the repo id IS the load identifier ──────────────
for (const host of ["linux", "mac", "win", "wsl"]) {
  ROWS.push({
    host,
    what: "a repo that loads by its own id is resident under that id",
    status: { active_model: REPO, model_identifier: REPO, gguf_variant: Q },
    pick: { id: REPO, loadPath: REPO, ggufVariant: Q },
    resident: true,
  });
  ROWS.push({
    host,
    what: "a row carrying no pin still names the repo it loaded by id",
    status: { active_model: REPO, model_identifier: REPO, gguf_variant: Q },
    pick: { id: REPO, ggufVariant: Q },
    resident: true,
  });
}

// ── A standalone .gguf: one file, no quant to choose between ─────────────────
for (const [host, file] of Object.entries(FILE)) {
  ROWS.push({
    host,
    what: "a standalone file matches the quant label the backend read off its name",
    status: { active_model: file, model_identifier: file, gguf_variant: Q },
    pick: { id: file },
    resident: true,
  });
  ROWS.push({
    host,
    what: "a different file in the same directory is a real reload",
    status: { active_model: file, model_identifier: file, gguf_variant: Q },
    pick: { id: file.replace("Qwen3-0.6B", "gemma-4-12b") },
    resident: false,
  });
  ROWS.push({
    host,
    what: "the same file name in another directory is a different file",
    status: { active_model: file, model_identifier: file, gguf_variant: Q },
    pick: { id: file.replace("models", "models-old") },
    resident: false,
  });
}

// ── Safetensors and MLX: no quant at all on either side ──────────────────────
ROWS.push({
  host: "mac",
  what: "an MLX repo is resident under its own id with no variant either side",
  status: {
    active_model: "mlx-community/Qwen3-0.6B-4bit",
    model_identifier: "mlx-community/Qwen3-0.6B-4bit",
  },
  pick: { id: "mlx-community/Qwen3-0.6B-4bit" },
  resident: true,
});
ROWS.push({
  host: "linux",
  what: "a safetensors repo is resident under its own id",
  status: {
    active_model: "unsloth/Qwen3-0.6B",
    model_identifier: "unsloth/Qwen3-0.6B",
  },
  pick: { id: "unsloth/Qwen3-0.6B" },
  resident: true,
});
ROWS.push({
  host: "linux",
  what: "a LoRA adapter is not its base model",
  status: {
    active_model: "unsloth/Qwen3-0.6B",
    model_identifier: "unsloth/Qwen3-0.6B",
  },
  pick: { id: "dev/my-qwen3-lora" },
  resident: false,
});
ROWS.push({
  host: "linux",
  what: "two adapters sharing a base are not each other",
  status: { active_model: "dev/lora-a", model_identifier: "dev/lora-a" },
  pick: { id: "dev/lora-b" },
  resident: false,
});

// ── Nothing is loaded ────────────────────────────────────────────────────────
for (const active of [null, undefined, ""]) {
  ROWS.push({
    host: "any",
    what: `nothing resident (active_model ${JSON.stringify(active)}) matches nothing`,
    status: {
      active_model: active,
      model_identifier: SNAP.linux,
      gguf_variant: Q,
    },
    pick: { id: REPO, loadPath: SNAP.linux, ggufVariant: Q },
    resident: false,
  });
}

// ── Older Unsloth backends, i.e. an install that predates these fields ────────
ROWS.push({
  host: "old-install",
  what: "a status with no model_identifier field falls back to the display id",
  status: { active_model: REPO, gguf_variant: Q },
  pick: { id: REPO, ggufVariant: Q },
  resident: true,
});
ROWS.push({
  host: "old-install",
  what: "a backend that put the raw path in active_model still matches its own pick",
  status: { active_model: SNAP.linux, gguf_variant: Q },
  pick: { id: REPO, loadPath: SNAP.linux, ggufVariant: Q },
  resident: true,
});
ROWS.push({
  host: "old-install",
  what: "a backend reporting neither a variant nor a pick variant still matches",
  status: { active_model: REPO },
  pick: { id: REPO },
  resident: true,
});
ROWS.push({
  host: "old-install",
  what: "an unversioned status must not match a different model",
  status: { active_model: REPO, gguf_variant: Q },
  pick: { id: OTHER, ggufVariant: Q },
  resident: false,
});

// ── A native-lease load withholds the raw path on every version ──────────────
ROWS.push({
  host: "native-lease",
  what: "a leased file matches the label it was granted under",
  status: {
    active_model: "Qwen3-0.6B-Q4_K_M.gguf",
    model_identifier: null,
    gguf_variant: Q,
  },
  pick: { id: "Qwen3-0.6B-Q4_K_M.gguf" },
  resident: true,
});
ROWS.push({
  host: "native-lease",
  what: "a leased file does not match another file of the same name on disk",
  status: {
    active_model: "Qwen3-0.6B-Q4_K_M.gguf",
    model_identifier: null,
    gguf_variant: Q,
  },
  pick: { id: FILE.linux },
  resident: false,
});
ROWS.push({
  host: "native-lease",
  what: "a leased file does not match a repo that happens to end in .gguf",
  status: {
    active_model: "Qwen3-0.6B-Q4_K_M.gguf",
    model_identifier: null,
    gguf_variant: Q,
  },
  pick: { id: "unsloth/Qwen3-0.6B-Q4_K_M.gguf" },
  resident: false,
});

// ── The documented false negative, pinned so it cannot drift into a true ─────
ROWS.push({
  host: "any",
  what:
    "a bare repo-id pick does NOT adopt a snapshot-resident model, and must not: " +
    "the backend would not reuse that load either, so the reload is the honest answer",
  status: { active_model: REPO, model_identifier: SNAP.linux, gguf_variant: Q },
  pick: { id: REPO, ggufVariant: Q },
  resident: false,
});

for (const row of ROWS) {
  test(`[${row.host}] ${row.what}`, () => {
    assert.equal(
      residentModelMatchesPick(row.status, row.pick),
      row.resident,
      `${JSON.stringify(row.status)} vs ${JSON.stringify(row.pick)}`,
    );
  });
}

test("the matrix covers every host and backend shape it claims to", () => {
  const hosts = new Set(ROWS.map((row) => row.host));
  for (const expected of [
    "linux",
    "mac",
    "win",
    "unc",
    "wsl",
    "old-install",
    "native-lease",
  ]) {
    assert.ok(hosts.has(expected), `no rows for ${expected}`);
  }
  assert.ok(ROWS.length >= 60, `only ${ROWS.length} rows`);
  // Both answers must be exercised, or the table proves only that it never matches.
  assert.ok(ROWS.some((row) => row.resident));
  assert.ok(ROWS.some((row) => !row.resident));
});

/**
 * A KNOWN limit, pinned here so it is a decision rather than a surprise.
 *
 * normalizeModelIdentity folds case under /mnt/<single letter>/ because that is where WSL
 * mounts a Windows drive, and Windows is case-insensitive. A Linux host with a real
 * single-letter mount point gets the same treatment, so two files there differing only in
 * case read as one. Fixing it needs a platform signal the browser does not have, and the
 * comparator is shared with the Hub, so this stays as-is: the cost is one skipped reload
 * for a path shape no Unsloth install creates on its own.
 */
test("case folding under /mnt/<letter> is WSL-shaped, and known to over-match on Linux", () => {
  assert.equal(
    residentModelMatchesPick(
      {
        active_model: "/mnt/d/models/Model.gguf",
        model_identifier: "/mnt/d/models/Model.gguf",
        gguf_variant: Q,
      },
      { id: "/mnt/d/models/model.gguf" },
    ),
    true,
  );
  // Any other Linux mount point is compared case-sensitively, as it must be.
  assert.equal(
    residentModelMatchesPick(
      {
        active_model: "/mnt/disks/models/Model.gguf",
        model_identifier: "/mnt/disks/models/Model.gguf",
        gguf_variant: Q,
      },
      { id: "/mnt/disks/models/model.gguf" },
    ),
    false,
  );
});
