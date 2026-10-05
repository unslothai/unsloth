// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { parseMusicCapabilities } = await import(
  "../src/features/audio/music/music-types.ts"
);
const {
  buildMusicRunRequest,
  durationLabel,
  effectiveMusicMode,
  musicModeHint,
  musicModeLabel,
  insertSectionTag,
  instrumentalChoice,
  lyricsShown,
  musicBlocker,
  musicDurationFor,
  reloadNotice,
  sectionTags,
  variationsFor,
} = await import("../src/features/audio/music/music-policy.ts");
const { buildAudioRunBody } = await import(
  "../src/features/audio/audio-run-request.ts"
);
const { DEFAULT_MUSIC_DRAFTS } = await import(
  "../src/features/audio/stores/audio-music-store.ts"
);

const STABLE = {
  modes: [
    {
      id: "song",
      lyrics: "unused",
      description: "required",
      instrumental: "always",
      duration: { min: 1, max: 120, default: 30, approximate: false },
      variations: { max: 4, how: "batch", loaded: 1 },
    },
    {
      id: "edit",
      actions: ["inpaint", "restyle"],
      max_ranges: 8,
      max_source_s: 120,
    },
  ],
};

const MINIMAX = {
  modes: [
    {
      id: "song",
      lyrics: "required",
      description: "required",
      instrumental: "never",
      section_case: "lower",
      duration: { min: 5, max: 240, default: 30, approximate: true },
      variations: null,
    },
  ],
};

const YUE = {
  modes: [
    {
      id: "song",
      lyrics: "optional",
      description: "required",
      instrumental: "toggle",
      section_case: "title",
      duration: { min: 5, max: 240, default: 30, approximate: true },
    },
  ],
};

function caps(raw: unknown) {
  const parsed = parseMusicCapabilities(raw);
  assert.ok(parsed);
  return parsed;
}

const PATHLIKE = /"(path|voice_ref|file|audio_path|trim)"\s*:/;

// biome-ignore lint/suspicious/noExplicitAny: fixtures stand in for any mode rule and draft.
function inputs(rule: any, overrides: any = {}) {
  return {
    rule,
    description: "",
    lyrics: "",
    ...structuredClone(DEFAULT_MUSIC_DRAFTS),
    ...overrides,
  };
}

test("status parsing keeps known modes and drops malformed ones", () => {
  assert.equal(parseMusicCapabilities(null), null);
  assert.equal(parseMusicCapabilities({ modes: [] }), null);
  assert.equal(parseMusicCapabilities({ modes: [{ id: "karaoke" }] }), null);
  const parsed = caps({
    modes: [...STABLE.modes, { id: "song" }, { id: "bogus" }],
  });
  assert.deepEqual(
    parsed.modes.map((mode) => mode.id),
    ["song", "edit"],
  );
  assert.equal(parsed.modes[0].variations?.how, "batch");
  assert.deepEqual(parsed.modes[1].actions, ["inpaint", "restyle"]);
  const single = caps({
    modes: [{ id: "sfx", variations: { max: 1, how: "batch" } }],
  });
  assert.equal(single.modes[0].variations, null);
});

test("the picked mode wins only when the model has it", () => {
  const stable = caps(STABLE);
  assert.equal(effectiveMusicMode(stable, "edit").id, "edit");
  assert.equal(effectiveMusicMode(stable, "sfx").id, "song");
});

test("section tags follow the model's casing", () => {
  assert.equal(sectionTags("lower")[1].tag, "[verse]");
  assert.equal(sectionTags("title")[1].tag, "[Verse]");
  assert.equal(sectionTags("title")[2].tag, "[Pre-Chorus]");
  assert.equal(sectionTags("lower")[2].tag, "[pre-chorus]");
});

test("a section tag lands on its own line at the cursor", () => {
  assert.deepEqual(insertSectionTag("", 0, "[verse]"), {
    text: "[verse]\n",
    cursor: 8,
  });
  const after = insertSectionTag("line one", 8, "[chorus]");
  assert.equal(after.text, "line one\n\n[chorus]\n");
  assert.equal(after.text.slice(0, after.cursor), "line one\n\n[chorus]\n");
});

test("instrumental: toggle, always and never", () => {
  const song = (raw: unknown) => caps(raw).modes[0];
  assert.deepEqual(instrumentalChoice(song(YUE), true).value, true);
  assert.equal(instrumentalChoice(song(STABLE), false).value, true);
  assert.equal(instrumentalChoice(song(MINIMAX), true).value, false);
  assert.equal(instrumentalChoice(song(MINIMAX), true).enabled, false);
  assert.ok(instrumentalChoice(song(MINIMAX), true).reason);
  assert.equal(lyricsShown(song(STABLE), true), false);
  assert.equal(lyricsShown(song(YUE), true), false);
  assert.equal(lyricsShown(song(YUE), false), true);
});

test("length stays inside the model's bounds", () => {
  const rule = caps(MINIMAX).modes[0];
  assert.equal(musicDurationFor(rule, null), 30);
  assert.equal(musicDurationFor(rule, 1), 5);
  assert.equal(musicDurationFor(rule, 999), 240);
});

test("blocker reasons are plain and per mode", () => {
  const minimax = caps(MINIMAX).modes[0];
  assert.match(String(musicBlocker(inputs(minimax))), /Describe the music/);
  assert.match(
    String(musicBlocker(inputs(minimax, { description: "pop" }))),
    /Write the lyrics/,
  );
  assert.equal(
    musicBlocker(inputs(minimax, { description: "pop", lyrics: "la" })),
    null,
  );
  const yue = caps(YUE).modes[0];
  const instrumental = inputs(yue, {
    description: "pop",
    song: { ...DEFAULT_MUSIC_DRAFTS.song, instrumental: true },
  });
  assert.equal(musicBlocker(instrumental), null);
  const sfx = caps({ modes: [{ id: "sfx" }] }).modes[0];
  assert.match(String(musicBlocker(inputs(sfx))), /Describe the sound/);
  const edit = caps(STABLE).modes[1];
  assert.equal(
    musicBlocker(inputs(edit, { editProblem: "Pick a clip." })),
    "Pick a clip.",
  );
});

test("the reload notice shows only when a batch needs more than the server holds", () => {
  const rule = caps(STABLE).modes[0];
  assert.equal(reloadNotice(rule, 1, "Stable Audio"), null);
  assert.match(
    String(reloadNotice(rule, 3, "Stable Audio")),
    /Reloads Stable Audio/,
  );
  const loaded = caps({
    modes: [
      { ...STABLE.modes[0], variations: { max: 4, how: "batch", loaded: 4 } },
    ],
  }).modes[0];
  assert.equal(reloadNotice(loaded, 3, "Stable Audio"), null);
  const sequential = caps({
    modes: [{ id: "sfx", variations: { max: 4, how: "sequential" } }],
  }).modes[0];
  assert.equal(reloadNotice(sequential, 4, "ControlFoley"), null);
  assert.equal(variationsFor(sequential, 9), 4);
});

test("a song request carries only music keys and no paths", () => {
  const rule = caps(YUE).modes[0];
  const request = buildMusicRunRequest({
    ...inputs(rule, { description: " dream pop ", lyrics: "[Verse]\nla" }),
    options: { cot: "melody" },
    seed: 7,
  });
  const body = buildAudioRunBody(request);
  assert.deepEqual(body, {
    workflow: "music",
    text: "dream pop",
    options: { cot: "melody" },
    seed: 7,
    mode: "song",
    lyrics: "[Verse]\nla",
    duration_s: 30,
  });
  assert.doesNotMatch(JSON.stringify(body), PATHLIKE);
});

test("instrumental songs send the switch and no lyrics", () => {
  const rule = caps(YUE).modes[0];
  const body = buildAudioRunBody(
    buildMusicRunRequest(
      inputs(rule, {
        description: "ambient",
        lyrics: "[Verse]\nkept for later",
        song: { ...DEFAULT_MUSIC_DRAFTS.song, instrumental: true },
      }),
    ),
  );
  assert.equal(body.instrumental, true);
  assert.equal("lyrics" in body, false);
});

test("variations go only to models that make them", () => {
  const stable = caps(STABLE).modes[0];
  const body = buildAudioRunBody(
    buildMusicRunRequest(
      inputs(stable, {
        description: "house",
        lyrics: "never sent",
        song: { ...DEFAULT_MUSIC_DRAFTS.song, variations: 3 },
      }),
    ),
  );
  assert.equal(body.variations, 3);
  assert.equal("lyrics" in body, false);
  assert.equal("instrumental" in body, false);
  const minimax = caps(MINIMAX).modes[0];
  const single = buildAudioRunBody(
    buildMusicRunRequest(
      inputs(minimax, {
        description: "pop",
        lyrics: "la",
        song: { ...DEFAULT_MUSIC_DRAFTS.song, variations: 3 },
      }),
    ),
  );
  assert.equal("variations" in single, false);
});

test("an edit sends the source by id, its ranges and the action's own value", () => {
  const edit = caps(STABLE).modes[1];
  const draft = {
    ...DEFAULT_MUSIC_DRAFTS.edit,
    source: { kind: "clip", id: "c".repeat(32), name: "take", durationS: 20 },
    action: "inpaint",
    ranges: [
      { start_s: 1, end_s: 2 },
      { start_s: 5, end_s: 6.5 },
    ],
    strength: 0.9,
    prompt: "brighter",
  };
  const body = buildAudioRunBody(
    buildMusicRunRequest(inputs(edit, { edit: draft })),
  );
  assert.deepEqual(body, {
    workflow: "music",
    text: "brighter",
    mode: "edit",
    inputs: { source: { clip_id: "c".repeat(32) } },
    edit: {
      action: "inpaint",
      ranges: [
        { start_s: 1, end_s: 2 },
        { start_s: 5, end_s: 6.5 },
      ],
    },
  });
  const restyle = buildAudioRunBody(
    buildMusicRunRequest(
      inputs(edit, { edit: { ...draft, action: "restyle" } }),
    ),
  );
  assert.deepEqual(restyle.edit, { action: "restyle", strength: 0.9 });
  // An untouched slider sends the value it shows, not the runtime's 1.0.
  const untouched = (action: "restyle" | "cover") =>
    buildAudioRunBody(
      buildMusicRunRequest(
        inputs(edit, {
          edit: { ...draft, action, ranges: [], strength: null },
        }),
      ),
    ).edit;
  assert.deepEqual(untouched("restyle"), { action: "restyle", strength: 0.45 });
  assert.deepEqual(untouched("cover"), { action: "cover", strength: 0.5 });
  const voice = buildAudioRunBody(
    buildMusicRunRequest(
      inputs(edit, {
        edit: {
          ...draft,
          source: { kind: "voice", id: "v", name: "x", durationS: 3 },
        },
      }),
    ),
  );
  assert.equal("inputs" in voice, false);
  assert.doesNotMatch(JSON.stringify(body), PATHLIKE);
});

test("extend sends its seconds and continue its length; repaint keeps a range past the end", () => {
  const ace = caps({
    modes: [
      {
        id: "edit",
        actions: ["repaint", "extend", "cover", "continue"],
        max_ranges: 1,
        duration: { min: 5, max: 240, default: 60 },
      },
    ],
  }).modes[0];
  const source = {
    kind: "input",
    id: "i".repeat(32),
    name: "a.wav",
    durationS: 30,
  };
  const extend = buildAudioRunBody(
    buildMusicRunRequest(
      inputs(ace, {
        edit: {
          ...DEFAULT_MUSIC_DRAFTS.edit,
          source,
          action: "extend",
          extendS: 12,
        },
      }),
    ),
  );
  assert.deepEqual(extend.edit, { action: "extend", extend_s: 12 });
  assert.equal("duration_s" in extend, false);
  const cont = buildAudioRunBody(
    buildMusicRunRequest(
      inputs(ace, {
        edit: { ...DEFAULT_MUSIC_DRAFTS.edit, source, action: "continue" },
      }),
    ),
  );
  assert.equal(cont.duration_s, 60);
  const repaint = buildAudioRunBody(
    buildMusicRunRequest(
      inputs(ace, {
        edit: {
          ...DEFAULT_MUSIC_DRAFTS.edit,
          source,
          action: "repaint",
          ranges: [{ start_s: 25, end_s: 40 }],
        },
      }),
    ),
  );
  assert.deepEqual((repaint.edit as { ranges: unknown }).ranges, [
    { start_s: 25, end_s: 40 },
  ]);
});

test("clone and speak bodies never carry music keys", () => {
  const body = buildAudioRunBody({
    workflow: "speak",
    text: "Hi",
    music: { mode: "song", lyrics: "x", variations: 3 },
  });
  for (const key of ["mode", "lyrics", "variations", "edit", "duration_s"]) {
    assert.equal(key in body, false, key);
  }
});

test("instrumental-only models say so in the mode label and length wording", () => {
  const stable = caps(STABLE).modes[0];
  assert.equal(musicModeLabel(stable), "Instrumental");
  assert.match(musicModeHint(stable), /without vocals/);
  assert.equal(musicModeLabel(caps(YUE).modes[0]), "Song");
  assert.equal(durationLabel(stable), "Length (seconds)");
  assert.equal(
    durationLabel(caps(YUE).modes[0]),
    "Length (seconds, approximate)",
  );
});
