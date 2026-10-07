// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const S = await import(
  "../src/features/settings/components/audio-api-snippets.ts"
);
type Model = Parameters<typeof S.pickAudioApiModel>[0][number];

const BASE = "http://127.0.0.1:8888";
const KEY = "sk-unsloth-abc";
const tts = (id: string, loaded = false): Model => ({
  id,
  loaded,
  task: "text-to-speech",
});
const asr = (id: string, loaded = false): Model => ({
  id,
  loaded,
  task: "automatic-speech-recognition",
});
const cpp = (name: string) => `audio-cpp/audio.cpp-gguf/${name}`;

const PATHS: Record<string, string[]> = {
  speak: ["/v1/audio/speech"],
  clone: ["/v1/audio/inputs", "/v1/audio/voices", "/v1/audio/speech"],
  transcribe: ["/v1/audio/transcriptions"],
  separate: ["/v1/audio/inputs", "/v1/audio/run"],
  convert: ["/v1/audio/inputs", "/v1/audio/run"],
  music: ["/v1/audio/run"],
  edit: ["/v1/audio/inputs", "/v1/audio/run"],
};

const EXAMPLES = Object.keys(
  PATHS,
) as (keyof typeof S.AUDIO_API_PLACEHOLDER_MODELS)[];
const VARIANTS = [
  { lang: "curl", os: "unix" },
  { lang: "curl", os: "windows" },
  { lang: "python", os: "unix" },
  { lang: "javascript", os: "unix" },
] as const;

test("every example names its routes, the server, the key and the model", () => {
  for (const example of EXAMPLES) {
    for (const { lang, os } of VARIANTS) {
      const model = S.AUDIO_API_PLACEHOLDER_MODELS[example];
      const code = S.buildAudioApiSnippet(example, {
        base: BASE,
        apiKey: KEY,
        model,
        lang,
        os,
      });
      const where = `${example} ${lang} ${os}`;
      for (const path of PATHS[example]) {
        const call =
          lang === "curl"
            ? path
            : path.replace(
                /^\/v1\/audio\/(speech|transcriptions)$/,
                "client.audio.$1",
              );
        assert.ok(code.includes(call), `${where}: ${call}`);
      }
      assert.ok(code.includes(BASE), `${where}: base`);
      assert.ok(code.includes(KEY), `${where}: key`);
      assert.ok(code.includes(model), `${where}: model`);
    }
  }
});

test("Windows curl runs curl.exe with backtick continuations, bash with backslashes", () => {
  const input = { base: BASE, apiKey: KEY, model: cpp("HTDemucs-GGUF") };
  const win = S.buildAudioApiSnippet("separate", {
    ...input,
    lang: "curl",
    os: "windows",
  });
  assert.match(win, /curl\.exe /);
  assert.match(win, / `\n/);
  assert.ok(!/ \\\n/.test(win));
  assert.match(win, /"input_id": "\$\(\$source\.id\)"/);
  const unix = S.buildAudioApiSnippet("separate", {
    ...input,
    lang: "curl",
    os: "unix",
  });
  assert.match(unix, / \\\n/);
  assert.ok(!unix.includes("curl.exe"));
  assert.match(unix, /jq -n --arg source "\$SOURCE"/);
});

test("model ids and text are escaped for each language", () => {
  const model = `odd"name'$x`;
  const input = { base: BASE, apiKey: KEY, model };
  const python = S.buildAudioApiSnippet("speak", {
    ...input,
    lang: "python",
    os: "unix",
  });
  assert.ok(python.includes(`model=${JSON.stringify(model)}`));
  const js = S.buildAudioApiSnippet("speak", {
    ...input,
    lang: "javascript",
    os: "unix",
  });
  assert.ok(js.includes(`model: ${JSON.stringify(model)}`));
  const bash = S.buildAudioApiSnippet("speak", {
    ...input,
    lang: "curl",
    os: "unix",
  });
  const quoted = bash.slice(
    bash.indexOf("-d '") + 4,
    bash.indexOf("' \\\n  -o"),
  );
  assert.equal(JSON.parse(quoted.replace(/'\\''/g, "'")).model, model);
  const ps = S.buildAudioApiSnippet("speak", {
    ...input,
    lang: "curl",
    os: "windows",
  });
  assert.ok(ps.includes('"model": "odd\\"name\'`$x"'));
});

test("the Edit body follows the model's family", () => {
  const input = {
    base: BASE,
    apiKey: KEY,
    lang: "python",
    os: "unix",
  } as const;
  const dots = S.buildAudioApiSnippet("edit", {
    ...input,
    model: cpp("DotTTS-Edit-GGUF"),
  });
  assert.ok(dots.includes('<sub targ=\\"red\\">brown</sub>'));
  const firered = S.buildAudioApiSnippet("edit", {
    ...input,
    model: cpp("FireRedAudio-GGUF"),
  });
  assert.ok(firered.includes("Replace 'brown' with 'red'."));
  const vevo = S.buildAudioApiSnippet("edit", {
    ...input,
    model: cpp("Vevo2-GGUF"),
  });
  assert.ok(!vevo.includes("markup") && !vevo.includes("instructions"));
});

test("timestamps are asked for only where the model returns them", () => {
  assert.ok(S.transcribeWithTimestamps(cpp("Parakeet-TDT-0.6B-v3-GGUF")));
  assert.ok(S.transcribeWithTimestamps(cpp("MOSS-Transcribe-Diarize-GGUF")));
  assert.ok(!S.transcribeWithTimestamps(cpp("Qwen3-ASR-0.6B-GGUF")));
  assert.ok(!S.transcribeWithTimestamps("openai/whisper-large-v3-turbo"));
  const input = { base: BASE, apiKey: KEY, lang: "curl", os: "unix" } as const;
  const timed = S.buildAudioApiSnippet("transcribe", {
    ...input,
    model: cpp("Parakeet-TDT-0.6B-v3-GGUF"),
  });
  assert.ok(timed.includes("timestamp_granularities[]=segment"));
  const plain = S.buildAudioApiSnippet("transcribe", {
    ...input,
    model: "openai/whisper-large-v3-turbo",
  });
  assert.ok(
    !plain.includes("timestamp_granularities") &&
      !plain.includes("verbose_json"),
  );
});

test("each example picks a downloaded model that can run it, loaded first", () => {
  const models = [
    tts(cpp("ACE-Step1.5-GGUF")),
    tts(cpp("Qwen3-TTS-12Hz-0.6B-Base-GGUF")),
    tts(cpp("Kokoro-82M-GGUF")),
    tts(cpp("VoxCPM2-GGUF"), true),
    tts(cpp("RVC-GGUF")),
    tts(cpp("DotTTS-Edit-GGUF")),
    asr("openai/whisper-small"),
    asr(cpp("Parakeet-TDT-0.6B-v3-GGUF"), true),
  ];
  assert.equal(S.pickAudioApiModel(models, "speak"), cpp("VoxCPM2-GGUF"));
  assert.equal(S.pickAudioApiModel(models, "clone"), cpp("VoxCPM2-GGUF"));
  assert.equal(S.pickAudioApiModel(models, "convert"), cpp("RVC-GGUF"));
  assert.equal(S.pickAudioApiModel(models, "music"), cpp("ACE-Step1.5-GGUF"));
  assert.equal(S.pickAudioApiModel(models, "edit"), cpp("DotTTS-Edit-GGUF"));
  assert.equal(
    S.pickAudioApiModel(models, "transcribe"),
    cpp("Parakeet-TDT-0.6B-v3-GGUF"),
  );
  assert.equal(
    S.pickAudioApiModel(
      [
        tts(cpp("Breeze-TTS-2-GGUF")),
        tts(cpp("Kokoro-82M-GGUF")),
        tts(cpp("Chatterbox-Turbo-GGUF")),
      ],
      "speak",
    ),
    cpp("Kokoro-82M-GGUF"),
  );
  assert.equal(
    S.pickAudioApiModel(
      [asr(cpp("Qwen3-ASR-0.6B-GGUF")), asr(cpp("Parakeet-TDT-0.6B-v3-GGUF"))],
      "transcribe",
    ),
    cpp("Parakeet-TDT-0.6B-v3-GGUF"),
  );
  assert.equal(S.pickAudioApiModel(models, "separate"), null);
});

test("music and clone-only models never speak, and an unknown repo uses the family hints", () => {
  const models = [
    tts(cpp("ACE-Step1.5-GGUF")),
    tts(cpp("Qwen3-TTS-12Hz-0.6B-Base-GGUF")),
  ];
  assert.equal(S.pickAudioApiModel(models, "speak"), null);
  assert.equal(
    S.pickAudioApiModel([tts("someone/IndexTTS2-GGUF")], "clone"),
    "someone/IndexTTS2-GGUF",
  );
  assert.equal(
    S.pickAudioApiModel([tts("someone/IndexTTS2-GGUF")], "speak"),
    null,
  );
  assert.equal(
    S.pickAudioApiModel([tts("unsloth/orpheus-3b-0.1-ft")], "speak"),
    "unsloth/orpheus-3b-0.1-ft",
  );
  assert.equal(S.pickAudioApiModel([], "transcribe"), null);
});

test("a page's model is used only where it can run the example", () => {
  const listed = [
    tts(cpp("Kokoro-82M-GGUF")),
    tts("unsloth/orpheus-3b-0.1-ft"),
  ];
  assert.ok(!S.audioApiModelFits(listed, cpp("HTDemucs-GGUF"), "separate"));
  assert.ok(!S.audioApiModelFits(listed, cpp("HTDemucs-GGUF"), "speak"));
  assert.ok(!S.audioApiModelFits(listed, cpp("Kokoro-82M-GGUF"), "clone"));
  assert.ok(S.audioApiModelFits(listed, "unsloth/orpheus-3b-0.1-ft", "speak"));
  assert.ok(!S.audioApiModelFits(listed, "unsloth/Qwen3-8B-GGUF", "speak"));
  assert.ok(S.audioApiModelFits(null, cpp("Kokoro-82M-GGUF"), "speak"));
  assert.ok(!S.audioApiModelFits(null, "unsloth/Qwen3-8B-GGUF", "speak"));
});

test("the Convert and Music bodies follow what the model accepts", () => {
  const input = {
    base: BASE,
    apiKey: KEY,
    lang: "python",
    os: "unix",
  } as const;
  const rvc = S.buildAudioApiSnippet("convert", {
    ...input,
    model: cpp("RVC-GGUF"),
  });
  assert.ok(!rvc.includes("voice.wav") && !rvc.includes('"target"'));
  assert.ok(rvc.includes('"voice": "default"'));
  const seed = S.buildAudioApiSnippet("convert", {
    ...input,
    model: cpp("SeedVC-MLX-GGUF"),
  });
  assert.ok(seed.includes('"target"') && seed.includes("voice.wav"));
  for (const name of ["Stable-Audio-3-Small-SFX-GGUF", "ControlFoley-GGUF"]) {
    const sfx = S.buildAudioApiSnippet("music", { ...input, model: cpp(name) });
    assert.ok(sfx.includes('"mode": "sfx"') && !sfx.includes("lyrics"), name);
  }
  const song = S.buildAudioApiSnippet("music", {
    ...input,
    model: cpp("Stable-Audio-3-Small-Music-GGUF"),
  });
  assert.ok(song.includes('"mode": "song"'));
});

test("the JavaScript run examples wait past fetch's 5 minute limit", () => {
  const input = {
    base: BASE,
    apiKey: KEY,
    lang: "javascript",
    os: "unix",
  } as const;
  for (const example of ["separate", "convert", "music", "edit"] as const) {
    const code = S.buildAudioApiSnippet(example, {
      ...input,
      model: S.AUDIO_API_PLACEHOLDER_MODELS[example],
    });
    assert.ok(
      code.includes("setGlobalDispatcher(new Agent({ headersTimeout: 0"),
      example,
    );
  }
  const speak = S.buildAudioApiSnippet("speak", {
    ...input,
    model: cpp("Kokoro-82M-GGUF"),
  });
  assert.ok(!speak.includes("undici"));
});

test("the key is the revealed one, else the placeholder this server admits", () => {
  const local = {
    base: "http://127.0.0.1:8888",
    tunnel: false,
    exposure: null,
  };
  assert.equal(
    S.audioApiKey("sk-unsloth-real", { ...local, scope: "full" }),
    "sk-unsloth-real",
  );
  assert.equal(S.audioApiKey(null, { ...local, scope: "full" }), "not-needed");
  assert.equal(
    S.audioApiKey(null, { ...local, scope: "inference" }),
    "sk-unsloth-YOUR_KEY",
  );
  assert.equal(
    S.audioApiKey(null, { ...local, scope: "off" }),
    "sk-unsloth-YOUR_KEY",
  );
  assert.equal(
    S.audioApiKey(null, { ...local, tunnel: true, scope: "full" }),
    "sk-unsloth-YOUR_KEY",
  );
  assert.equal(
    S.audioApiKey(null, {
      base: "http://192.168.1.20:8888",
      tunnel: false,
      exposure: null,
      scope: "full",
    }),
    "sk-unsloth-YOUR_KEY",
  );
});

test("picking a language keeps the chat card's variant of it", () => {
  assert.equal(S.audioLangToStore("pythonTools", "python"), null);
  assert.equal(S.audioLangToStore("javascriptAdvanced", "javascript"), null);
  assert.equal(S.audioLangToStore("pythonTools", "curl"), "curl");
  assert.equal(S.audioLangToStore(null, "python"), "python");
  assert.equal(S.langFromStored("curlTools"), "curl");
});

test("each Audio page opens the example for its workflow", () => {
  assert.deepEqual(S.audioApiExampleFor("speak"), { tab: "speak", run: null });
  assert.deepEqual(S.audioApiExampleFor("transcribe"), {
    tab: "transcribe",
    run: null,
  });
  assert.deepEqual(S.audioApiExampleFor("separate"), {
    tab: "workflows",
    run: "separate",
  });
  assert.deepEqual(S.audioApiExampleFor("edit"), {
    tab: "workflows",
    run: "edit",
  });
});

test("the workflows a row advertises win over the name hints", () => {
  const music = {
    id: "MiniMaxAI/MiniMax-Music3",
    loaded: true,
    task: "text-to-speech" as const,
    workflows: ["music"],
  };
  const separator = {
    id: cpp("BS-RoFormer-GGUF"),
    loaded: true,
    task: "audio-to-audio" as const,
    workflows: ["separate"],
  };
  assert.equal(S.pickAudioApiModel([music], "speak"), null);
  assert.equal(S.pickAudioApiModel([music], "music"), music.id);
  assert.equal(S.pickAudioApiModel([separator], "separate"), separator.id);
  assert.ok(!S.audioApiModelFits([music], music.id, "speak"));
  assert.ok(S.audioApiModelFits([separator], separator.id, "separate"));
  assert.ok(!S.audioApiModelFits([], cpp("HTDemucs-GGUF"), "separate"));
});
