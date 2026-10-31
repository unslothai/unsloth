// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DeferredTextarea } from "@/components/deferred-textarea";
// DeferredTextarea under a deliberately slow, impure owner (the Images page in WebKitGTK). No backend.
/* eslint-disable react-hooks/purity, react-hooks/immutability, react-refresh/only-export-components */
import { useState } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

const SLOW_MS = Number(new URLSearchParams(location.search).get("slow") ?? 40);

declare global {
  interface Window {
    ownerRenders: number;
    ownerRendersInKey: number;
  }
}
window.ownerRenders = 0;
window.ownerRendersInKey = 0;
// Brackets each keystroke's input dispatch: the first listener to see it and the last.
let inKey = false;
document.addEventListener(
  "input",
  () => {
    inKey = true;
  },
  true,
);
window.addEventListener("input", () => {
  inKey = false;
});

function Owner() {
  const [prompt, setPrompt] = useState("seed prompt");
  const [generated, setGenerated] = useState<string[]>([]);
  window.ownerRenders += 1;
  if (inKey) window.ownerRendersInKey += 1;
  const until = performance.now() + SLOW_MS;
  while (performance.now() < until) {
  }
  return (
    <div className="flex max-w-xl flex-col gap-3 p-6">
      <DeferredTextarea
        aria-label="Instruction"
        value={prompt}
        onValueChange={setPrompt}
      />
      <output aria-label="Owner prompt">{prompt}</output>
      <button type="button" onClick={() => setGenerated((g) => [...g, prompt])}>
        Generate
      </button>
      <button type="button" onClick={() => setPrompt("from recall")}>
        Recall
      </button>
      <button
        type="button"
        onClick={() => setPrompt((p) => `${p}, transparent background`)}
      >
        Append
      </button>
      <output aria-label="Generated">{JSON.stringify(generated)}</output>
    </div>
  );
}

const root = document.getElementById("root");
if (root) createRoot(root).render(<Owner />);
