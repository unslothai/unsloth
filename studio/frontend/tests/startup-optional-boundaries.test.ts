// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";
import { researchStatusLabel } from "../src/features/chat/components/research-status-label.ts";
const source=(path:string)=>readFileSync(new URL("../src/"+path,import.meta.url),"utf8");
test("research labels retain every runtime state without importing editor UI",()=>{
  assert.deepEqual(["planning","awaiting_approval","queued","running","paused","cancelling","cancelled","completed","failed"].map(s=>researchStatusLabel(s as Parameters<typeof researchStatusLabel>[0])),
    ["Planning","Review plan","Queued","Researching","Paused","Stopping","Cancelled","Complete","Failed"]);
  assert.ok(!source("features/chat/components/research-message.tsx").includes('from "./research-activity-panel"'));
});
test("research shells keep close and failure controls while editor loading is deferred",()=>{
  const text=source("features/chat/components/research-activity-mount.tsx");
  assert.ok(text.includes('lazy(() => import("./research-activity-panel")'));
  assert.ok(text.includes("LazyImportBoundary"));assert.ok(text.includes("SheetTitle"));
  assert.ok(text.includes("props.onClose"));assert.ok(text.includes("{open ? <ResearchActivityPanel"));
});
test("project controller stays mounted after first activation",()=>{
  const text=source("features/chat/components/new-project-dialog-mount.tsx");
  assert.ok(text.includes('lazy(() => import("./new-project-dialog")'));
  assert.ok(text.includes("if (!activated && !props.open) return null"));
  assert.ok(text.includes("<ProjectDialog {...props} />"));
  assert.ok(!text.includes("props.open ? <ProjectDialog"));
});
test("YAML codec is absent from startup barrel and loaded only in protected actions",()=>{
  assert.ok(!source("features/training/index.ts").includes('./lib/yaml-config'));
  const text=source("features/studio/wizard/config-actions.tsx");
  assert.equal((text.match(/await import\("@\/features\/training\/lib\/yaml-config"\)/g)??[]).length,2);
  assert.ok(text.includes("useTrainingConfigStore.getState() !== observed"));
  assert.ok(text.indexOf("const state = useTrainingConfigStore.getState()") < text.indexOf("const { serializeConfigToYaml } = await import"));
});
