// SPDX-License-Identifier: AGPL-3.0-only
import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { stripTypeScriptTypes } from "node:module";

function harness() {
  const source = readFileSync(
    new URL("../src/features/chat/tool-isolation.ts", import.meta.url),
    "utf8",
  );
  const code = source
    .slice(
      source.indexOf("let capabilityRequest:"),
      source.indexOf("export class ToolIsolationRequestError"),
    )
    .replaceAll("export ", "");
  const calls: any[] = [];
  const timers = new Set<() => void>();
  const api = runInNewContext(
    stripTypeScriptTypes(
      code +
        `
({fetchToolIsolationCapability,fetchLimitedToolGrant,cancelLimitedToolGrant,cancelToolIsolationCheck,setupWindowsToolIsolation})`,
    ),
    {
      AbortController,
      Error,
      setTimeout: (fn: () => void) => {
        timers.add(fn);
        return fn;
      },
      clearTimeout: (fn: () => void) => timers.delete(fn),
      parseCapability: (x: unknown) => x,
      parseGrant: (x: unknown) => x,
      responseError: (_body: unknown, fallback: string) => fallback,
      authFetch: (path: string, options: RequestInit) =>
        new Promise((resolve, reject) => {
          calls.push({
            path,
            options,
            resolve: (body: unknown) =>
              resolve({ ok: true, json: async () => body }),
          });
          options.signal?.addEventListener("abort", () =>
            reject(options.signal?.reason),
          );
        }),
    },
  );
  return { api, calls, timers };
}

test("capability consumers share the actual HTTP request and explicit refresh is serialized", async () => {
  const h = harness();
  const a = h.api.fetchToolIsolationCapability(true);
  const b = h.api.fetchToolIsolationCapability();
  assert.equal(h.calls.length, 1);
  assert.equal(
    h.calls[0].path,
    "/api/inference/tool-isolation/capability?refresh=true",
  );
  h.calls[0].resolve({ probe_generation: "g" });
  assert.deepEqual(await a, await b);
  assert.equal(h.timers.size, 0);
});

test("Limited cancel aborts the serialized request without returning a late grant", async () => {
  const h = harness();
  const pending = h.api.fetchLimitedToolGrant("page", "generation");
  const rejected = assert.rejects(pending);
  assert.deepEqual(JSON.parse(h.calls[0].options.body), {
    ui_session_id: "page",
    probe_generation: "generation",
  });
  h.api.cancelLimitedToolGrant();
  assert.equal(h.calls[0].options.signal.aborted, true);
  h.calls[0].resolve({ grant: "late" });
  await rejected;
  assert.equal(h.timers.size, 0);
});

test("timeout rejects permission and a later manual request can recover", async () => {
  const h = harness();
  const pending = h.api.fetchLimitedToolGrant("page", "g");
  const rejected = assert.rejects(pending, /timed out/);
  for (const timer of h.timers) timer();
  await rejected;
  const retry = h.api.fetchLimitedToolGrant("page", "g");
  h.calls[1].resolve({ grant: "explicit-retry" });
  assert.equal((await retry).grant, "explicit-retry");
});

test("setup sends only explicit confirmation, never a user command or runtime override", async () => {
  const h = harness();
  const pending = h.api.setupWindowsToolIsolation();
  assert.equal(h.calls[0].path, "/api/inference/tool-isolation/windows-setup");
  assert.deepEqual(JSON.parse(h.calls[0].options.body), { confirm: true });
  h.calls[0].resolve({ status: "installed", message: "done" });
  assert.equal((await pending).status, "installed");
  assert.equal(h.calls.length, 1);
});

function setupDialogHarness(values: Record<string, unknown>) {
  const source = readFileSync(new URL("../src/features/chat/permission-mode-select.tsx", import.meta.url), "utf8");
  const code = source.slice(source.indexOf("function WindowsToolIsolationSetup()"),source.indexOf("function ToolIsolationMenuSection("));
  let state: any = {
    windowsToolIsolationSetupOpen:false, windowsToolIsolationSetupRequested:true,
    toolIsolationCapability:{environment:"win32",available:false}, toolExecutionMode:"os_isolation_required",
    refreshToolIsolationCapability:async()=>{},clearLimitedToolGrant:()=>{},...values,
  };
  const store:any = (select:any)=>select(state);
  store.setState = (patch:any)=>{state={...state,...patch};};
  store.getState = ()=>state;
  const context:any = {
    useChatRuntimeStore:store,useState:(value:any)=>[value,()=>{}],useEffect:(fn:any)=>fn(),
    React:{createElement:(type:any,props:any,...children:any[])=>({type,props,children})},
  };
  for (const name of ["AlertDialog","AlertDialogContent","AlertDialogHeader","AlertDialogTitle","AlertDialogDescription","AlertDialogFooter","AlertDialogCancel","AlertDialogAction"]) context[name]=name;
  const compiled=ts.transpileModule(code,{compilerOptions:{jsx:ts.JsxEmit.React,target:ts.ScriptTarget.ES2022}}).outputText;
  const render=runInNewContext(compiled+"\nWindowsToolIsolationSetup;",context);
  const rendered = render();
  state.rendered = rendered;
  return state;
}

test("turning on tools offers Windows setup directly only when needed",()=>{
  assert.equal(setupDialogHarness({}).windowsToolIsolationSetupOpen,true);
  for(const values of [
    {toolIsolationCapability:{environment:"win32",available:true}},
    {toolIsolationCapability:{environment:"linux",available:false}},
    {toolExecutionMode:"limited"}, {toolExecutionMode:"full"},
    {windowsToolIsolationSetupRequested:false},
  ]) assert.equal(setupDialogHarness(values).windowsToolIsolationSetupOpen,false);
});

test("a remembered setup conflict offers explicit repair after reopening", () => {
  const state = setupDialogHarness({toolIsolationCapability: {
    environment: "win32", available: false, reason_code: "setup_conflict",
    remediation: "Repair replaces the existing sandbox network filters.",
  }});
  const tree = JSON.stringify(state.rendered);
  assert.match(tree, /Repair existing setup/);
  assert.match(tree, /Repair replaces the existing sandbox network filters/);
});

test("technical details render only for errors and omit the Windows limitation list",()=>{
  const source=readFileSync(new URL("../src/features/chat/permission-mode-select.tsx",import.meta.url),"utf8");
  const code=source.slice(source.indexOf("function ToolIsolationDetailsDialog("),source.indexOf("function WindowsToolIsolationSetup()"));
  let state:any={toolIsolationCapability:{available:true},toolIsolationError:null};
  const context:any={useChatRuntimeStore:(select:any)=>select(state),React:{createElement:(type:any,props:any,...children:any[])=>({type,props,children})}};
  for(const name of ["Dialog","DialogContent","DialogHeader","DialogTitle","DialogDescription"])context[name]=name;
  const compiled=ts.transpileModule(code,{compilerOptions:{jsx:ts.JsxEmit.React,target:ts.ScriptTarget.ES2022}}).outputText;
  const render=runInNewContext(compiled+"\nToolIsolationDetailsDialog;",context);
  assert.equal(render({open:true,onOpenChange:()=>{}}),null);
  state={toolIsolationCapability:{available:false,reason:"probe failed",diagnostic:{code:"probe_failed",stage:"probe"},limitations:["srt_windows_system_dns_unfenced","srt_windows_shared_account_grants"]}};
  const tree=JSON.stringify(render({open:true,onOpenChange:()=>{}}));
  assert.match(tree,/Tool isolation error/);
  assert.match(tree,/probe_failed/);
  assert.doesNotMatch(tree,/system_dns|shared_account|Limitations/);
});

test("replacing existing setup requires a distinct explicit request", async()=>{
  const h=harness();const pending=h.api.setupWindowsToolIsolation(true);
  assert.deepEqual(JSON.parse(h.calls[0].options.body),{confirm:true,repair_existing:true});
  h.calls[0].resolve({status:"installed",message:"done"});await pending;
});
