// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import path from "node:path";
import { pathToFileURL } from "node:url";

const frontendPath = path.resolve(process.argv[2]);
const root = path.resolve(process.argv[3]);
const { createServer, normalizePath } = await import(
  pathToFileURL(path.join(frontendPath, "node_modules/vite/dist/node/index.js"))
);
const frontend = normalizePath(frontendPath);
let state;
const reset = (mode = "normal") => {
  for (const res of state?.streams ?? []) res.end();
  for (const done of state?.waiting ?? []) done();
  state = {
    mode,
    uploads: [],
    jobs: {},
    docs: [],
    streams: [],
    waiting: [],
    eventRequests: [],
    polls: 0,
  };
};
reset();
const json = (res, body, status = 200) => {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
};
const read = async (req) => {
  const chunks = [];
  for await (const data of req) chunks.push(data);
  return Buffer.concat(chunks).toString("utf8");
};
const terminal = (status = "completed") => {
  for (const job of Object.values(state.jobs))
    Object.assign(job, {
      status,
      numChunks: 2,
      error: status === "failed" ? "simulated indexing failure" : null,
    });
  for (const doc of state.docs) Object.assign(doc, { status, numChunks: 2 });
};
const fixture = async (req, res, next) => {
  const url = new URL(req.url, "http://localhost").pathname;
  if (url === "/__scenario") {
    reset(JSON.parse(await read(req)).mode);
    return json(res, { ok: true });
  }
  if (url === "/__state")
    return json(res, {
      ...state,
      streams: state.streams.filter((r) => !r.destroyed).length,
      waiting: state.waiting.length,
    });
  if (url === "/__release-one") {
    state.waiting.shift()?.();
    return json(res, { ok: true });
  }
  if (url === "/__release") {
    terminal(state.mode.includes("fail") ? "failed" : "completed");
    for (const done of state.waiting.splice(0)) done();
    for (const stream of state.streams) stream.end();
    return json(res, { ok: true });
  }
  if (url.endsWith("/documents") && url.startsWith("/api/rag/threads/")) {
    const thread = url.split("/")[4];
    if (req.method === "GET")
      return json(res, {
        documents: state.docs.filter((d) => d.threadId === thread),
      });
    const body = await read(req);
    const filename = /filename="([^"]*)"/.exec(body)?.[1] ?? "report.txt";
    state.uploads.push({
      thread,
      filename,
      multipart: req.headers["content-type"],
    });
    if (state.mode === "reject")
      return json(res, { detail: "simulated upload rejection" }, 413);
    const id = state.mode.includes("server-duplicate")
      ? "doc-1"
      : `doc-${state.uploads.length}`;
    const jobId = `job-${id}`;
    if (!state.jobs[jobId]) {
      state.docs.push({
        id,
        threadId: thread,
        filename,
        status: "running",
        managed: false,
        numChunks: 0,
      });
      state.jobs[jobId] = {
        id: jobId,
        documentId: id,
        status: "running",
        progress: 0.4,
        stage: "captioning",
      };
    }
    if (state.mode.startsWith("delayed"))
      await new Promise((resolve) => state.waiting.push(resolve));
    return json(res, { documentId: id, jobId, filename });
  }
  if (url.startsWith("/api/rag/jobs/")) {
    const id = url.split("/")[4];
    const job = state.jobs[id];
    if (!job) return json(res, { detail: "missing job" }, 404);
    if (!url.endsWith("/events")) {
      state.polls++;
      return json(res, job);
    }
    state.eventRequests.push({ id, method: req.method });
    if (state.mode === "get-only" && req.method === "POST")
      return json(res, {}, 405);
    if (state.mode === "sse-unavailable") return json(res, {}, 503);
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
    });
    const payload = Buffer.from(
      'data: {"type":"progress","progress":0.4,"stage":"captioning","text":"東京 café"}\r\n\r\n',
    );
    for (let i = 0; i < payload.length; i += 7)
      res.write(payload.subarray(i, i + 7));
    if (state.mode === "early-end" || state.mode.startsWith("delayed"))
      return res.end();
    if (state.mode === "hold" || state.mode === "server-duplicate") {
      state.streams.push(res);
      return;
    }
    if (state.mode === "error") {
      terminal("failed");
      return res.end(
        'data: {"type":"error","error":"simulated indexing failure","stage":"error"}\n\n',
      );
    }
    terminal();
    res.end('data: {"type":"complete","num_chunks":2}\n\n');
    return;
  }
  next();
};

const entry = `
import React, {useCallback, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {useRagDocuments} from './src/features/rag/components/use-rag-documents';
import {listThreadDocuments} from './src/features/rag/api/rag-api';
window.errors = []; window.resolveCalls = 0; window.pageErrors = [];
window.addEventListener('error', event => window.pageErrors.push(event.message));
window.addEventListener('unhandledrejection', event => window.pageErrors.push(String(event.reason)));
let chosen = [];
function Uploads({scopeId, setScope}) {
  const lister = useCallback(() => scopeId ? listThreadDocuments(scopeId) : Promise.resolve([]), [scopeId]);
  const api = useRagDocuments(scopeId ? {type:'thread',threadId:scopeId} : null, lister);
  window.sim = { ...api, scopeId,
    uploadNames(names, failScope=false) {
      chosen=names.map(name => new File(['fixture document'], name, {lastModified:1}));
      return api.upload(chosen, failScope ? async()=>{window.resolveCalls++;throw Error('simulated chat failure')} : undefined);
    },
    reselect() { return api.upload(chosen, async()=>{window.resolveCalls++;throw Error('duplicate initialized chat')}); },
    materialize() { return api.upload([new File(['x'],'materialize.txt')], async()=>{setScope('thread'); return {type:'thread',threadId:'thread'};}); }
  };
  return React.createElement(React.Fragment, null,
    React.createElement('input', {id:'files',type:'file',multiple:true,onChange:e=>void api.upload(e.target.files)}),
    React.createElement('pre',{id:'state'},JSON.stringify({documents:api.documents,uploading:api.uploading,hasIndexing:api.hasIndexing,scopeId})));
}
function App() {
 const [scopeId,setScope]=useState(new URLSearchParams(location.search).has('empty')?null:'thread');
 const [mounted,setMounted]=useState(true);
 return React.createElement(React.Fragment,null,
 React.createElement('button',{id:'switch',onClick:()=>setScope('other')},'Switch'),
 React.createElement('button',{id:'unmount',onClick:()=>setMounted(false)},'Unmount'),
 mounted ? React.createElement(Uploads,{scopeId,setScope}) : React.createElement('div',{id:'gone'},'Unmounted'));
}
createRoot(document.getElementById('root')).render(React.createElement(React.StrictMode,null,React.createElement(App)));
`;
const stubs = {
  "@/features/auth": "export const authFetch=(url,init)=>fetch(url,init);",
  "@/lib/api-base": "export const apiUrl=url=>url;",
  "@/lib/toast":
    "export const toast={error:(message,details)=>window.errors.push({message,details}),info:message=>window.errors.push({message})};",
  "@/features/native-intents":
    "export const consumeNativePathToken=async()=>({nativePathLease:'fixture'});",
};
const server = await createServer({
  configFile: false,
  root: frontend,
  cacheDir: path.join(root, "vite-cache"),
  resolve: { alias: { "@": path.join(frontend, "src") } },
  server: {
    host: "127.0.0.1",
    port: 18948,
    strictPort: true,
    fs: { allow: [frontend] },
  },
  plugins: [
    {
      name: "upload-simulation",
      enforce: "pre",
      resolveId(id) {
        if (stubs[id]) return "\0" + id;
        for (const name of Object.keys(stubs)) {
          if (
            normalizePath(id) ===
            normalizePath(path.join(frontend, "src", name.slice(2)))
          )
            return "\0" + name;
        }
        if (
          id.endsWith("/vision-overrides") ||
          id.endsWith("/vision-overrides.ts") ||
          id === "./vision-overrides"
        )
          return "\0vision";
        if (id === "/__simulation_entry.jsx")
          return path.join(frontend, "__simulation_entry.jsx");
      },
      load(id) {
        if (id.startsWith("\0") && stubs[id.slice(1)])
          return stubs[id.slice(1)];
        if (id === "\0vision")
          return "export const resolveVisionOverrides=async()=>({});";
        if (id.endsWith("__simulation_entry.jsx")) return entry;
      },
      configureServer(server) {
        server.middlewares.use(
          (req, res, next) =>
            void fixture(req, res, next).catch((e) =>
              json(res, { error: String(e) }, 500),
            ),
        );
        server.middlewares.use((req, res, next) => {
          if (req.url === "/" || req.url.startsWith("/?")) {
            res.setHeader("Content-Type", "text/html");
            res.end(
              '<html><div id="root"></div><script type="module" src="/__simulation_entry.jsx"></script></html>',
            );
          } else next();
        });
      },
    },
  ],
});
await server.listen();
console.log("Upload simulation ready at http://127.0.0.1:18948");
