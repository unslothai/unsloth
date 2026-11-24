# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Controlled backend-failure regression; no GPU or running backend required.

Prepare the fixture with --prepare-only, serve studio/frontend with Vite (errors uses
--config smoke-model-errors.config.ts), then rerun with --port PORT --label after.
For a clean baseline, pass --frontend BASE/studio/frontend --before --label before.
Each side must have its own Vite process, port and VITE_TEST_CACHE_DIR.
Screenshots exercise production components/handlers; they are not full-app or GPU proof.
"""

import json, re, argparse, hashlib
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from playwright.sync_api import sync_playwright, expect

ap = argparse.ArgumentParser()
ap.add_argument("--port", type = int, default = 5221)
ap.add_argument("--label", default = "after")
ap.add_argument("--before", action = "store_true")
ap.add_argument("--browsers", default = "chromium,firefox,webkit")
ap.add_argument(
    "--frontend",
    type = Path,
    default = Path(__file__).resolve().parents[2] / "studio/frontend",
)
ap.add_argument("--out", type = Path, default = Path("logs/model-error-recovery"))
ap.add_argument("--prepare-only", action = "store_true")
ap.set_defaults(suite = "unload")
a = ap.parse_args()
f = a.frontend.resolve()
source = (f / "src/features/api-monitor/api-monitor-page.tsx").read_text()
start = source.index("  const unloadActiveModel = async")
end = source.index("  const [statusFilter", start)
api = source[start:end]
source = (f / "src/features/audio/hooks/use-audio-model-slot.ts").read_text()
start = source.index("    const activeModel = status?.active_model;")
end = source.index("  }, [", start)
audio = "  function ejectAudio() {\n" + source[start:end] + "  }\n"
main = (
    r"""// Generated from the exact production API and audio handlers; only host state and backend responses are fixtures.
import React,{useState} from "react";
import {createRoot} from "react-dom/client";
import {Toaster} from "./src/components/ui/sonner";
import {toast} from "./src/lib/toast";
import {StopRunningChatsDialog} from "./src/features/chat/components/stop-running-chats-dialog";
import {confirmStopRunningChatsIfNeeded,type StopRunningChatsDecision} from "./src/features/chat/utils/confirm-stop-running-chats";
import {requestScopedLocalPromptQueueStop,requestLocalPromptQueueStop,PROMPT_QUEUE_STOP_EVENT} from "./src/features/chat/utils/prompt-queue-boundary";
import {cancelPreStreamRunReservations} from "./src/features/chat/utils/pre-stream-run-reservation";
import {useChatRuntimeStore} from "./src/features/chat/stores/chat-runtime-store";
import {usePromptQueueUI} from "./src/features/chat/stores/prompt-queue-ui-store";
import {getInferenceStatus,unloadModel} from "./src/features/chat/api/chat-api";
import {resolveInferenceCheckpointId} from "./src/features/chat/lib/apply-inference-status-to-store";
import {modelIdsMatch} from "./src/features/hub/lib/model-identity";
import {isExternalModelId} from "./src/features/chat/external-providers";
import {audioSlotRoutingId} from "./src/features/audio/api";
import {unloadResident} from "./src/features/api-monitor/unload-resident";
import {initializeLocale} from "./src/i18n";
import "./src/index.css";
initializeLocale();
const events:any[]=[];window.addEventListener(PROMPT_QUEUE_STOP_EVENT,(e:any)=>events.push(e.detail));
const label=new URLSearchParams(location.search).get('label');
function App(){
 const [unloading,setUnloading]=useState(false),[unloadError,setUnloadError]=useState<string|null>(null);
 const [busy,setBusy]=useState<string|null>(null),[refreshes,setRefreshes]=useState(0);
 const status={active_model:"model-a",serving:[{id:"model-a"},{id:"model-b"}]};const statusRef={current:status};
 const refresh=()=>setRefreshes(v=>v+1);const refreshStatus=async()=>refresh();
 const invalidatePendingStagedTts=()=>{};const stageTtsDownload=()=>{};
"""
    + api
    + audio
    + r""" (window as any).unloadSmoke={events,refreshes,busy:()=>useChatRuntimeStore.getState().modelLoading,checkpoint:()=>useChatRuntimeStore.getState().params.checkpoint,
 seed:()=>{useChatRuntimeStore.getState().setCheckpoint('model-a',null);useChatRuntimeStore.setState({runningByThreadId:{other:true},localRunByThreadId:{other:true}});usePromptQueueUI.setState({byThreadId:{other:{local:true,runId:'other-run',paused:false}} as any});},
 hold:()=>useChatRuntimeStore.getState().beginModelLoading('loading')};
 return <><main className="mx-auto max-w-3xl p-8 font-heading"><p>{label} · production unload handlers</p><h1 className="my-5 text-2xl">Model unload recovery</h1><p className="mb-6">Model A has an active request. Model B is serving a separate chat.</p><div className="flex gap-3"><button className="rounded-full border px-4 py-2" disabled={unloading} onClick={()=>void unloadActiveModel()}>API: Eject model A</button><button className="rounded-full border px-4 py-2" disabled={busy!==null} onClick={ejectAudio}>Audio: Eject model A</button></div>{unloadError&&<p role="alert" className="mt-6 text-destructive">{unloadError}</p>}<p className="mt-6">Completed refreshes: {refreshes}</p></main><StopRunningChatsDialog/><Toaster/></>;
}
createRoot(document.getElementById('root')!).render(<App/>);
"""
)
(f / "smoke-model-unload-main.tsx").write_text(main)
(f / "smoke-model-unload.html").write_text(
    '<html><head><script src="/crypto-boot.js"></script><meta name="viewport" content="width=device-width,initial-scale=1" /></head><body><div id="root"></div><script type="module" src="/smoke-model-unload-main.tsx"></script></body></html>'
)
if a.prepare_only:
    print(f"Prepared {a.suite} fixture in {f}")
    raise SystemExit(0)
out = a.out / a.suite / a.label
out.mkdir(parents = True, exist_ok = True)
facts = []
with sync_playwright() as pw:
    for engine in a.browsers.split(","):
        browser = getattr(pw, engine).launch()

        def new_page(width = 1100, height = 800):
            page = browser.new_page(
                viewport = {"width": width, "height": height}, color_scheme = "dark"
            )
            page.set_default_timeout(15000)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.errors = errors
            page.add_init_script(
                "localStorage.setItem('theme','dark');localStorage.setItem('unsloth_auth_token','smoke.eyJzdWIiOiJ1bnNsb3RoIiwicm9sZSI6Im93bmVyIn0.smoke')"
            )
            return page

        for surface in ["api", "audio"]:
            scenarios = (
                ["active"]
                if a.before
                else [
                    "decline",
                    "confirm",
                    "idle",
                    "unload-fails",
                    "switch",
                    "gate-held",
                ]
            )
            for scenario in scenarios:
                if surface == "audio" and scenario == "switch":
                    continue
                page = new_page()
                state = {
                    "resident": "model-a",
                    "sent": [],
                    "queries": [],
                    "active": scenario != "idle",
                    "status_reads": 0,
                }

                def route(rt):
                    path = urlparse(rt.request.url).path
                    if path.endswith("/inference/status"):
                        state["status_reads"] += 1
                        rt.fulfill(
                            json = {
                                "active_model": state["resident"],
                                "model_identifier": state["resident"],
                                "status": "ready" if state["resident"] else "idle",
                            }
                        )
                        return
                    if path.endswith("/active-generations"):
                        model = parse_qs(urlparse(rt.request.url).query).get("model", [None])[0]
                        state["queries"].append(model)
                        active = state["active"]
                        rt.fulfill(
                            json = {
                                "count": 1 if active else 0,
                                "thread_ids": ["thread-a"] if active else [],
                                "active": (
                                    [{"thread_id": "thread-a", "kind": "chat"}] if active else []
                                ),
                                "parallel_slots": 1,
                            }
                        )
                        return
                    if path.endswith("/inference/unload"):
                        payload = rt.request.post_data_json
                        state["sent"].append(payload)
                        if scenario == "unload-fails":
                            rt.fulfill(status = 500, json = {"detail": "Controlled unload failure"})
                            return
                        if state["active"] and not payload.get("force_cancel_active"):
                            rt.fulfill(
                                status = 409,
                                json = {
                                    "detail": {
                                        "error": "active_generations",
                                        "message": "Unloading this model would stop 1 chat that is still generating. Stop it first.",
                                    }
                                },
                            )
                            return
                        state["resident"] = (
                            "model-b" if scenario == "switch" and len(state["sent"]) == 1 else None
                        )
                        rt.fulfill(json = {"status": "unloaded"})
                        return
                    if "conversations" in path or "threads" in path:
                        rt.fulfill(json = {"threads": [], "conversations": []})
                        return
                    rt.fulfill(json = {})

                page.route(re.compile(r"^https?://[^/]+/api/"), route)
                page.goto(
                    f"http://127.0.0.1:{a.port}/smoke-model-unload.html?label={a.label}",
                    wait_until = "domcontentloaded",
                )
                page.wait_for_function("!!window.unloadSmoke", timeout = 120000)
                page.evaluate("window.unloadSmoke.seed()")
                if scenario == "gate-held":
                    page.evaluate("window.unloadSmoke.hold()")
                page.get_by_role(
                    "button",
                    name = f'{surface.upper() if surface=="api" else "Audio"}: Eject model A',
                ).click()
                if a.before:
                    page.wait_for_timeout(800)
                    assert len(state["sent"]) == 1 and not state["sent"][0].get(
                        "force_cancel_active"
                    )
                    assert state["resident"] == "model-a"
                    page.screenshot(path = str(out / f"{engine}-{surface}-refusal.png"))
                elif scenario == "gate-held":
                    page.wait_for_timeout(300)
                    assert not state["sent"]
                    expect(page.get_by_role("alertdialog")).to_have_count(0)
                elif scenario == "idle":
                    page.wait_for_function("window.unloadSmoke.refreshes>0")
                    assert state["sent"] == [
                        {"model_path": "model-a", "force_cancel_active": False}
                    ]
                    expect(page.get_by_role("alertdialog")).to_have_count(0)
                else:
                    dialog = page.get_by_role("alertdialog")
                    expect(dialog).to_be_visible()
                    assert not state["sent"]
                    assert page.evaluate("window.unloadSmoke.busy()")
                    if scenario == "confirm":
                        page.screenshot(path = str(out / f"{engine}-{surface}-confirmation.png"))
                    if scenario == "decline":
                        page.get_by_role("button", name = "Keep generating").click()
                        page.wait_for_function("!window.unloadSmoke.busy()")
                        assert not state["sent"]
                        assert page.evaluate("window.unloadSmoke.events.length") == 0
                    else:
                        page.get_by_role("button", name = "Stop and unload").click()
                        if scenario == "switch":
                            expect(dialog).to_be_visible()
                            assert len(state["sent"]) == 1
                            page.get_by_role("button", name = "Keep generating").click()
                        page.wait_for_function("!window.unloadSmoke.busy()")
                        assert state["sent"][0] == {
                            "model_path": "model-a",
                            "force_cancel_active": True,
                        }
                        assert all(
                            "other" not in event.get("threadIds", [])
                            for event in page.evaluate("window.unloadSmoke.events")
                        )
                        if scenario == "unload-fails":
                            assert page.evaluate("window.unloadSmoke.checkpoint()") == "model-a"
                if not a.before and scenario != "gate-held":
                    assert all(q in ["model-a", "model-b"] for q in state["queries"]), state
                assert not page.errors, page.errors
                facts.append(
                    {
                        "browser": engine,
                        "surface": surface,
                        "scenario": scenario,
                        **state,
                    }
                )
                page.close()
        browser.close()
(out / "facts.json").write_text(json.dumps(facts, indent = 2))
print(
    json.dumps({"label": a.label, "suite": a.suite, "cases": len(facts), "passed": True}),
    flush = True,
)

(out / "source.json").write_text(
    json.dumps(
        {
            "frontend": str(f),
            "fixture_sha256": hashlib.sha256(main.encode()).hexdigest(),
            "label": a.label,
            "port": a.port,
            "before": a.before,
        },
        indent = 2,
    )
)
