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
ap.set_defaults(suite = "errors")
a = ap.parse_args()
f = a.frontend.resolve()
main = r"""import React,{useState} from "react";
import {createRoot} from "react-dom/client";
import {Toaster} from "./src/components/ui/sonner";
import {toast} from "./src/lib/toast";
import {smokeImageLoadFailure} from "./src/features/images/images-page";
import {smokeVideoLoadFailure,smokeVideoGenerationFailure} from "./src/features/video/video-page";
import {initializeLocale} from "./src/i18n";
import "./src/index.css";
import {modelErrorToast} from "./src/lib/model-error-toast";
import { ModelLoadError } from "./src/features/api-monitor/components/model-load-error";
initializeLocale();
const label=new URLSearchParams(location.search).get("label")||"Component regression fixture";
const diagnostic="Failed to load model: Couldn't allocate memory. Try lowering the context length or choosing a smaller model.\n\nllama-server output:\n"+Array.from({length:22},(_,i)=>`${i}: load_tensors: CUDA buffer allocation failed while loading model tensors`).join("\n")+"\n\nFull log: C:\\Unsloth\\logs\\model.log";
let actions=0;
function App(){const [error,setError]=useState<string|null>(null);
 (window as any).errorsSmoke={show:(kind:string,message=diagnostic)=>{
 if(kind==='api')setError(message);
 else if(kind==='image-load')smokeImageLoadFailure(new Error(message),"Failed to load model");
 else if(kind==='video-load')smokeVideoLoadFailure(new Error(message),"Failed to load model");
 else if(kind==='video-generation')smokeVideoGenerationFailure(message);
 else modelErrorToast(message,{id:'model-load',duration:Infinity,action:{label:'Retry',onClick:()=>{actions++}}});
 },actions:()=>actions,diagnostic,loading:()=>toast.loading('Loading model',{id:'model-load'}),dismiss:()=>toast.dismiss()};
 return <><main className="mx-auto max-w-4xl p-6 font-heading"><p>{label}</p><h1 className="my-4 text-2xl">Model error presentation</h1><p className="mb-6 text-sm text-muted-foreground">Production error handlers and components · injected backend failure</p>
 <section className="flex min-w-0 flex-wrap items-center gap-2 rounded-xl bg-card p-4"><button className="rounded-full border px-4 py-2">Qwen3-Coder-Next-GGUF</button>{error&&<ModelLoadError message={error} onDismiss={()=>setError(null)}/>}</section>
 <section className="mt-5 rounded-xl bg-card p-5"><h2>Other controls stay available</h2><button onClick={()=>actions++} className="mt-3 rounded-full border px-3 py-2">Check interaction</button></section></main><Toaster/></>}
createRoot(document.getElementById('root')!).render(<App/>);
"""
if a.before:
    source = (f / "src/features/api-monitor/components/api-model-load-controls.tsx").read_text()
    start = source.index(
        '        <span\n          role="alert"', source.index("      {actionError ? (")
    )
    end = source.index("        </span>", start) + len("        </span>")
    original = (
        "function ModelLoadError({message}: {message:string;onDismiss:()=>void}) {const actionError=message;return ("
        + source[start:end]
        + ");}"
    )
    main = main.replace(
        'import { ModelLoadError } from "./src/features/api-monitor/components/model-load-error";',
        original,
    )
    main = main.replace(
        'import {modelErrorToast} from "./src/lib/model-error-toast";',
        "const modelErrorToast=toast.error;",
    )
(f / "smoke-model-errors-main.tsx").write_text(main)
(f / "smoke-model-errors.html").write_text(
    '<html class="dark"><head><script src="/crypto-boot.js"></script><meta name="viewport" content="width=device-width,initial-scale=1" /></head><body><div id="root"></div><script type="module" src="/smoke-model-errors-main.tsx"></script></body></html>'
)
(f / "smoke-model-errors.config.ts").write_text(
    'import {mergeConfig} from "vite";\nimport base from "./vite.config";\nexport default mergeConfig(base,{plugins:[{name:"model-error-test-exports",enforce:"pre",transform(code,id){\n if(id.endsWith("/features/images/images-page.tsx")) return code+"\\nexport {reportLoadFailure as smokeImageLoadFailure};";\n if(id.endsWith("/features/video/video-page.tsx")) return code+"\\nexport {reportLoadFailure as smokeVideoLoadFailure,showVideoFailureToast as smokeVideoGenerationFailure};";\n}}]});\n'
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

        for width, height in [(1100, 800), (390, 844)]:
            page = new_page(width, height)
            page.route(re.compile(r"^https?://[^/]+/api/"), lambda rt: rt.fulfill(json = {}))
            page.goto(
                f"http://127.0.0.1:{a.port}/smoke-model-errors.html?label={a.label}",
                wait_until = "domcontentloaded",
            )
            page.wait_for_function("!!window.errorsSmoke", timeout = 120000)
            for kind in [
                "api",
                "audio-load",
                "image-load",
                "video-load",
                "video-generation",
            ]:
                # A video failure's existing structured reason must remain available.
                message = (
                    None
                    if kind != "video-generation"
                    else "Video generation failed. RuntimeError: "
                    + ("CUDA allocation failed. " * 100)
                )
                page.evaluate(
                    "([kind,message])=>message?window.errorsSmoke.show(kind,message):window.errorsSmoke.show(kind)",
                    [kind, message],
                )
                panel = (
                    page.get_by_role("alert")
                    if kind == "api"
                    else page.locator("[data-sonner-toast]").last
                )
                expect(panel).to_be_visible()
                page.wait_for_timeout(350)
                shot = f"{engine}-{width}-{kind}.png"
                page.screenshot(path = str(out / shot))
                box = panel.bounding_box()
                record = {
                    "browser": engine,
                    "width": width,
                    "surface": kind,
                    "height": round(box["height"], 1),
                    "top": round(box["y"], 1),
                    "screenshot": shot,
                }
                if not a.before:
                    disclosure = page.get_by_text("Technical details", exact = True).last
                    expect(disclosure).to_be_visible()
                    disclosure.click()
                    detail = page.get_by_label("Error technical details").last
                    expect(detail).to_be_visible()
                    assert detail.evaluate("(e)=>e.clientHeight") <= 192
                    assert "CUDA" in detail.inner_text()
                    assert not page.evaluate("document.documentElement.scrollWidth>innerWidth")
                    box = panel.bounding_box()
                    assert box["y"] >= 0, (kind, box)
                    if kind == "api":
                        page.screenshot(path = str(out / f"{engine}-{width}-api-expanded.png"))
                        page.get_by_role("button", name = "Dismiss model load error").click()
                        expect(page.get_by_role("alert")).to_have_count(0)
                    else:
                        expect(panel.locator("[data-close-button]")).to_be_visible()
                        if kind == "audio-load":
                            previous = page.evaluate("window.errorsSmoke.actions()")
                            panel.get_by_role("button", name = "Retry").click()
                            assert page.evaluate("window.errorsSmoke.actions()") == previous + 1
                        else:
                            panel.locator("[data-close-button]").click()
                    record["details_bounded"] = True
                else:
                    if kind == "api":
                        page.reload()
                        page.wait_for_function("!!window.errorsSmoke")
                    else:
                        page.evaluate("window.errorsSmoke.dismiss()")
                        page.wait_for_timeout(400)
                facts.append(record)
            if not a.before:
                # Short errors, arbitrary multiline text, HTML, and replacement of a loading toast.
                for message in [
                    "Try a smaller model.",
                    "x\n" * 40,
                    "x" * 5000,
                    '<img src=x onerror="window.unexpected=true">' + "x" * 400,
                ]:
                    page.evaluate(
                        '(m)=>{window.errorsSmoke.loading();window.errorsSmoke.show("audio-load",m)}',
                        message,
                    )
                    toast = page.locator("[data-sonner-toast]").last
                    expect(toast).to_be_visible()
                    page.wait_for_timeout(350)
                    assert toast.bounding_box()["y"] >= 0
                    assert not page.evaluate("!!window.unexpected")
                    if len(message) > 240 or message.count("\n") > 3:
                        page.get_by_text("Technical details", exact = True).last.click()
                        expect(page.get_by_label("Error technical details").last).to_have_text(
                            message
                        )
                    else:
                        expect(toast).to_contain_text(message)
                    assert page.locator('[data-sonner-toast][data-type="loading"]').count() == 0
                    toast.locator("[data-close-button]").click()
                    page.wait_for_timeout(350)
            assert not page.errors, page.errors
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
