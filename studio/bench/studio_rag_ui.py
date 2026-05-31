#!/usr/bin/env python3
"""Drive a running Studio's RAG through the real web UI with Playwright:
log in, enable RAG, upload a document via the composer, time indexing to the
"RAG index ready" signal, and capture screenshots + video.

Usage:
  python studio_rag_ui.py --base http://127.0.0.1:8905 --label baseline \
     --password "<bootstrap-or-changed>" --doc ../data/rag_corpus/bert_1810.04805.pdf \
     --out ../outputs/ui_baseline
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from playwright.async_api import async_playwright


def get_token(base, user, pw):
    new = pw + "Aa1!"
    with httpx.Client(timeout=30) as c:
        r = c.post(f"{base}/api/auth/login", json={"username": user, "password": pw})
        if r.status_code == 401:
            r = c.post(f"{base}/api/auth/login", json={"username": user, "password": new})
        r.raise_for_status()
        body = r.json()
        tok = body["access_token"]
        if body.get("must_change_password"):
            r2 = c.post(f"{base}/api/auth/change-password",
                        headers={"Authorization": f"Bearer {tok}"},
                        json={"current_password": pw, "new_password": new})
            r2.raise_for_status()
            tok = r2.json()["access_token"]
            refresh = r2.json().get("refresh_token", "")
        else:
            refresh = body.get("refresh_token", "")
    return tok, refresh


def init_script(tok, refresh):
    seed = {"unsloth_auth_token": tok, "unsloth_refresh_token": refresh}
    return f"""(() => {{ const s={json.dumps(seed)};
      for (const k of Object.keys(s)) {{ try {{ localStorage.setItem(k, s[k]); }} catch(e){{}} }} }})();"""


async def run(args):
    out = Path(args.out)
    (out / "video").mkdir(parents=True, exist_ok=True)
    tok, refresh = get_token(args.base, args.username, args.password)
    result = {"label": args.label, "doc": Path(args.doc).name}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(out / "video"),
            record_video_size={"width": 1440, "height": 900},
        )
        await ctx.add_init_script(init_script(tok, refresh))
        page = await ctx.new_page()
        await page.goto(f"{args.base}/chat", wait_until="domcontentloaded")
        await page.locator("form:has(textarea) textarea").first.wait_for(state="visible", timeout=30000)
        await page.screenshot(path=str(out / "01_chat.png"))

        # Enable RAG so the document attach control renders.
        enable = page.locator('button[aria-label="Enable RAG"]').first
        try:
            await enable.click(timeout=8000)
        except Exception:
            pass  # already enabled
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(out / "02_rag_on.png"))

        # Upload via the hidden file input (native picker can't be driven).
        file_input = page.locator('input[type="file"][accept*=".pdf"]').first
        await file_input.wait_for(state="attached", timeout=8000)
        t0 = time.perf_counter()
        await file_input.set_input_files(args.doc)

        # Wait for the global "RAG index ready" toast (fallback: chip "Ready").
        ready_toast = page.get_by_text("RAG index ready", exact=False).first
        chip_ready = page.get_by_text("Ready", exact=True).first
        indexed = False
        deadline = time.perf_counter() + 180
        while time.perf_counter() < deadline:
            try:
                if await ready_toast.is_visible():
                    indexed = True
                    break
            except Exception:
                pass
            try:
                if await chip_ready.is_visible():
                    indexed = True
                    break
            except Exception:
                pass
            await page.wait_for_timeout(250)
        elapsed = time.perf_counter() - t0
        result["index_seconds"] = round(elapsed, 2)
        result["indexed"] = indexed
        await page.screenshot(path=str(out / "03_indexed.png"), full_page=True)
        print(f"[{args.label}] UI upload->ready: {elapsed:.2f}s indexed={indexed}")

        await ctx.close()
        await browser.close()

    # rename video
    webms = sorted((out / "video").glob("*.webm"))
    if webms:
        webms[-1].rename(out / "video" / f"{args.label}.webm")
    Path(out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[{args.label}] wrote {out/'result.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--username", default="unsloth")
    ap.add_argument("--doc", required=True)
    ap.add_argument("--out", required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
