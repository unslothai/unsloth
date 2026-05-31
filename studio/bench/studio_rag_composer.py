#!/usr/bin/env python3
"""Full RAG-in-chat visual via Playwright: with a GGUF model already loaded,
enable the RAG toggle, upload a PDF through the composer, ask a grounded
question, and capture the answer + sources. Records screenshots + video.

Usage:
  python studio_rag_composer.py --base http://127.0.0.1:8912 --label improved \
     --password StudioBench2026! --doc ../data/rag_corpus/bert_1810.04805.pdf \
     --question "What are BERT's two pre-training objectives?" --out ../outputs/composer_improved
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
from playwright.async_api import async_playwright


def get_token(base, pw):
    new = pw + "Aa1!"
    with httpx.Client(timeout = 30) as c:
        r = c.post(
            f"{base}/api/auth/login", json = {"username": "unsloth", "password": pw}
        )
        if r.status_code == 401:
            r = c.post(
                f"{base}/api/auth/login", json = {"username": "unsloth", "password": new}
            )
        r.raise_for_status()
        b = r.json()
        t = b["access_token"]
        if b.get("must_change_password"):
            r2 = c.post(
                f"{base}/api/auth/change-password",
                headers = {"Authorization": f"Bearer {t}"},
                json = {"current_password": pw, "new_password": new},
            )
            r2.raise_for_status()
            t, b = r2.json()["access_token"], r2.json()
        return t, b.get("refresh_token", "")


def init_script(tok, refresh):
    seed = {"unsloth_auth_token": tok, "unsloth_refresh_token": refresh}
    return f"""(() => {{ const s={json.dumps(seed)};
      for (const k of Object.keys(s)) {{ try {{ localStorage.setItem(k, s[k]); }} catch(e){{}} }} }})();"""


async def run(args):
    out = Path(args.out)
    (out / "video").mkdir(parents = True, exist_ok = True)
    tok, refresh = get_token(args.base, args.password)
    result = {
        "label": args.label,
        "doc": Path(args.doc).name,
        "question": args.question,
    }

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless = True)
        ctx = await browser.new_context(
            viewport = {"width": 1440, "height": 900},
            record_video_dir = str(out / "video"),
            record_video_size = {"width": 1440, "height": 900},
        )
        await ctx.add_init_script(init_script(tok, refresh))
        page = await ctx.new_page()
        await page.goto(f"{args.base}/chat", wait_until = "domcontentloaded")
        await page.locator("form:has(textarea) textarea").first.wait_for(
            state = "visible", timeout = 30000
        )
        await page.wait_for_timeout(1500)
        await page.screenshot(path = str(out / "01_chat.png"))

        # Enable the RAG toggle (the pill is active once a model is loaded).
        for lbl in ("Enable RAG", "Disable RAG"):
            btn = page.locator(f'button[aria-label="{lbl}"]').first
            if await btn.count():
                if lbl == "Enable RAG":
                    await btn.click()
                break
        await page.wait_for_timeout(700)
        await page.screenshot(path = str(out / "02_rag_on.png"))

        # Upload a document through the composer attach control.
        file_input = page.locator('input[type="file"][accept*=".pdf"]').first
        await file_input.wait_for(state = "attached", timeout = 10000)
        t0 = time.perf_counter()
        await file_input.set_input_files(args.doc)
        ready = page.get_by_text("Ready", exact = True).first
        toast = page.get_by_text("RAG index ready", exact = False).first
        indexed = False
        while time.perf_counter() - t0 < 180:
            for sig in (toast, ready):
                try:
                    if await sig.is_visible():
                        indexed = True
                        break
                except Exception:
                    pass
            if indexed:
                break
            await page.wait_for_timeout(250)
        result["index_seconds"] = round(time.perf_counter() - t0, 2)
        result["indexed"] = indexed
        await page.screenshot(path = str(out / "03_indexed.png"))

        # Ask a grounded question.
        box = page.locator("form:has(textarea) textarea").first
        await box.click()
        await box.fill(args.question)
        await box.press("Enter")
        stop = page.locator(
            'button[aria-label="Stop generating"], button:has-text("Stop")'
        ).first
        try:
            await stop.wait_for(state = "visible", timeout = 30000)
        except Exception:
            pass
        try:
            await stop.wait_for(state = "hidden", timeout = 180000)
        except Exception:
            pass
        await page.wait_for_timeout(1000)
        await page.screenshot(path = str(out / "04_answer.png"), full_page = True)

        answer = await page.evaluate(
            """() => Array.from(document.querySelectorAll('[data-role="assistant"], [data-message-role="assistant"]'))
                 .map(n => n.textContent || '').join('\\n')"""
        )
        result["answer_excerpt"] = (answer or "")[-600:]
        print(
            f"[{args.label}] indexed={indexed} in {result['index_seconds']}s; answer chars={len(answer or '')}"
        )

        await ctx.close()
        await browser.close()

    webms = sorted((out / "video").glob("*.webm"))
    if webms:
        webms[-1].rename(out / "video" / f"{args.label}.webm")
    Path(out / "result.json").write_text(json.dumps(result, indent = 2))
    print(f"[{args.label}] wrote {out/'result.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required = True)
    ap.add_argument("--label", required = True)
    ap.add_argument("--password", required = True)
    ap.add_argument("--doc", required = True)
    ap.add_argument("--question", required = True)
    ap.add_argument("--out", required = True)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
