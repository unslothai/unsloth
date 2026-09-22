// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// playwright-cli run-code --filename=tests/studio/reasoning-codex-checks.js
async (page) => {
  const check = (condition, message) => { if (!condition) throw new Error(message); };
  await page.addInitScript(() => {
    localStorage.setItem("unsloth_chat_collapse_html_artifacts", "true");
    if (window.__fenceTransfers) return;
    window.__fenceTransfers = { full: [], delta: [] };
    const post = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function (message, ...args) {
      if (typeof message.source === "string") window.__fenceTransfers.full.push(message.source.length);
      else if (message.source?.text !== undefined) window.__fenceTransfers.delta.push(message.source.text.length);
      return post.call(this, message, ...args);
    };
  });
  await page.reload();
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.waitForFunction(() => window.__reasoning);
  await page.evaluate(() => window.__reasoning.run({ size: 250000, kind: "code", chunk: 512, gap: 12 }));
  await page.waitForFunction(() => window.__reasoning.stats().done, { timeout: 60000 });
  await page.waitForTimeout(400);
  const streaming = await page.evaluate(() => ({ ...window.__reasoning.stats(), transfers: window.__fenceTransfers }));
  check(streaming.mounted < 30000, "live code mounted unbounded content");
  check(streaming.transfers.delta.length > 20, "live fence did not use incremental worker transfers");
  check(streaming.transfers.full.length < 10, "live fence repeatedly transferred its complete history");

  const selectedCode = await page.evaluate(() => {
    const row = [...document.querySelectorAll("[data-reasoning-code-row]")].find((node) => {
      const rect = node.getBoundingClientRect(); return rect.top > 100 && rect.top < 600;
    });
    if (!row) throw new Error("no visible code row to select");
    const range = document.createRange(); range.selectNodeContents(row);
    const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
    window.__codeSelectionAnchor = { node: selection.anchorNode, offset: selection.anchorOffset };
    return selection.toString();
  });
  await page.mouse.move(550, 350); await page.mouse.wheel(0, -7000);
  await page.waitForTimeout(500);
  check(await page.evaluate(() => getSelection().toString()) === selectedCode, "scrolling to earlier code destroyed native selection");
  await page.keyboard.down("Shift"); await page.mouse.click(500, 150); await page.keyboard.up("Shift");
  await page.waitForTimeout(300);
  check(await page.evaluate(() => {
    const selection = getSelection(), anchor = window.__codeSelectionAnchor;
    return selection.toString().length > 0 && selection.anchorNode === anchor.node && selection.anchorOffset === anchor.offset;
  }), "extending selection across a code gap lost its original endpoint");
  await page.evaluate(() => getSelection().removeAllRanges());

  await page.evaluate(() => window.__reasoning.seed({ text: "~~~text\n" + "x".repeat(25000) + "\n~~~" }));
  await page.mouse.move(550, 350); await page.mouse.wheel(0, 100000);
  await page.waitForTimeout(500);
  const selection = await page.evaluate(() => {
    const rows = [...document.querySelectorAll("[data-reasoning-code-row]")];
    const range = document.createRange();
    range.setStart(rows[0], 0); range.setEnd(rows[1], rows[1].childNodes.length);
    const selection = getSelection(); selection.removeAllRanges(); selection.addRange(range);
    const result = { text: selection.toString(), expected: rows[0].textContent + rows[1].textContent };
    const rectAt = (row, last) => {
      const walker = document.createTreeWalker(row, NodeFilter.SHOW_TEXT);
      const texts = []; for (let n = walker.nextNode(); n; n = walker.nextNode()) if (n.length) texts.push(n);
      const node = last ? texts.at(-1) : texts[0];
      const r = document.createRange(); const start = last ? node.length - 1 : 0;
      r.setStart(node, start); r.setEnd(node, start + 1);
      const rect = r.getBoundingClientRect(); return { top: rect.top, left: rect.left, right: rect.right, width: rect.width };
    };
    const a = rectAt(rows[0], true), b = rectAt(rows[1], false);
    const right = rows[0].closest("pre").getBoundingClientRect().right;
    selection.removeAllRanges();
    return { ...result, a, b, right };
  });
  check(selection.text === selection.expected && !selection.text.includes("\n"), "native selection inserted a false code newline");
  if (selection.a.right + selection.a.width < selection.right - 1)
    check(Math.abs(selection.a.top - selection.b.top) < 1, "logical line has a forced visual break at its fragment seam");

  await page.evaluate(() => window.__reasoning.run({ text: Array.from({ length: 3000 }, (_, i) => `word${String(i).padStart(4, "0")}`).join(" "), chunk: 4000, gap: 800 }));
  await page.waitForFunction(() => document.querySelector('[data-slot="reasoning-text"]')?.textContent.length >= 12000);
  await page.mouse.move(550, 350); await page.mouse.wheel(0, -1500);
  await page.waitForTimeout(150);
  const anchor = await page.evaluate(() => {
    const paragraph = document.querySelector('[data-slot="reasoning-text"] p');
    const text = paragraph.firstChild;
    const viewport = document.querySelector(".aui-thread-viewport").getBoundingClientRect();
    for (const match of text.textContent.matchAll(/word\d{4}/g)) {
      const range = document.createRange(); range.setStart(text, match.index); range.setEnd(text, match.index + match[0].length);
      const top = range.getBoundingClientRect().top;
      if (top > viewport.top + 40 && top < viewport.top + 150) return { text: match[0], top };
    }
    throw new Error("no visible word inside tall paragraph");
  });
  await page.waitForSelector('[data-slot="reasoning-transcript"]');
  await page.waitForTimeout(250);
  const after = await page.evaluate((anchor) => {
    const root = document.querySelector('[data-slot="reasoning-text"]');
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const start = node.textContent.indexOf(anchor.text); if (start < 0) continue;
      const range = document.createRange(); range.setStart(node, start); range.setEnd(node, start + anchor.text.length);
      return range.getBoundingClientRect().top;
    }
    return null;
  }, anchor);
  check(after !== null && Math.abs(after - anchor.top) < 30, `tall-block threshold anchor drifted: ${after - anchor.top}px`);
  await page.waitForFunction(() => window.__reasoning.stats().done);

  await page.evaluate(() => window.__reasoning.seed({ text: "Earlier thought.\n\n".repeat(1000) + "```mermaid\ngraph TD\nA-->B\n```" }));
  await page.mouse.move(550, 350); await page.mouse.wheel(0, 100000);
  await page.waitForSelector('[data-streamdown="mermaid-block"]', { timeout: 15000 });
  await page.evaluate(() => window.__reasoning.seed({ text: "| Item | Value |\n| :--- | ---: |\n" + "| bird | 42 |\n".repeat(3000) }));
  await page.mouse.move(550, 350); await page.mouse.wheel(0, 100000);
  await page.waitForTimeout(500);
  check(await page.locator('[data-table-continuation] table').count() > 0, "table continuation became plain text");
  check(await page.locator('[data-table-continuation] thead').first().evaluate((node) => getComputedStyle(node).display) === "none", "table continuation repeated its header");
  for (const language of ["svg", "SVG", "xml", "html"]) {
    await page.evaluate((language) => window.__reasoning.seed({ text: "Earlier thought.\n\n".repeat(1000) + '```' + language + '\n<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80"><circle cx="40" cy="40" r="30"/></svg>\n```' }), language);
    await page.mouse.wheel(0, 100000);
    await page.waitForFunction(() => {
      const image = document.querySelector('img[alt="SVG preview"]');
      return image?.complete && image.naturalWidth > 0;
    });
    check(await page.getByAltText("SVG preview").count() === 1, `bounded ${language} SVG preview did not load`);
  }
  await page.evaluate(() => window.__reasoning.seed({ text: "Earlier thought.\n\n".repeat(1000) + '```html\n<!doctype html><html><head><title>Fence regression</title></head><body>Game</body></html>\n```' }));
  await page.mouse.wheel(0, 100000);
  await page.getByRole("button", { name: "Open HTML preview preview", exact: true }).waitFor();

  const codeSource = "const bird = { x: 80, y: 140, velocity: 0 };\n".repeat(1000).trimEnd();
  await page.evaluate((source) => {
    const open = "```javascript\n" + source + "\n";
    window.__reasoning.run({ text: open + "```\n\n" + "Later thought.\n\n".repeat(6000), chunk: open.length, gap: 4000 });
  }, codeSource);
  await page.waitForTimeout(250);
  await page.mouse.wheel(0, -100000);
  await page.waitForTimeout(400);
  check(await page.getByTitle("Copy code", { exact: true }).isDisabled(), "open large fence allowed copy while streaming");
  check(await page.getByTitle("Download file", { exact: true }).isDisabled(), "open large fence allowed download while streaming");
  await page.waitForFunction(() => document.querySelector('button[title="Copy code"]:not(:disabled)'));
  check(!await page.evaluate(() => window.__reasoning.stats().done), "fence actions only enabled after all reasoning finished");
  await page.context().grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.getByTitle("Copy code", { exact: true }).click();
  check(await page.evaluate(() => navigator.clipboard.readText()) === codeSource, "fence copy lost offscreen code");
  const downloadPromise = page.waitForEvent("download");
  await page.getByTitle("Download file", { exact: true }).click();
  const download = await downloadPromise;
  const chunks = []; for await (const chunk of await download.createReadStream()) chunks.push(chunk.toString());
  check(chunks.join("") === codeSource, "fence download lost offscreen code");
  check(!await page.evaluate(() => window.__reasoning.stats().done), "copy/download test did not run during later streaming prose");
  await page.waitForFunction(() => window.__reasoning.stats().done);
  return { passed: true, streaming, longLineSelectionLength: selection.text.length, tallParagraphDrift: after - anchor.top };
}
