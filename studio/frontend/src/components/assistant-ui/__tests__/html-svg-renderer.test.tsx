// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  HtmlSvgRenderer,
  isHtmlFence,
  isSvgFence,
  parseCodeFence,
  parseIncompleteCodeFence,
  sanitizeSvgSource,
} from "../html-svg-renderer";

// Fake the /api/preview/html POST round-trip so the iframe reaches a
// deterministic post-load state under jsdom.
let _previewIdCounter = 0;
function installFetchStub(): void {
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    if (typeof input === "string" && input === "/api/preview/html") {
      const url = `/api/preview/html/test-token-${++_previewIdCounter}`;
      return new Response(JSON.stringify({ url }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }
    return new Response("not-found", { status: 404 });
  }) as typeof fetch;
}
function installFailingFetchStub(): void {
  globalThis.fetch = vi.fn(async () =>
    new Response("server boom", { status: 500 }),
  ) as typeof fetch;
}

describe("HtmlSvgRenderer", () => {
  beforeEach(() => {
    installFetchStub();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders an HTML preview inside a sandboxed iframe pointing at the same-origin preview route", async () => {
    const html = "<html><body><h1>hello</h1></body></html>";
    render(<HtmlSvgRenderer language="html" source={html} />);

    const root = screen.getByTestId("html-svg-renderer");
    expect(root.getAttribute("data-active-tab")).toBe("preview");
    expect(root.getAttribute("data-language")).toBe("html");

    const iframe = screen.getByTestId(
      "html-svg-renderer-iframe",
    ) as HTMLIFrameElement;
    expect(iframe.tagName).toBe("IFRAME");
    // allow-scripts/allow-modals/allow-popups grant the runtime the
    // route CSP unlocks; NOT granting allow-same-origin / allow-top-nav /
    // allow-popups-to-escape-sandbox blocks parent access and tabnabbing.
    const sandbox = iframe.getAttribute("sandbox") ?? "";
    const sandboxTokens = sandbox.split(/\s+/);
    expect(sandboxTokens).toContain("allow-scripts");
    expect(sandboxTokens).toContain("allow-modals");
    expect(sandboxTokens).toContain("allow-popups");
    expect(sandboxTokens).not.toContain("allow-popups-to-escape-sandbox");
    expect(sandbox).not.toContain("allow-same-origin");
    expect(sandbox).not.toContain("allow-top-navigation");

    // In-flight: about:blank (no flash of previous preview).
    expect(["about:blank", null]).toContain(iframe.getAttribute("src"));

    // After POST resolves: iframe src is the returned same-origin URL.
    await waitFor(() => {
      expect(iframe.getAttribute("data-preview-state")).toBe("ready");
    });
    const src = iframe.getAttribute("src") ?? "";
    expect(src.startsWith("/api/preview/html/")).toBe(true);
    expect(iframe.getAttribute("srcdoc")).toBeNull();
  });

  it("falls back to srcdoc with the defense-in-depth meta CSP when the preview API is unreachable", async () => {
    installFailingFetchStub();
    const html = "<html><body><h1>fallback</h1></body></html>";
    render(<HtmlSvgRenderer language="html" source={html} />);

    const iframe = screen.getByTestId(
      "html-svg-renderer-iframe",
    ) as HTMLIFrameElement;
    await waitFor(() => {
      expect(iframe.getAttribute("data-preview-state")).toBe("error");
    });
    const srcdoc = (iframe.getAttribute("srcdoc") ?? iframe.srcdoc) ?? "";
    expect(srcdoc).toContain("<h1>fallback</h1>");
    expect(srcdoc).toContain('http-equiv="Content-Security-Policy"');
    expect(srcdoc).toContain("connect-src 'none'");
    expect(srcdoc).toContain("frame-src 'none'");
  });

  it("renders an SVG preview inside a no-script sandboxed iframe with srcdoc carrying the sanitized markup", () => {
    const malicious = `<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50">
  <circle cx="25" cy="25" r="20" fill="blue" onclick="alert('pwn')" />
  <script>window.parent.alert("pwn")</script>
</svg>`;

    render(<HtmlSvgRenderer language="svg" source={malicious} />);

    const iframe = screen.getByTestId(
      "html-svg-renderer-svg-preview",
    ) as HTMLIFrameElement;
    expect(iframe.tagName).toBe("IFRAME");
    // SVG iframe must NEVER allow scripts or same-origin (XSS / host leak).
    expect(iframe.getAttribute("sandbox")).toBe("");
    const srcdoc = (iframe.getAttribute("srcdoc") ?? iframe.srcdoc).toLowerCase();
    expect(srcdoc).toContain("<circle");
    expect(srcdoc).not.toContain("<script");
    expect(srcdoc).not.toContain("onclick");
    expect(srcdoc).not.toContain("alert");
    // CSP backstop: data: images only so a sanitizer regression cannot beacon.
    expect(srcdoc).toContain("default-src 'none'");
  });

  it("toggles between Preview and Code tabs", async () => {
    const html = "<html><body>hi</body></html>";
    render(
      <HtmlSvgRenderer
        language="html"
        source={html}
        codeView={<pre data-testid="custom-code-view">{html}</pre>}
      />,
    );

    // Default tab is preview.
    const iframe = screen.getByTestId(
      "html-svg-renderer-iframe",
    ) as HTMLIFrameElement;
    expect(iframe).toBeTruthy();
    expect(screen.queryByTestId("custom-code-view")).toBeNull();
    // Settle the preview POST so it does not raise an out-of-act() warning.
    await waitFor(() => {
      expect(iframe.getAttribute("data-preview-state")).toBe("ready");
    });

    const codeTab = screen.getByRole("tab", { name: /code/i });
    act(() => {
      fireEvent.click(codeTab);
    });

    // Iframe is unmounted, custom code view appears.
    expect(screen.queryByTestId("html-svg-renderer-iframe")).toBeNull();
    expect(screen.getByTestId("custom-code-view")).toBeTruthy();

    const previewTab = screen.getByRole("tab", { name: /preview/i });
    act(() => {
      fireEvent.click(previewTab);
    });

    expect(screen.getByTestId("html-svg-renderer-iframe")).toBeTruthy();
    expect(screen.queryByTestId("custom-code-view")).toBeNull();
  });

  it("locks to the Code tab while the fence is still streaming in", () => {
    const html = "<html><body>partial";
    render(
      <HtmlSvgRenderer language="html" source={html} isIncomplete={true} />,
    );

    const root = screen.getByTestId("html-svg-renderer");
    expect(root.getAttribute("data-active-tab")).toBe("code");

    // Preview disabled while streaming so users see tokens, not flicker.
    const previewTab = screen.getByRole("tab", { name: /preview/i });
    expect(previewTab.hasAttribute("disabled")).toBe(true);
  });

  it("HtmlPreview POSTs the assistant source to /api/preview/html before mounting the iframe src", async () => {
    const html = "<button onclick=\"alert('x')\">go</button>";
    render(<HtmlSvgRenderer language="html" source={html} />);
    const iframe = screen.getByTestId(
      "html-svg-renderer-iframe",
    ) as HTMLIFrameElement;
    await waitFor(() => {
      expect(iframe.getAttribute("data-preview-state")).toBe("ready");
    });
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    const [url, init] = (globalThis.fetch as ReturnType<typeof vi.fn>).mock
      .calls[0] as [string, RequestInit];
    expect(url).toBe("/api/preview/html");
    expect(init.method).toBe("POST");
    expect(init.credentials).toBe("same-origin");
    const parsedBody = JSON.parse(init.body as string) as { source: string };
    expect(parsedBody.source).toBe(html);
    // iframe lands on the token URL; no srcdoc.
    const src = iframe.getAttribute("src") ?? "";
    expect(src.startsWith("/api/preview/html/")).toBe(true);
    expect(iframe.getAttribute("srcdoc")).toBeNull();
  });

  it("wires tabs to their panels with aria-controls / aria-labelledby", async () => {
    const html = "<html><body>hi</body></html>";
    render(<HtmlSvgRenderer language="html" source={html} />);

    const iframe = screen.getByTestId(
      "html-svg-renderer-iframe",
    ) as HTMLIFrameElement;
    await waitFor(() => {
      expect(iframe.getAttribute("data-preview-state")).toBe("ready");
    });

    const previewTab = screen.getByRole("tab", { name: /preview/i });
    const codeTab = screen.getByRole("tab", { name: /code/i });
    const panel = screen.getByRole("tabpanel");

    const controls = previewTab.getAttribute("aria-controls");
    expect(controls).toBeTruthy();
    expect(panel.getAttribute("id")).toBe(controls);
    expect(panel.getAttribute("aria-labelledby")).toBe(
      previewTab.getAttribute("id"),
    );

    // Roving tabindex per WAI-ARIA APG tab pattern.
    expect(previewTab.getAttribute("tabindex")).toBe("0");
    expect(codeTab.getAttribute("tabindex")).toBe("-1");
  });

  it("constrains the SVG preview so a square viewBox does not overflow", () => {
    const svg =
      "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 200 200\"><circle cx=\"100\" cy=\"100\" r=\"95\" fill=\"red\"/></svg>";
    render(<HtmlSvgRenderer language="svg" source={svg} />);

    const iframe = screen.getByTestId(
      "html-svg-renderer-svg-preview",
    ) as HTMLIFrameElement;
    const srcdoc = (iframe.getAttribute("srcdoc") ?? iframe.srcdoc).toLowerCase();
    // Cap BOTH dimensions or a square viewBox clips vertically.
    expect(srcdoc).toContain("max-width:100%");
    expect(srcdoc).toContain("max-height:100%");
    expect(srcdoc).toContain("height:100%");
  });
});

describe("parseIncompleteCodeFence", () => {
  it("returns lang and body for a fence that has not closed yet", () => {
    const partial = "```svg\n<svg><circle";
    const fence = parseIncompleteCodeFence(partial);
    expect(fence).not.toBeNull();
    expect(fence?.language).toBe("svg");
    expect(fence?.source).toBe("<svg><circle");
  });

  it("strips an in-flight trailing ``` so the partial body does not leak it", () => {
    const partial = "```html\n<div>hi</div>\n``";
    const fence = parseIncompleteCodeFence(partial);
    expect(fence?.source).toBe("<div>hi</div>\n``");
  });

  it("returns null when the block is not a fence at all", () => {
    expect(parseIncompleteCodeFence("just text")).toBeNull();
  });

  it("recovers a final-but-never-closed fence (small LLMs drop the closing ```)", () => {
    // Regression caught by live Qwen3-0.6B probe: complete body, no closing
    // ```. Without this path HtmlSvgRenderer would not mount.
    const finalNoClose =
      "```svg\n<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 10 10\">" +
      "<circle cx=\"5\" cy=\"5\" r=\"4\" fill=\"orange\"/></svg>";
    expect(parseCodeFence(finalNoClose)).toBeNull();
    const fence = parseIncompleteCodeFence(finalNoClose);
    expect(fence).not.toBeNull();
    expect(fence?.language).toBe("svg");
    expect(fence?.source).toContain("<circle");
    expect(fence?.source).toContain("</svg>");
  });
});

describe("Fence helpers", () => {
  it("parses a typical markdown code fence", () => {
    const block = "```python\nprint('hi')\n```";
    const fence = parseCodeFence(block);
    expect(fence).not.toBeNull();
    expect(fence?.language).toBe("python");
    expect(fence?.source).toBe("print('hi')");
  });

  it("isSvgFence picks up explicit svg fences and html/xml fences that begin with <svg", () => {
    expect(isSvgFence({ language: "svg", source: "<svg/>" })).toBe(true);
    expect(
      isSvgFence({ language: "html", source: "<svg xmlns='...'></svg>" }),
    ).toBe(true);
    expect(
      isSvgFence({
        language: "xml",
        source: "<?xml version='1.0'?><svg></svg>",
      }),
    ).toBe(true);
    expect(isSvgFence({ language: "html", source: "<div></div>" })).toBe(false);
  });

  it("isHtmlFence is true only for non-SVG html fences", () => {
    expect(isHtmlFence({ language: "html", source: "<div></div>" })).toBe(true);
    expect(isHtmlFence({ language: "html", source: "<svg></svg>" })).toBe(
      false,
    );
    expect(isHtmlFence({ language: "python", source: "print()" })).toBe(false);
  });

  it("non-HTML / non-SVG fences are not handled by the renderer", () => {
    // markdown-text.tsx invokes HtmlSvgRenderer only when these agree.
    const fence = parseCodeFence("```python\nprint('hi')\n```");
    expect(fence).not.toBeNull();
    if (!fence) return;
    expect(isSvgFence(fence)).toBe(false);
    expect(isHtmlFence(fence)).toBe(false);
  });
});

describe("sanitizeSvgSource", () => {
  it("removes <script>, on* handlers, and javascript: URLs", () => {
    const malicious = `<svg xmlns="http://www.w3.org/2000/svg">
  <a href="javascript:alert(1)"><circle cx="5" cy="5" r="4" onload="alert(1)"/></a>
  <script>alert("pwn")</script>
</svg>`;

    const clean = sanitizeSvgSource(malicious).toLowerCase();
    expect(clean).not.toContain("<script");
    expect(clean).not.toContain("onload");
    expect(clean).not.toContain("javascript:");
    expect(clean).toContain("<circle");
  });

  it("drops <?xml ... ?> processing instructions", () => {
    const svg = `<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>`;
    const clean = sanitizeSvgSource(svg);
    expect(clean.startsWith("<?xml")).toBe(false);
    expect(clean).toContain("<rect");
  });

  it("keeps inline <style> blocks so class-styled SVG exports still render", () => {
    // Iframe sandbox="" + default-src 'none' contain <style> blast radius;
    // diagram exporters relying on class styles outweigh the selector leak.
    const svg = `<svg xmlns="http://www.w3.org/2000/svg"><style>.fg{fill:red}</style><rect class="fg"/></svg>`;
    const clean = sanitizeSvgSource(svg).toLowerCase();
    expect(clean).toContain("<style");
    expect(clean).toContain(".fg");
    expect(clean).toContain("<rect");
    expect(clean).toContain('class="fg"');
  });

  it("strips style attributes so inline CSS cannot fire url()/@import requests", () => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg"><circle style="background:url(https://evil.example/x)"/></svg>`;
    const clean = sanitizeSvgSource(svg).toLowerCase();
    expect(clean).toContain("<circle");
    expect(clean).not.toContain("style=");
    expect(clean).not.toContain("evil.example");
  });

  it("drops <image>/<use> tags so SVG cannot beacon to external URLs", () => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><image href="https://evil.example/pixel"/><use xlink:href="https://evil.example/use"/></svg>`;
    const clean = sanitizeSvgSource(svg).toLowerCase();
    expect(clean).not.toContain("<image");
    expect(clean).not.toContain("<use");
    expect(clean).not.toContain("evil.example");
  });

  it("strips filter/mask/clip-path url(...) attrs that would still fetch", () => {
    const svg =
      "<svg xmlns=\"http://www.w3.org/2000/svg\">" +
      "<circle filter=\"url(https://evil.example/f)\" mask=\"url(https://evil.example/m)\" clip-path=\"url(https://evil.example/c)\" r=\"10\"/>" +
      "</svg>";
    const clean = sanitizeSvgSource(svg).toLowerCase();
    expect(clean).toContain("<circle");
    expect(clean).not.toContain("filter=");
    expect(clean).not.toContain("mask=");
    expect(clean).not.toContain("clip-path=");
    expect(clean).not.toContain("evil.example");
  });

  it("keeps safe same-document fragment hrefs used by textPath/gradients", () => {
    const svg =
      "<svg xmlns=\"http://www.w3.org/2000/svg\">" +
      "<defs><linearGradient id=\"g1\"/></defs>" +
      "<text><textPath href=\"#labelPath\">hi</textPath></text>" +
      "<circle fill=\"url(#g1)\" r=\"10\"/>" +
      "</svg>";
    const clean = sanitizeSvgSource(svg).toLowerCase();
    // Fragment hrefs must survive so textPath / gradient refs resolve.
    expect(clean).toContain("href=\"#labelpath\"");
  });

  it("strips external-scheme hrefs even though same-doc fragments are kept", () => {
    const svg =
      "<svg xmlns=\"http://www.w3.org/2000/svg\">" +
      "<a href=\"https://evil.example/exfil\"><circle r=\"10\"/></a>" +
      "</svg>";
    const clean = sanitizeSvgSource(svg).toLowerCase();
    expect(clean).not.toContain("evil.example");
    expect(clean).not.toContain("href=\"https");
  });
});
