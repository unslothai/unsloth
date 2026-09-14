# Inline image simulations

The fixture mounts the production Markdown renderer with the desktop content security policy. A loopback server supplies generated raster files, checks bearer authentication, and records the requested sandbox scope. It does not start an inference backend.

Before testing, consider path normalization, encoded separators, stale thread or project scope, streaming partial links, failed requests, delayed responses, and browser image decoding. The frontend pipeline test covers 2,100 path and markup combinations. The browser fixture covers 59 cases; the automated runner also verifies the downloaded filename and PNG dimensions.

From the repository root on macOS or Linux:

```sh
uv venv temp/inline-image-validation/venv
uv pip install --python temp/inline-image-validation/venv/bin/python playwright==1.62.0 pillow==12.3.0
export PLAYWRIGHT_BROWSERS_PATH="$PWD/temp/inline-image-validation/browsers"
temp/inline-image-validation/venv/bin/python -m playwright install chromium firefox webkit
temp/inline-image-validation/venv/bin/python tests/studio/playwright_inline_images.py
```

On Windows, use `venv/Scripts/python.exe` and set `PLAYWRIGHT_BROWSERS_PATH` in PowerShell. Install frontend dependencies first with `npm ci` in `studio/frontend`.

Use `--browsers chrome msedge` for installed Chrome and Edge. `--manual` prints a URL for testing installed browsers, including Safari. WebKit automation is engine coverage, not a Safari application test.

To reproduce the original failure, pass `--baseline FULL_COMMIT_SHA --output temp/inline-image-validation/before`. The baseline must lack the relative-image fix. The control expects a bare filename to fail while an explicit sandbox URL loads. Reports, downloads, and failure screenshots stay beneath the selected output directory.
