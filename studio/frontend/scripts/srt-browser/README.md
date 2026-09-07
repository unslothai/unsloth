# Native SRT browser check

Run on a prepared native SRT host after the backend native tests pass:

```sh
python -m pip install uvicorn httpx PyJWT python-multipart huggingface_hub requests packaging pyyaml python-dotenv cryptography
cd studio/frontend
npm ci
npm install --prefix "$RUNNER_TEMP/srt-browser-deps" --no-save --package-lock=false --ignore-scripts playwright@1.58.2
node "$RUNNER_TEMP/srt-browser-deps/node_modules/playwright/cli.js" install chromium
SRT_PLAYWRIGHT_MODULE="$RUNNER_TEMP/srt-browser-deps/node_modules/playwright/index.mjs" SRT_BROWSER_ARTIFACTS="$RUNNER_TEMP/srt-browser" SRT_BROWSER_PYTHON="$(command -v python)" node scripts/srt-browser/run.mjs
```

The existing native test dependencies (FastAPI, structlog, psutil and Pillow) are also required. No model weights or GPU libraries are installed by this fixture.

The runner starts loopback-only FastAPI and Vite servers, uses real production consent controls and capability routes, and executes fixed benign Python and Terminal payloads through production `execute_tool`. Authentication dependencies use an isolated fixture identity. Chromium clicks prove Required execution and server-owned SRT records, Full confirmation, and reload back to Required; Escape closes the limitation menu. This is a native tool/API integration check, not a model-provider chat conversation or a broader platform qualification.

Screenshots, execution records, source hashes and service logs are written even when assertions fail. The runner closes its browser and services in `finally`. Always upload the artifact directory in CI. Default ports 5197/5198 must be free. For a local installed browser, set `SRT_BROWSER_CHANNEL=chrome`; `SRT_PLAYWRIGHT_MODULE` may point to an isolated Playwright `index.mjs` instead of adding a frontend dependency.

