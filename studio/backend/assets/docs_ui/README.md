# Swagger UI and ReDoc, served from this origin

FastAPI's built-in `/docs` and `/redoc` pages load their bundles from `cdn.jsdelivr.net` and
start Swagger with an inline script. Those pages sit on the same origin as the frontend, and
`localStorage` is origin-scoped rather than path-scoped, so anything executing there can read
the access and refresh tokens `frontend/src/features/auth/session.ts` stores and call the API
as that user. `main.py` re-registers both pages on FastAPI's own paths against the files in
this directory, so `script-src` stays `'self'` and the docs work with no network.

| file | package | version | licence |
| --- | --- | --- | --- |
| `swagger-ui-bundle.js`, `swagger-ui.css`, `favicon-32x32.png` | `swagger-ui-dist` | 5.30.2 | Apache-2.0 (`LICENSE.swagger-ui`) |
| `redoc.standalone.js` | `redoc` | 2.5.1 | MIT (`LICENSE.redoc`) |

The bytes are the published releases, unmodified, so they carry no Unsloth licence header.
`tests/test_docs_ui_assets.py` pins every file by sha256 against `docs_ui_manifest.json`.

## Bumping a version

1. Download the files for the new version from `https://cdn.jsdelivr.net/npm/<package>@<version>/...`.
2. Refresh the version, source URL and every sha256 in `docs_ui_manifest.json`.
3. Run `pytest tests/test_docs_ui_assets.py tests/test_middleware.py` and open `/docs` and
   `/redoc` in a browser. Swagger's inline init is matched by a marker in `main.py`; if
   upstream retemplates that page the route raises rather than serving a blank one.
