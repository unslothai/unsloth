# SPDX-License-Identifier: AGPL-3.0-only
import argparse
import http.server
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid


def prepare(source):
    frontend = source / "studio/frontend"
    (frontend / "native-parallel.tsx").write_text(
        Path(__file__).with_name("main.tsx").read_text(encoding = "utf-8"), encoding = "utf-8"
    )
    (frontend / "index.html").write_text(
        '<html><head><meta charset="utf-8"></head><body><div id="root"></div><script type="module" src="/native-parallel.tsx"></script></body></html>',
        encoding = "utf-8",
    )


def run(binary, out):
    out.mkdir(parents = True, exist_ok = True)
    results = {}
    finished = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(
            self,
            data,
            kind = "application/json",
        ):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self.send(b"")

        def do_GET(self):
            if self.path.startswith("/result/"):
                self.send(
                    json.dumps(
                        results.get(
                            self.path.rsplit("/", 1)[-1],
                            {"ok": False, "error": "No result was posted"},
                        )
                    ).encode()
                )
                return
            letter = "A" if self.path == "/A" else "B"
            token = str(uuid.uuid4())
            html = f"""<html><body style="font:20px sans-serif;background:{'#e7f1ff' if letter == 'A' else '#e8ffed'}"><h1>Native {letter}</h1><p>TOKEN:{token}</p><label>Draft note<input id="note" value="{letter} original"></label><button onclick="document.getElementById('applied').textContent='Applied: '+document.getElementById('note').value">Apply note</button><p id="applied">Applied:</p></body></html>"""
            self.send(html.encode(), "text/html")

        def do_POST(self):
            data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path == "/api/inference/browser-action-result":
                item = json.loads(data)
                results[item["action_id"]] = item["result"]
            elif self.path == "/progress":
                with (out / "progress.jsonl").open("ab") as file:
                    file.write(data + b"\n")
                print(data.decode(), flush = True)
            elif self.path in ("/capture-0.png", "/capture-1.png", "/capture-2.png"):
                (out / self.path[1:]).write_bytes(data)
            elif self.path == "/complete":
                (out / "result.json").write_bytes(data)
                finished.set()
            self.send(b"{}")

    servers = [
        http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler) for port in (19044, 19045)
    ]
    for server in servers:
        threading.Thread(target = server.serve_forever, daemon = True).start()
    env = dict(
        os.environ, UNSLOTH_DISABLE_UPDATE_CHECK = "1", UNSLOTH_STUDIO_DISABLE_PUBLIC_CHECK = "1"
    )
    log = (out / "app.log").open("wb")
    process = subprocess.Popen(
        [str(binary.resolve())],
        stdout = log,
        stderr = subprocess.STDOUT,
        env = env,
        start_new_session = os.name != "nt",
    )
    try:
        end = time.monotonic() + 300
        while not finished.wait(1):
            if process.poll() is not None:
                raise RuntimeError(f"Desktop exited before completion: {process.returncode}")
            if time.monotonic() >= end:
                raise TimeoutError("Native browser probe did not report within 300 seconds")
        result = json.loads((out / "result.json").read_text(encoding = "utf-8"))
        print(json.dumps(result), flush = True)
        return 0 if result["passed"] else 1
    except Exception as error:
        (out / "infra.json").write_text(json.dumps({"error": str(error)}), encoding = "utf-8")
        raise
    finally:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check = False)
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout = 15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
        for server in servers:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type = Path)
    parser.add_argument("--bin", type = Path)
    parser.add_argument("--out", type = Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    else:
        raise SystemExit(run(args.bin, args.out))
