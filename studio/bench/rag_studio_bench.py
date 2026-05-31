#!/usr/bin/env python3
"""Benchmark a running Studio's RAG over its real HTTP API: indexing latency,
scaling, and retrieval accuracy. Same script drives baseline and improved.

Usage:
  python rag_studio_bench.py --base http://127.0.0.1:8901 --label baseline \
     --password "<bootstrap>" --corpus ../data/rag_corpus --out ../logs/bench_baseline.json
"""
import argparse
import json
import time
from pathlib import Path

import httpx

API = "/api/rag"

# Gold queries: (query, filename_substring, [accepted answer phrases any-of]).
# A hit counts when a retrieved chunk is FROM the right document AND contains an
# accepted phrase, so retrieval is scored independently of the chat model.
GOLD = [
    ("What are the sinusoidal positional encodings based on?", "attention",
     ["sine", "sinusoid", "wavelength"]),
    ("How many attention heads does the base Transformer use?", "attention",
     ["eight", "h = 8", "8 parallel", "8 attention", "h=8"]),
    ("What are BERT's two pre-training objectives?", "bert",
     ["masked", "next sentence"]),
    ("How many Transformer layers does BERT-large have?", "bert",
     ["24", "l = 24", "l=24"]),
    ("Difference between RAG-Sequence and RAG-Token models?", "rag",
     ["rag-token", "rag-sequence"]),
    ("Which retriever does RAG use to fetch passages?", "rag",
     ["dpr", "dense passage", "bi-encoder", "mips"]),
    ("What does the HTTP GET method do?", "rfc9110",
     ["transfer a current representation", "retrieve", "selector"]),
    ("Which HTTP status code means the resource was not found?", "rfc9110",
     ["404"]),
]


def login(c, base, user, pw):
    new = pw + "Aa1!"
    # Re-run safe: bootstrap pw works first time; after we change it, the changed
    # pw works on later runs.
    r = c.post(f"{base}/api/auth/login", json={"username": user, "password": pw})
    if r.status_code == 401:
        r = c.post(f"{base}/api/auth/login", json={"username": user, "password": new})
    r.raise_for_status()
    body = r.json()
    tok = body["access_token"]
    if body.get("must_change_password"):
        r2 = c.post(f"{base}/api/auth/change-password", headers=H(tok),
                    json={"current_password": pw, "new_password": new})
        r2.raise_for_status()
        tok = r2.json()["access_token"]
    return tok


def H(tok):
    return {"Authorization": f"Bearer {tok}"}


def warmup(c, base, tok):
    t = time.perf_counter()
    try:
        c.post(f"{base}{API}/warmup", headers=H(tok), timeout=600)
    except Exception as e:
        print("warmup err:", e)
    return time.perf_counter() - t


def create_kb(c, base, tok, name):
    r = c.post(f"{base}{API}/knowledge-bases", headers=H(tok),
               json={"name": name, "mode": "text", "chunking_strategy": "standard"})
    r.raise_for_status()
    return r.json()["kb_id" if "kb_id" in r.json() else "id"]


def upload(c, base, tok, kb_id, path: Path):
    with open(path, "rb") as f:
        r = c.post(f"{base}{API}/knowledge-bases/{kb_id}/documents",
                   headers=H(tok), files={"file": (path.name, f, "application/octet-stream")},
                   timeout=600)
    r.raise_for_status()
    return r.json()


def wait_indexed(c, base, tok, kb_id, document_id, timeout=600):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        r = c.get(f"{base}{API}/knowledge-bases/{kb_id}/documents", headers=H(tok))
        r.raise_for_status()
        for d in r.json()["documents"]:
            if d["id"] == document_id:
                if d["status"] == "completed":
                    return time.perf_counter() - t0, d["num_chunks"], "completed"
                if d["status"] == "failed":
                    return time.perf_counter() - t0, 0, "failed"
        time.sleep(0.1)
    return timeout, 0, "timeout"


def index_doc(c, base, tok, kb_id, path):
    t0 = time.perf_counter()
    up = upload(c, base, tok, kb_id, path)
    if up.get("already_indexed"):
        return {"file": path.name, "elapsed_s": 0.0, "chunks": 0, "status": "dup"}
    elapsed, chunks, status = wait_indexed(c, base, tok, kb_id, up["document_id"])
    total = time.perf_counter() - t0
    return {"file": path.name, "elapsed_s": round(total, 3), "chunks": chunks,
            "status": status, "document_id": up["document_id"]}


def search(c, base, tok, kb_id, query, mode, top_k=10):
    r = c.post(f"{base}{API}/search", headers=H(tok),
               json={"query": query, "kb_id": kb_id, "mode": mode, "top_k": top_k})
    r.raise_for_status()
    return r.json()["hits"]


def score_mode(c, base, tok, kb_id, mode):
    r1 = r3 = r5 = 0
    mrr = 0.0
    lat = []
    for query, fsub, phrases in GOLD:
        t = time.perf_counter()
        hits = search(c, base, tok, kb_id, query, mode, top_k=10)
        lat.append((time.perf_counter() - t) * 1000)
        rank = None
        for i, h in enumerate(hits):
            fn = (h.get("filename") or "").lower()
            txt = (h.get("text") or "").lower()
            if fsub in fn and any(p in txt for p in phrases):
                rank = i
                break
        if rank is not None:
            if rank == 0:
                r1 += 1
            if rank < 3:
                r3 += 1
            if rank < 5:
                r5 += 1
            mrr += 1.0 / (rank + 1)
    n = len(GOLD)
    lat.sort()
    return {"recall@1": round(r1 / n, 3), "recall@3": round(r3 / n, 3),
            "recall@5": round(r5 / n, 3), "mrr": round(mrr / n, 3),
            "search_ms_median": round(lat[len(lat) // 2], 2)}


def make_synthetic(dirpath: Path, n: int):
    dirpath.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = dirpath / f"syn_{i:02d}.txt"
        body = (f"Synthetic document number {i}. Project codename Orbit-{i} concerns "
                f"widget {i} calibration at {100 + i} hertz. Unique token zglyph{i} marks "
                f"this file. " * 8)
        p.write_text(body)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--username", default="unsloth")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scaling-n", type=int, default=8)
    args = ap.parse_args()

    res = {"label": args.label, "base": args.base}
    with httpx.Client(timeout=120) as c:
        tok = login(c, args.base, args.username, args.password)
        res["warmup_s"] = round(warmup(c, args.base, tok), 2)

        # --- Real corpus: index timing (first = cold, rest = warm) + accuracy ---
        corpus = sorted(Path(args.corpus).glob("*"))
        corpus = [p for p in corpus if p.suffix.lower() in
                  (".pdf", ".txt", ".md", ".html", ".htm", ".docx")]
        kb = create_kb(c, args.base, tok, f"{args.label}-corpus")
        res["kb"] = kb
        res["corpus_index"] = []
        for i, p in enumerate(corpus):
            r = index_doc(c, args.base, tok, kb, p)
            r["which"] = "cold" if i == 0 else "warm"
            res["corpus_index"].append(r)
            print(f"[{args.label}] index {p.name}: {r['elapsed_s']}s ({r['chunks']} chunks) {r['which']}")

        res["accuracy"] = {m: score_mode(c, args.base, tok, kb, m)
                           for m in ("bm25", "dense", "hybrid")}
        for m, s in res["accuracy"].items():
            print(f"[{args.label}] {m}: R@1={s['recall@1']} R@5={s['recall@5']} MRR={s['mrr']}")

        # --- Scaling: N small docs into one fresh KB, per-doc index time ---
        kb2 = create_kb(c, args.base, tok, f"{args.label}-scaling")
        syn = make_synthetic(Path(args.corpus).parent / "rag_synthetic", args.scaling_n)
        res["scaling"] = []
        for i, p in enumerate(syn):
            r = index_doc(c, args.base, tok, kb2, p)
            res["scaling"].append({"n": i + 1, "elapsed_s": r["elapsed_s"]})
            print(f"[{args.label}] scaling doc {i+1}/{len(syn)}: {r['elapsed_s']}s")

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"[{args.label}] wrote {args.out}")


if __name__ == "__main__":
    main()
