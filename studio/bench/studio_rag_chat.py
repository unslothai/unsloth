#!/usr/bin/env python3
"""End-to-end RAG tool-call test against a running Studio with a GGUF loaded.
Asks gold questions grounded in an indexed KB, with rag_scope + tools enabled,
and reports whether the model called search_knowledge_base and answered correctly.
"""
import argparse
import json
import sys

import httpx

QUESTIONS = [
    ("What are BERT's two pre-training objectives?", ["masked", "next sentence"]),
    ("How many attention heads does the base Transformer use?", ["8", "eight"]),
    ("What retriever does the RAG paper use to fetch passages?", ["dpr", "dense passage", "bi-encoder"]),
]


def token(c, base, pw):
    new = pw + "Aa1!"
    r = c.post(f"{base}/api/auth/login", json={"username": "unsloth", "password": pw})
    if r.status_code == 401:
        r = c.post(f"{base}/api/auth/login", json={"username": "unsloth", "password": new})
    r.raise_for_status()
    b = r.json()
    t = b["access_token"]
    if b.get("must_change_password"):
        r2 = c.post(f"{base}/api/auth/change-password", headers={"Authorization": f"Bearer {t}"},
                    json={"current_password": pw, "new_password": new})
        r2.raise_for_status()
        t = r2.json()["access_token"]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--kb-name-contains", default="corpus")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {"base": args.base, "qa": []}
    with httpx.Client(timeout=300) as c:
        tok = token(c, args.base, args.password)
        H = {"Authorization": f"Bearer {tok}"}
        kbs = c.get(f"{args.base}/api/rag/knowledge-bases", headers=H).json()["knowledge_bases"]
        kb = next((k for k in kbs if args.kb_name_contains in k["name"]), kbs[0] if kbs else None)
        if not kb:
            sys.exit("no KB found")
        kb_id = kb["id"]
        print(f"KB: {kb['name']} ({kb_id})")

        for q, phrases in QUESTIONS:
            payload = {
                "model": "local",
                "messages": [{"role": "user", "content": q +
                              " Use the knowledge base; cite the source."}],
                "stream": False,
                "enable_tools": True,
                "enable_thinking": False,
                "max_tokens": 400,
                "temperature": 0.3,
                "rag_scope": {"kb_id": kb_id, "default_top_k": 5, "min_score": 0.0, "mode": "hybrid"},
            }
            # The endpoint streams SSE regardless of stream flag; parse it.
            answer, called_tool, tool_query, src = "", False, "", ""
            status = 0
            with c.stream("POST", f"{args.base}/api/inference/chat/completions",
                          headers=H, json=payload) as r:
                status = r.status_code
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except Exception:
                        continue
                    if ev.get("type") == "tool_start" and ev.get("tool_name") == "search_knowledge_base":
                        called_tool = True
                        tool_query = (ev.get("arguments") or {}).get("query", "")
                    elif ev.get("type") == "tool_end":
                        res = ev.get("result", "")
                        m = res.split('source="')
                        if len(m) > 1:
                            src = m[1].split('"')[0]
                    elif ev.get("choices"):
                        delta = ev["choices"][0].get("delta", {})
                        if delta.get("content"):
                            answer += delta["content"]
            hit = any(p in answer.lower() for p in phrases)
            print(f"\nQ: {q}\n  status={status} tool_called={called_tool} src={src} answer_has_fact={hit}")
            print(f"  tool_query={tool_query!r}")
            print(f"  A: {answer[:300].replace(chr(10),' ')}")
            results["qa"].append({"q": q, "status": status, "tool_called": called_tool,
                                  "retrieved_source": src, "tool_query": tool_query,
                                  "answer_has_fact": hit, "answer": answer[:600]})

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")
    n = len(results["qa"])
    print(f"\nSUMMARY: {sum(r['answer_has_fact'] for r in results['qa'])}/{n} answers contain the fact, "
          f"{sum(r['tool_called'] for r in results['qa'])}/{n} called the RAG tool")


if __name__ == "__main__":
    main()
