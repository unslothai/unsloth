"""Make a copy of a checkpoint directory whose tokenizer pad token is not EOS (weights are symlinked).

usage: python make_pad_overlay.py <src_model_dir> <dst_dir> [pad_token_id]

DeepSeek-V2-Lite ships pad_token == eos_token and only two special tokens, so unsloth_zoo's
fix_pad_token cannot find a replacement and raises. The CP runs never pad, so BOS (or the given id)
is used as pad. Only tokenizer_config.json and config.json are rewritten; the source is untouched.
"""
import json
import os
import sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok = True)
    for name in os.listdir(src):
        if name in ("tokenizer_config.json", "config.json"):
            continue
        link = os.path.join(dst, name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.join(src, name), link)

    tok = json.load(open(os.path.join(src, "tokenizer_config.json")))
    cfg = json.load(open(os.path.join(src, "config.json")))
    pad_id = int(sys.argv[3]) if len(sys.argv) > 3 else cfg.get("bos_token_id", 0)
    bos = tok.get("bos_token")
    if len(sys.argv) <= 3 and bos is not None:
        tok["pad_token"] = dict(bos) if isinstance(bos, dict) else bos
    cfg["pad_token_id"] = pad_id
    with open(os.path.join(dst, "tokenizer_config.json"), "w") as f:
        json.dump(tok, f, ensure_ascii = False, indent = 1)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent = 1)
    print(f"overlay {dst}: pad_token_id={pad_id}")


if __name__ == "__main__":
    main()
