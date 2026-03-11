#!/usr/bin/env python3
"""Quick check of activation_log.jsonl structure."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/leo/qwen3_activation_logs/activation_log.jsonl"

with open(path) as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        print(f"Step {d['step']}: keys={list(d.keys())}")
        if 'grad_norms' in d:
            gn = d['grad_norms']
            print(f"  grad_norms: {len(gn)} layers, sample: layer 0 = {gn.get('0', 'N/A'):.4f}" if gn.get('0') else f"  grad_norms: {len(gn)} layers")
        if 'lora_norms' in d:
            ln = d['lora_norms']
            sample_keys = list(ln.keys())[:4]
            print(f"  lora_norms: {len(ln)} entries, sample: {sample_keys}")
        if i >= 2:
            break
