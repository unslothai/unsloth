---
name: strict-xml-format
kind: rule
description: The whole reply is <reasoning>...</reasoning> followed by <answer>...</answer>.
---
type: regex
mode: fullmatch
pattern: '<reasoning>\s*.*?\s*</reasoning>\s*<answer>\s*.*?\s*</answer>'
score: {match: 0.5, miss: 0.0}
