---
name: strict-xml-format
kind: rule
description: The whole reply is <reasoning>...</reasoning> followed by <answer>...</answer>.
---
type: regex
mode: fullmatch
pattern: '<reasoning>.*?</reasoning>\s*<answer>.*?</answer>'
score: {match: 0.5, miss: 0.0}
