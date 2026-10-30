---
name: soft-xml-format
kind: rule
description: An <answer>...</answer> block appears anywhere in the reply.
---
type: regex
mode: search
pattern: '<answer>.*?</answer>'
score: {match: 0.5, miss: 0.0}
