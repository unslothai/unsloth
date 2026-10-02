---
name: numeric-close
kind: rule
description: Partial credit when the number inside <answer> is close to the dataset's answer.
---
type: numeric
extract: {between: ['<answer>', '</answer>']}
compare_to: answer
reference_extract: {regex: '####\s*(.+?)\s*$'}
bands:
  - {within: 0.0, score: 3.0}
  - {within: 0.1, score: 1.5}
  - {within: 0.2, score: 0.5}
else: -1.0
missing: -2.0
