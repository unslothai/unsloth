---
name: solution-numeric
kind: rule
description: The number inside <SOLUTION> matches the dataset's answer (reasoning format).
---
type: numeric
extract: {between: ['<SOLUTION>', '</SOLUTION>']}
compare_to: answer
reference_extract: {regex: '####\s*(.+?)\s*$'}
bands:
  - {within: 0.0, score: 3.5}
else: -1.5
missing: -2.5
