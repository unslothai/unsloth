---
name: exact-answer
kind: rule
description: The text inside <answer> equals the dataset's answer column.
---
type: exact_match
extract: {between: ['<answer>', '</answer>']}
compare_to: answer
reference_extract: {regex: '####\s*(.+?)\s*$'}
normalize: [strip, remove_commas, lower]
score: {match: 2.0, miss: 0.0}
missing: 0.0
