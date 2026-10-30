---
name: solution-exact
kind: rule
description: The text inside <SOLUTION> equals the dataset's answer column (reasoning format).
---
type: exact_match
extract: {between: ['<SOLUTION>', '</SOLUTION>']}
compare_to: answer
reference_extract: {regex: '####\s*(.+?)\s*$'}
normalize: [strip, remove_commas, lower]
score: {match: 3.0, miss: 0.0}
missing: 0.0
