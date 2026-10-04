---
name: solution-format
kind: rule
description: The reply closes <end_working_out> and ends with <SOLUTION>...</SOLUTION> (reasoning format).
---
type: regex
mode: search
pattern: '<end_working_out>.*?<SOLUTION>.+?</SOLUTION>\s*$'
score: {match: 3.0, miss: 0.0}
