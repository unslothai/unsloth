# Graph Report - worked/mixed-corpus/raw  (2026-04-05)

## Corpus Check
- 4 files · ~2,500 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 22 nodes · 38 edges · 5 communities detected
- Extraction: 50% EXTRACTED · 50% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `_cross_file_surprises()` - 7 edges
2. `_is_file_node()` - 5 edges
3. `_cross_community_surprises()` - 5 edges
4. `_node_community_map()` - 4 edges
5. `_is_concept_node()` - 4 edges
6. `_surprise_score()` - 4 edges
7. `suggest_questions()` - 4 edges
8. `god_nodes()` - 3 edges
9. `surprising_connections()` - 3 edges
10. `_file_category()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `suggest_questions()` --calls--> `_node_community_map()`  [INFERRED]
  worked/mixed-corpus/raw/analyze.py → worked/mixed-corpus/raw/analyze.py  _Bridges community 3 → community 2_
- `_cross_file_surprises()` --calls--> `_surprise_score()`  [INFERRED]
  worked/mixed-corpus/raw/analyze.py → worked/mixed-corpus/raw/analyze.py  _Bridges community 1 → community 3_

## Communities

### Community 0 - "Community 0"
Cohesion: 0.47
Nodes (4): cluster(), cohesion_score(), score_all(), _split_community()

### Community 1 - "Community 1"
Cohesion: 0.6
Nodes (3): _file_category(), _surprise_score(), _top_level_dir()

### Community 2 - "Community 2"
Cohesion: 0.67
Nodes (4): god_nodes(), _is_concept_node(), _is_file_node(), suggest_questions()

### Community 3 - "Community 3"
Cohesion: 0.83
Nodes (4): _cross_community_surprises(), _cross_file_surprises(), _node_community_map(), surprising_connections()

### Community 4 - "Community 4"
Cohesion: 1.0
Nodes (2): build(), build_from_json()

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `_cross_file_surprises()` connect `Community 3` to `Community 1`, `Community 2`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **Why does `_is_file_node()` connect `Community 2` to `Community 1`, `Community 3`?**
  _High betweenness centrality (0.008) - this node is a cross-community bridge._
- **Why does `_surprise_score()` connect `Community 1` to `Community 3`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `_cross_file_surprises()` (e.g. with `surprising_connections()` and `_node_community_map()`) actually correct?**
  _`_cross_file_surprises()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `_is_file_node()` (e.g. with `god_nodes()` and `_cross_file_surprises()`) actually correct?**
  _`_is_file_node()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `_cross_community_surprises()` (e.g. with `surprising_connections()` and `_cross_file_surprises()`) actually correct?**
  _`_cross_community_surprises()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `_node_community_map()` (e.g. with `_cross_file_surprises()` and `_cross_community_surprises()`) actually correct?**
  _`_node_community_map()` has 3 INFERRED edges - model-reasoned connections that need verification._