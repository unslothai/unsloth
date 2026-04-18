# Document Pipeline Architecture

This is a small document ingestion and search system. Files come in, get parsed and validated, keywords get extracted, cross-references get built, and everything ends up queryable via a simple API.

## How data flows

Raw files on disk go through four stages before they are searchable.

**Parsing** reads the file, detects the format (markdown, JSON, plaintext), and converts it into a structured dict. The parser handles each format differently. Markdown gets title, sections, and links extracted. JSON gets loaded directly. Plaintext gets split into paragraphs.

**Validation** checks that the parsed document has the required fields and a known format. It also normalizes text fields (lowercase, trim whitespace, strip control characters) using the processor before the document moves forward.

**Processing** enriches the validated document with a keyword index and cross-references. Cross-references are built by comparing the document's keywords against every other document already in the index. If they share three or more keywords they get linked.

**Storage** persists everything to disk as JSON files and maintains a flat index that maps record IDs to metadata. All other modules read and write through the storage interface so there is one source of truth.

## Module responsibilities

- parser.py: reads files, detects format, calls validate_document and save_parsed
- validator.py: enforces schema, normalizes fields, calls normalize_text from processor
- processor.py: extract_keywords, find_cross_references, calls load_index and save_processed
- storage.py: load_index, save_parsed, save_processed, load_record, delete_record, list_records
- api.py: HTTP handlers that orchestrate the above modules

## Design decisions

The pipeline is intentionally linear. Each stage has one job and calls the next stage explicitly. There is no event bus or dependency injection. This makes the call graph easy to follow and easy to test.

Storage is intentionally simple. A flat JSON index plus one file per document is enough at small scale. If the corpus grows past a few thousand documents this becomes the bottleneck and should be replaced with SQLite or a proper document store.

Cross-reference detection is intentionally naive. Keyword overlap of three is a reasonable threshold for short documents but will produce too many false positives on long ones. A real system would use TF-IDF or embedding similarity instead.

## Extending the pipeline

To add a new file format, add a branch in parser.py's parse_file function and a new parse_* function. The rest of the pipeline does not need to change.

To add a new enrichment step, add a function in processor.py and call it from enrich_document. Store the result in the document dict and add the field to the index in save_processed if you want it searchable.
