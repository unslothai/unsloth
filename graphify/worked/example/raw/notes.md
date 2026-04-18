# Research Notes

Thoughts and open questions while building the document pipeline. Not polished, just a running log.

## On keyword extraction

The current approach strips stopwords and returns unique tokens. Simple and fast. The problem is it treats all keywords equally. "database" appearing once in a title carries more weight than "database" buried in a paragraph but the code doesn't know that.

TF-IDF would fix this. Term frequency times inverse document frequency gives higher scores to words that are distinctive to a document rather than common across the corpus. Worth switching once the index is big enough for IDF to be meaningful (probably 50+ documents).

Embedding-based similarity is the other option. Run each document through a sentence transformer, store the vector, do nearest-neighbor search at query time. Much better recall but adds a dependency and makes the index opaque. The keyword approach is at least debuggable.

## On cross-reference detection

Three shared keywords is arbitrary. Tuned it by hand on a small test set. On short documents (under 500 words) it produces reasonable results. On long documents everything shares keywords with everything else and the cross-reference graph becomes noise.

A per-document threshold based on document length would be better. Or weight by keyword specificity so rare keywords count more than common ones.

## On storage

Flat files work fine for now. The index fits in memory. Load times are under 10ms for a few hundred documents.

SQLite becomes worth it when you need range queries or you want to update individual fields without rewriting the whole record. The current save_processed rewrites the entire JSON file on every update which is wasteful.

One thing flat files do well: they are easy to inspect. Open the store directory and you can read every document directly. No tooling required. This matters for debugging.

## On the API layer

The API is a thin wrapper. Every handler does one thing: call the right combination of parser, validator, processor, storage. No business logic lives in api.py.

The risk is that this breaks down when you need transactions. Right now parse_and_save in parser.py calls validate_document and save_parsed in sequence. If save_parsed fails after validate_document succeeds you have a partially written record. Not a problem at small scale, becomes a problem under load.

## Open questions

Should validation happen in the parser or as a separate step? Currently it's separate which means the parser can return invalid documents. That feels wrong but keeping them separate makes each module easier to test.

Should cross-references be stored on the document or computed at query time? Storing them is fast to read but goes stale. Computing at query time is always fresh but slow for large indexes.

Is the storage interface the right abstraction? Right now parser, validator, and processor all import from storage directly. A repository pattern would centralize access but adds indirection. Probably not worth it until the storage backend needs to change.
