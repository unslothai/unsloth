package nlp

import (
	"reflect"
	"testing"
)

func TestPruneRetrievalSearchResultPreservesRawChunkIdentity(t *testing.T) {
	keptRaw := map[string]interface{}{
		"_id": "chunk-kept", "_index": "ragflow_tenant", "doc_id": "doc-kept",
	}
	staleRaw := map[string]interface{}{
		"_id": "chunk-stale", "_index": "ragflow_tenant", "doc_id": "doc-stale",
	}
	result := &RetrievalSearchResult{
		Chunks: []map[string]interface{}{keptRaw, staleRaw},
		IDs:    []string{"chunk-kept", "chunk-stale"},
		Field: map[string]map[string]interface{}{
			"chunk-kept":  {"doc_id": "doc-kept", "content_ltks": "kept"},
			"chunk-stale": {"doc_id": "doc-stale", "content_ltks": "stale"},
		},
		Highlight:  map[string]string{"chunk-kept": "kept", "chunk-stale": "stale"},
		IndexNames: []string{"ragflow_tenant"},
	}

	filtered, removed := pruneRetrievalSearchResult(result, map[string]struct{}{"doc-kept": {}})
	if removed != 1 {
		t.Fatalf("removed=%d, want 1", removed)
	}
	if len(filtered.Chunks) != 1 || filtered.Chunks[0]["_id"] != "chunk-kept" || filtered.Chunks[0]["_index"] != "ragflow_tenant" {
		t.Fatalf("raw chunk identity not preserved: %#v", filtered.Chunks)
	}
	if !reflect.DeepEqual(filtered.IndexNames, result.IndexNames) {
		t.Fatalf("index names=%v, want %v", filtered.IndexNames, result.IndexNames)
	}
	if _, exists := filtered.Field["chunk-stale"]; exists {
		t.Fatalf("stale field entry was not removed")
	}
}
