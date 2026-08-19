package codexhistory

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func call(callID string) CachedCall {
	return CachedCall{CallID: callID, Item: map[string]any{
		"type":    "function_call",
		"call_id": callID,
		"name":    "exec",
	}}
}

func TestStoreRecordAndLookupByResponse(t *testing.T) {
	s := NewStore(8)
	s.Record("resp_1", []CachedCall{call("call_a"), call("call_b")})

	got := s.Lookup("resp_1", map[string]bool{"call_b": true})
	assert.Len(t, got, 1)
	assert.Equal(t, "call_b", got[0].CallID)

	// 顺序保持原 response 的调用顺序，而非 requested 的迭代顺序。
	got = s.Lookup("resp_1", map[string]bool{"call_a": true, "call_b": true})
	assert.Len(t, got, 2)
	assert.Equal(t, "call_a", got[0].CallID)
	assert.Equal(t, "call_b", got[1].CallID)
}

func TestStoreLookupFallsBackToCallID(t *testing.T) {
	s := NewStore(8)
	s.Record("resp_1", []CachedCall{call("call_a")})

	// Codex subagent 会改写 previous_response_id，全局 call_id 兜底必须命中。
	got := s.Lookup("resp_rewritten", map[string]bool{"call_a": true})
	assert.Len(t, got, 1)
	assert.Equal(t, "call_a", got[0].CallID)
}

func TestStoreLookupEmptyRequested(t *testing.T) {
	s := NewStore(8)
	s.Record("resp_1", []CachedCall{call("call_a")})
	assert.Empty(t, s.Lookup("resp_1", nil))
}

func TestStoreEvictsOldestResponse(t *testing.T) {
	s := NewStore(2)
	s.Record("resp_1", []CachedCall{call("call_1")})
	s.Record("resp_2", []CachedCall{call("call_2")})
	s.Record("resp_3", []CachedCall{call("call_3")})

	assert.Empty(t, s.Lookup("resp_1", map[string]bool{"call_1": true}), "oldest response must be evicted")
	assert.Empty(t, s.Lookup("resp_3", map[string]bool{"call_1": true}), "evicted call_id fallback must be gone")
	assert.Len(t, s.Lookup("resp_3", map[string]bool{"call_3": true}), 1)
}

func TestStoreReRecordDoesNotDuplicateQueue(t *testing.T) {
	s := NewStore(2)
	s.Record("resp_1", []CachedCall{call("call_1")})
	s.Record("resp_2", []CachedCall{call("call_2")})
	// 重复记录同一 response 不应挤占队列位置导致 resp_1 被驱逐。
	s.Record("resp_1", []CachedCall{call("call_1b")})
	assert.Len(t, s.Lookup("resp_1", map[string]bool{"call_1b": true}), 1)
	assert.Len(t, s.Lookup("resp_2", map[string]bool{"call_2": true}), 1)
}

func TestStoreIgnoresEmptyRecord(t *testing.T) {
	s := NewStore(1)
	s.Record("", []CachedCall{call("call_a")})
	s.Record("resp_1", nil)
	for i := 0; i < 4; i++ {
		s.Record(fmt.Sprintf("resp_%d", i), []CachedCall{call(fmt.Sprintf("call_%d", i))})
	}
	assert.Len(t, s.Lookup("resp_3", map[string]bool{"call_3": true}), 1)
}
