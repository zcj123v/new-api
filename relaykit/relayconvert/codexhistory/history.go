// Package codexhistory 提供 Codex Responses 会话的工具调用历史缓存。
//
// Codex 续轮可能只带 previous_response_id + 新的 *_output 项，而把对应的
// function_call/custom_tool_call/tool_search_call 项省略掉；chat 上游需要
// 看到完整的 assistant tool_calls，否则输出没有归属。响应方向把每个
// response 的工具调用项按 response id 缓存，请求方向据此补回。
//
// 参考 CC Switch 的 codex_chat_history.rs（MIT）。
package codexhistory

import "sync"

const defaultCapacity = 512

// CachedCall 是一条 Responses 格式的工具调用项（function_call /
// custom_tool_call / tool_search_call 的原始 map 形状）。
type CachedCall struct {
	CallID string
	Item   map[string]any
}

type Store struct {
	mu         sync.Mutex
	capacity   int
	byResponse map[string][]CachedCall
	responseQ  []string // FIFO
	byCallID   map[string]CachedCall
}

func NewStore(capacity int) *Store {
	if capacity <= 0 {
		capacity = defaultCapacity
	}
	return &Store{
		capacity:   capacity,
		byResponse: make(map[string][]CachedCall),
		byCallID:   make(map[string]CachedCall),
	}
}

var global = NewStore(defaultCapacity)

// Record 记录一个 response 的工具调用项（进程级缓存）。
func Record(responseID string, calls []CachedCall) {
	global.Record(responseID, calls)
}

// Lookup 返回可恢复的工具调用项：优先 previous_response_id 命中的
// response，再用全局 call_id 兜底（Codex subagent 会改写
// previous_response_id）。返回顺序保持原 response 的调用顺序。
func Lookup(previousResponseID string, requested map[string]bool) []CachedCall {
	return global.Lookup(previousResponseID, requested)
}

func (s *Store) Record(responseID string, calls []CachedCall) {
	if responseID == "" || len(calls) == 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.byResponse[responseID]; !exists {
		s.responseQ = append(s.responseQ, responseID)
	}
	s.byResponse[responseID] = calls
	for _, call := range calls {
		if call.CallID != "" {
			s.byCallID[call.CallID] = call
		}
	}
	for len(s.responseQ) > s.capacity {
		evict := s.responseQ[0]
		s.responseQ = s.responseQ[1:]
		for _, call := range s.byResponse[evict] {
			delete(s.byCallID, call.CallID)
		}
		delete(s.byResponse, evict)
	}
}

func (s *Store) Lookup(previousResponseID string, requested map[string]bool) []CachedCall {
	if len(requested) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]CachedCall, 0, len(requested))
	seen := make(map[string]bool, len(requested))
	if previousResponseID != "" {
		for _, call := range s.byResponse[previousResponseID] {
			if requested[call.CallID] && !seen[call.CallID] {
				seen[call.CallID] = true
				out = append(out, call)
			}
		}
	}
	for callID := range requested {
		if seen[callID] {
			continue
		}
		if call, ok := s.byCallID[callID]; ok {
			seen[callID] = true
			out = append(out, call)
		}
	}
	return out
}
