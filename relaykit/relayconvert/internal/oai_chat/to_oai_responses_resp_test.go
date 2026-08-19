package oaichat

import (
	"testing"

	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/relayconvert/codexhistory"
	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChatCompletionsResponseToResponsesPreservesTextToolCallsAndUsage(t *testing.T) {
	chat := &dto.OpenAITextResponse{
		Id:      "chatcmpl_1",
		Model:   "gpt-test",
		Created: 456,
		Choices: []dto.OpenAITextResponseChoice{
			{
				Message:      assistantMessageWithTool("I will call.", "call_1", "lookup", `{"q":"x"}`),
				FinishReason: "tool_calls",
			},
		},
		Usage: dto.Usage{PromptTokens: 3, CompletionTokens: 5, TotalTokens: 8},
	}

	resp, usage, err := ChatCompletionsResponseToResponsesResponse(chat, "resp_1")
	require.NoError(t, err)
	require.NotNil(t, usage)

	assert.Equal(t, "resp_1", resp.ID)
	assert.Equal(t, "response", resp.Object)
	assert.Equal(t, `"completed"`, string(resp.Status))
	assert.Equal(t, 3, resp.Usage.InputTokens)
	assert.Equal(t, 5, resp.Usage.OutputTokens)
	require.Len(t, resp.Output, 2)
	assert.Equal(t, responsesOutputTypeMessage, resp.Output[0].Type)
	assert.Equal(t, "I will call.", resp.Output[0].Content[0].Text)
	assert.Equal(t, responsesOutputTypeFunctionCall, resp.Output[1].Type)
	assert.Equal(t, "call_1", resp.Output[1].CallId)
	assert.Equal(t, "lookup", resp.Output[1].Name)
	assert.Equal(t, `"{\"q\":\"x\"}"`, string(resp.Output[1].Arguments))
}

func TestChatCompletionsResponseToResponsesMapsIncompleteFinishReasons(t *testing.T) {
	tests := []struct {
		name         string
		finishReason string
		wantReason   string
	}{
		{name: "length", finishReason: "length", wantReason: responsesIncompleteReasonMaxTokens},
		{name: "content filter", finishReason: "content_filter", wantReason: responsesIncompleteReasonContentFilter},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, _, err := ChatCompletionsResponseToResponsesResponse(&dto.OpenAITextResponse{
				Id:    "chatcmpl_1",
				Model: "gpt-test",
				Choices: []dto.OpenAITextResponseChoice{
					{
						Message:      dto.Message{Role: "assistant", Content: "partial"},
						FinishReason: tt.finishReason,
					},
				},
			}, "resp_1")
			require.NoError(t, err)

			assert.Equal(t, `"incomplete"`, string(resp.Status))
			require.NotNil(t, resp.IncompleteDetails)
			assert.Equal(t, tt.wantReason, resp.IncompleteDetails.Reason)
			require.Len(t, resp.Output, 1)
			assert.Equal(t, "incomplete", resp.Output[0].Status)
		})
	}
}

func TestChatCompletionsStreamToResponsesEventsAggregatesUsageAndToolArgs(t *testing.T) {
	state := NewChatToResponsesStreamState("resp_1", "gpt-test")
	state.Created = 123
	toolIndex := 0

	var events []ChatToResponsesStreamEvent
	events = append(events, mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Id:      "chatcmpl_1",
		Model:   "gpt-test",
		Created: 123,
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, Delta: dto.ChatCompletionsStreamResponseChoiceDelta{Role: "assistant"}},
		},
	})...)
	events = append(events, mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, Delta: dto.ChatCompletionsStreamResponseChoiceDelta{Content: lo.ToPtr("hello")}},
		},
	})...)
	events = append(events, mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, Delta: dto.ChatCompletionsStreamResponseChoiceDelta{ToolCalls: []dto.ToolCallResponse{
				{Index: &toolIndex, ID: "call_1", Type: "function", Function: dto.FunctionResponse{Name: "lookup"}},
			}}},
		},
	})...)
	events = append(events, mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, Delta: dto.ChatCompletionsStreamResponseChoiceDelta{ToolCalls: []dto.ToolCallResponse{
				{Index: &toolIndex, Function: dto.FunctionResponse{Arguments: `{"q":"x"}`}},
			}}},
		},
	})...)
	finishReason := "tool_calls"
	events = append(events, mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, FinishReason: &finishReason},
		},
	})...)
	events = append(events, mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Usage: &dto.Usage{PromptTokens: 2, CompletionTokens: 4, TotalTokens: 6},
	})...)
	events = append(events, FinalizeChatCompletionsStreamToResponses(state)...)

	require.Len(t, events, 10)
	assert.Equal(t, responsesEventCreated, events[0].Type)
	assert.Equal(t, responsesEventOutputTextDelta, events[2].Type)
	assert.Equal(t, "hello", events[2].Payload.Delta)
	assert.Equal(t, responsesEventFunctionArgsDelta, events[4].Type)
	assert.Equal(t, `{"q":"x"}`, events[4].Payload.Delta)
	assert.Equal(t, responsesEventCompleted, events[9].Type)
	require.NotNil(t, events[9].Payload.Response)
	assert.Equal(t, 6, events[9].Payload.Response.Usage.TotalTokens)
	require.Len(t, events[9].Payload.Response.Output, 2)
	assert.Equal(t, "hello", events[9].Payload.Response.Output[0].Content[0].Text)
	assert.Equal(t, `"{\"q\":\"x\"}"`, string(events[9].Payload.Response.Output[1].Arguments))
}

func mustResponsesEventsFromChatChunk(t *testing.T, state *ChatToResponsesStreamState, chunk *dto.ChatCompletionsStreamResponse) []ChatToResponsesStreamEvent {
	t.Helper()
	events, err := ChatCompletionsStreamChunkToResponsesEvents(chunk, state)
	require.NoError(t, err)
	return events
}

// Upstream OpenAI always emits output_tokens_details in Responses usage
// (reasoning_tokens may be 0) and strict clients such as Codex require the
// field to exist, so UsageFromChatUsage must populate it even when the chat
// usage carries no reasoning tokens at all (e.g. Kimi local counting).
func TestUsageFromChatUsageAlwaysEmitsOutputTokensDetails(t *testing.T) {
	tests := []struct {
		name          string
		src           dto.Usage
		wantReasoning int
	}{
		{
			name:          "top-level reasoning tokens",
			src:           dto.Usage{PromptTokens: 3, CompletionTokens: 5, TotalTokens: 8, ReasoningTokens: 2},
			wantReasoning: 2,
		},
		{
			name:          "completion tokens details fallback",
			src:           dto.Usage{PromptTokens: 3, CompletionTokens: 5, TotalTokens: 8, CompletionTokenDetails: dto.OutputTokenDetails{ReasoningTokens: 4}},
			wantReasoning: 4,
		},
		{
			name:          "no reasoning tokens still emits zero details",
			src:           dto.Usage{PromptTokens: 3, CompletionTokens: 5, TotalTokens: 8},
			wantReasoning: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			usage := UsageFromChatUsage(&tt.src)
			require.NotNil(t, usage)
			require.NotNil(t, usage.OutputTokensDetails,
				"output_tokens_details must always be present")
			assert.Equal(t, tt.wantReasoning, usage.OutputTokensDetails.ReasoningTokens)
			assert.Equal(t, tt.wantReasoning, usage.ReasoningTokens)
		})
	}
}

func TestUsageFromChatUsagePreservesExistingOutputTokensDetails(t *testing.T) {
	src := &dto.Usage{
		PromptTokens:        3,
		CompletionTokens:    5,
		TotalTokens:         8,
		OutputTokensDetails: &dto.OutputTokenDetails{ReasoningTokens: 7},
	}
	usage := UsageFromChatUsage(src)
	require.NotNil(t, usage)
	require.NotNil(t, usage.OutputTokensDetails)
	assert.Equal(t, 7, usage.OutputTokensDetails.ReasoningTokens)
}

// 伪装成 function 的 freeform 工具（Codex apply_patch）：命中 customTools
// 名字的 function_call 必须还原为 custom_tool_call，arguments 解包回 input。
func TestChatCompletionsResponseToResponsesMapsCustomToolCall(t *testing.T) {
	chat := &dto.OpenAITextResponse{
		Id:    "chatcmpl_1",
		Model: "gpt-test",
		Choices: []dto.OpenAITextResponseChoice{
			{
				Message:      assistantMessageWithTool("", "call_1", "apply_patch", `{"input":"*** Begin Patch"}`),
				FinishReason: "tool_calls",
			},
		},
	}

	resp, _, err := ChatCompletionsResponseToResponsesResponseWithCustomTools(chat, "resp_1", map[string]bool{"apply_patch": true}, false)
	require.NoError(t, err)
	require.Len(t, resp.Output, 1)
	assert.Equal(t, responsesOutputTypeCustomToolCall, resp.Output[0].Type)
	assert.Equal(t, "call_1", resp.Output[0].CallId)
	assert.Equal(t, "apply_patch", resp.Output[0].Name)
	assert.Equal(t, "*** Begin Patch", resp.Output[0].Input)
	assert.Empty(t, resp.Output[0].Arguments)

	// 不在 customTools 里的同名调用保持 function_call（nil map 也安全）
	resp2, _, err := ChatCompletionsResponseToResponsesResponseWithCustomTools(chat, "resp_1", nil, false)
	require.NoError(t, err)
	require.Len(t, resp2.Output, 1)
	assert.Equal(t, responsesOutputTypeFunctionCall, resp2.Output[0].Type)
}

// 流式方向：custom 工具不发 function_call_arguments.*，完成时一次性发
// response.custom_tool_call_input.done + custom_tool_call 的 output_item.done。
func TestChatCompletionsStreamToResponsesEventsCustomToolCall(t *testing.T) {
	state := NewChatToResponsesStreamState("resp_1", "gpt-test")
	state.CustomTools = map[string]bool{"apply_patch": true}
	toolIndex := 0

	var events []ChatToResponsesStreamEvent
	events = append(events, mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, Delta: dto.ChatCompletionsStreamResponseChoiceDelta{ToolCalls: []dto.ToolCallResponse{
				{Index: &toolIndex, ID: "call_1", Type: "function", Function: dto.FunctionResponse{Name: "apply_patch"}},
			}}},
		},
	})...)
	events = append(events, mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, Delta: dto.ChatCompletionsStreamResponseChoiceDelta{ToolCalls: []dto.ToolCallResponse{
				{Index: &toolIndex, Function: dto.FunctionResponse{Arguments: `{"input":"*** Begin`}},
			}}},
		},
	})...)
	events = append(events, mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, Delta: dto.ChatCompletionsStreamResponseChoiceDelta{ToolCalls: []dto.ToolCallResponse{
				{Index: &toolIndex, Function: dto.FunctionResponse{Arguments: ` Patch"}`}},
			}}},
		},
	})...)
	finishReason := "tool_calls"
	events = append(events, mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, FinishReason: &finishReason},
		},
	})...)
	events = append(events, FinalizeChatCompletionsStreamToResponses(state)...)

	for _, e := range events {
		assert.NotEqual(t, responsesEventFunctionArgsDelta, e.Type, "custom tool must not emit function args deltas")
		assert.NotEqual(t, responsesEventFunctionArgsDone, e.Type, "custom tool must not emit function args done")
	}

	var added, inputDone, itemDone, completed *ChatToResponsesStreamEvent
	for i := range events {
		switch events[i].Type {
		case responsesEventOutputItemAdded:
			added = &events[i]
		case responsesEventCustomToolInputDone:
			inputDone = &events[i]
		case responsesEventOutputItemDone:
			itemDone = &events[i]
		case responsesEventCompleted:
			completed = &events[i]
		}
	}
	require.NotNil(t, added)
	assert.Equal(t, responsesOutputTypeCustomToolCall, added.Payload.Item.Type)
	assert.Equal(t, "apply_patch", added.Payload.Item.Name)
	require.NotNil(t, inputDone)
	assert.Equal(t, "*** Begin Patch", inputDone.Payload.Input)
	require.NotNil(t, itemDone)
	assert.Equal(t, responsesOutputTypeCustomToolCall, itemDone.Payload.Item.Type)
	assert.Equal(t, "*** Begin Patch", itemDone.Payload.Item.Input)
	require.NotNil(t, completed)
	require.NotNil(t, completed.Payload.Response)
	require.Len(t, completed.Payload.Response.Output, 1)
	assert.Equal(t, responsesOutputTypeCustomToolCall, completed.Payload.Response.Output[0].Type)
	assert.Equal(t, "*** Begin Patch", completed.Payload.Response.Output[0].Input)
}

// 流式方向：tool_search 还原为 tool_search_call（execution: client），
// 不发 function_call_arguments.* 增量，arguments 在 output_item.done 一次性携带。
func TestChatCompletionsStreamToResponsesEventsToolSearchCall(t *testing.T) {
	state := NewChatToResponsesStreamState("resp_1", "gpt-test")
	state.ToolSearchEnabled = true
	toolIndex := 0

	var events []ChatToResponsesStreamEvent
	events = append(events, mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, Delta: dto.ChatCompletionsStreamResponseChoiceDelta{ToolCalls: []dto.ToolCallResponse{
				{Index: &toolIndex, ID: "ts_1", Type: "function", Function: dto.FunctionResponse{Name: "tool_search"}},
			}}},
		},
	})...)
	events = append(events, mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, Delta: dto.ChatCompletionsStreamResponseChoiceDelta{ToolCalls: []dto.ToolCallResponse{
				{Index: &toolIndex, Function: dto.FunctionResponse{Arguments: `{"query":"github","limit":3}`}},
			}}},
		},
	})...)
	finishReason := "tool_calls"
	events = append(events, mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, FinishReason: &finishReason},
		},
	})...)
	events = append(events, FinalizeChatCompletionsStreamToResponses(state)...)

	for _, e := range events {
		assert.NotEqual(t, responsesEventFunctionArgsDelta, e.Type, "tool_search must not emit function args deltas")
		assert.NotEqual(t, responsesEventFunctionArgsDone, e.Type, "tool_search must not emit function args done")
	}

	var added, itemDone, completed *ChatToResponsesStreamEvent
	for i := range events {
		switch events[i].Type {
		case responsesEventOutputItemAdded:
			added = &events[i]
		case responsesEventOutputItemDone:
			itemDone = &events[i]
		case responsesEventCompleted:
			completed = &events[i]
		}
	}
	require.NotNil(t, added)
	assert.Equal(t, responsesOutputTypeToolSearchCall, added.Payload.Item.Type)
	assert.Equal(t, "client", added.Payload.Item.Execution)
	require.NotNil(t, itemDone)
	assert.Equal(t, responsesOutputTypeToolSearchCall, itemDone.Payload.Item.Type)
	assert.Equal(t, "client", itemDone.Payload.Item.Execution)
	assert.JSONEq(t, `{"query":"github","limit":3}`, string(itemDone.Payload.Item.Arguments))
	require.NotNil(t, completed)
	require.Len(t, completed.Payload.Response.Output, 1)
	assert.Equal(t, responsesOutputTypeToolSearchCall, completed.Payload.Response.Output[0].Type)
}

// 上游给出 tool_calls 但所有调用名为空：finalize 转 response.failed。
func TestChatCompletionsStreamToResponsesDroppedToolCallsFail(t *testing.T) {
	state := NewChatToResponsesStreamState("resp_1", "gpt-test")
	toolIndex := 0

	mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, Delta: dto.ChatCompletionsStreamResponseChoiceDelta{ToolCalls: []dto.ToolCallResponse{
				{Index: &toolIndex, ID: "call_1", Type: "function", Function: dto.FunctionResponse{Name: ""}},
			}}},
		},
	})
	finishReason := "tool_calls"
	mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, FinishReason: &finishReason},
		},
	})
	events := FinalizeChatCompletionsStreamToResponses(state)

	require.Len(t, events, 1)
	assert.Equal(t, responsesEventFailed, events[0].Type)
	require.NotNil(t, events[0].Payload.Response)
	assert.JSONEq(t, `"failed"`, string(events[0].Payload.Response.Status))
	errMap, ok := events[0].Payload.Response.Error.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "upstream_tool_call_dropped", errMap["code"])
}

// 工具调用历史记录：completed 响应的 function_call 进入 codexhistory 缓存。
func TestChatCompletionsStreamRecordsToolCallHistory(t *testing.T) {
	state := NewChatToResponsesStreamState("resp_hist_1", "gpt-test")
	toolIndex := 0

	mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, Delta: dto.ChatCompletionsStreamResponseChoiceDelta{ToolCalls: []dto.ToolCallResponse{
				{Index: &toolIndex, ID: "call_h1", Type: "function", Function: dto.FunctionResponse{Name: "exec", Arguments: `{"cmd":"ls"}`}},
			}}},
		},
	})
	finishReason := "tool_calls"
	mustResponsesEventsFromChatChunk(t, state, &dto.ChatCompletionsStreamResponse{
		Choices: []dto.ChatCompletionsStreamResponseChoice{
			{Index: 0, FinishReason: &finishReason},
		},
	})
	FinalizeChatCompletionsStreamToResponses(state)

	got := codexhistory.Lookup("resp_hist_1", map[string]bool{"call_h1": true})
	require.Len(t, got, 1)
	assert.Equal(t, "call_h1", got[0].CallID)
	assert.Equal(t, "function_call", got[0].Item["type"])
	assert.Equal(t, "exec", got[0].Item["name"])
}
