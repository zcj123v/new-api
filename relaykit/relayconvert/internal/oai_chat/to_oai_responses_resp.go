package oaichat

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/relayconvert/codexhistory"
	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
)

const (
	chatFinishReasonLength        = "length"
	chatFinishReasonContentFilter = "content_filter"

	responsesEventCreated                  = "response.created"
	responsesEventCompleted                = "response.completed"
	responsesEventIncomplete               = "response.incomplete"
	responsesEventFailed                   = "response.failed"
	responsesEventOutputTextDelta          = "response.output_text.delta"
	responsesEventOutputItemAdded          = "response.output_item.added"
	responsesEventOutputItemDone           = "response.output_item.done"
	responsesEventFunctionArgsDelta        = "response.function_call_arguments.delta"
	responsesEventFunctionArgsDone         = "response.function_call_arguments.done"
	responsesEventCustomToolInputDone      = "response.custom_tool_call_input.done"
	responsesEventReasoningSummaryDelta    = "response.reasoning_summary_text.delta"
	responsesEventReasoningSummaryDone     = "response.reasoning_summary_text.done"
	responsesOutputTypeFunctionCall        = "function_call"
	responsesOutputTypeCustomToolCall      = "custom_tool_call"
	responsesOutputTypeToolSearchCall      = "tool_search_call"
	responsesOutputTypeMessage             = "message"
	responsesOutputTypeReasoning           = "reasoning"
	responsesIncompleteReasonContentFilter = "content_filter"
	responsesIncompleteReasonMaxTokens     = "max_output_tokens"
)

func ChatCompletionsResponseToResponsesResponse(resp *dto.OpenAITextResponse, id string) (*dto.OpenAIResponsesResponse, *dto.Usage, error) {
	return ChatCompletionsResponseToResponsesResponseWithCustomTools(resp, id, nil, false)
}

// ChatCompletionsResponseToResponsesResponseWithCustomTools converts a chat
// response to Responses format, mapping function calls whose names are in
// customTools back to custom_tool_call items (unwrapping {"input": "..."}).
// toolSearchEnabled additionally maps calls to the synthesized tool_search
// function back to tool_search_call items (execution: client).
func ChatCompletionsResponseToResponsesResponseWithCustomTools(resp *dto.OpenAITextResponse, id string, customTools map[string]bool, toolSearchEnabled bool) (*dto.OpenAIResponsesResponse, *dto.Usage, error) {
	if resp == nil {
		return nil, nil, errors.New("response is nil")
	}

	usage := UsageFromChatUsage(&resp.Usage)
	out := &dto.OpenAIResponsesResponse{
		ID:        id,
		Object:    "response",
		CreatedAt: chatCreatedAt(resp.Created),
		Status:    []byte(`"completed"`),
		Model:     resp.Model,
		Output:    make([]dto.ResponsesOutput, 0),
		Usage:     usage,
	}

	if len(resp.Choices) == 0 {
		return out, usage, nil
	}

	choice := resp.Choices[0]
	if status, details := ResponsesStatusFromChatFinishReason(choice.FinishReason); status != "" {
		out.Status = []byte(fmt.Sprintf("%q", status))
		out.IncompleteDetails = details
	}

	if text := choice.Message.StringContent(); text != "" {
		out.Output = append(out.Output, dto.ResponsesOutput{
			Type:   responsesOutputTypeMessage,
			ID:     fmt.Sprintf("%s_msg_0", id),
			Status: responseOutputStatus(out),
			Role:   "assistant",
			Content: []dto.ResponsesOutputContent{
				{
					Type:        "output_text",
					Text:        text,
					Annotations: []interface{}{},
				},
			},
		})
	}
	if reasoning := choice.Message.GetReasoningContent(); reasoning != "" {
		out.Output = append(out.Output, dto.ResponsesOutput{
			Type:   responsesOutputTypeReasoning,
			ID:     fmt.Sprintf("%s_reasoning_0", id),
			Status: responseOutputStatus(out),
			Content: []dto.ResponsesOutputContent{
				{
					Type: "summary_text",
					Text: reasoning,
				},
			},
		})
	}

	droppedToolCalls := 0
	for i, toolCall := range choice.Message.ParseToolCalls() {
		// 丢弃无名/纯空白名的 tool call（上游偶发）；若本应 completed 的
		// tool_calls 回合一个可用调用都不剩，下方统一转 failed。
		if strings.TrimSpace(toolCall.Function.Name) == "" {
			droppedToolCalls++
			continue
		}
		toolOutput, err := chatToolCallToResponsesOutput(toolCall, id, i, responseOutputStatus(out), customTools, toolSearchEnabled)
		if err != nil {
			return nil, nil, err
		}
		out.Output = append(out.Output, toolOutput)
	}
	if droppedToolCalls > 0 && choice.FinishReason == "tool_calls" {
		hasToolOutput := false
		for _, output := range out.Output {
			if output.Type == responsesOutputTypeFunctionCall ||
				output.Type == responsesOutputTypeCustomToolCall ||
				output.Type == responsesOutputTypeToolSearchCall {
				hasToolOutput = true
				break
			}
		}
		if !hasToolOutput {
			out.Status = []byte(`"failed"`)
			out.Error = map[string]any{
				"code":    "upstream_tool_call_dropped",
				"message": "upstream returned tool_calls finish reason but every tool call had an empty name",
			}
		}
	}

	// 记录工具调用历史，供 previous_response_id 续轮恢复（参考 CC Switch）。
	recordResponsesToolCallHistory(out)

	return out, usage, nil
}

// recordResponsesToolCallHistory 把 response 的工具调用项按 Responses 原始
// 形状存入历史缓存。
func recordResponsesToolCallHistory(resp *dto.OpenAIResponsesResponse) {
	if resp == nil || resp.ID == "" {
		return
	}
	calls := make([]codexhistory.CachedCall, 0, len(resp.Output))
	for _, output := range resp.Output {
		callID := strings.TrimSpace(output.CallId)
		if callID == "" {
			continue
		}
		switch output.Type {
		case responsesOutputTypeFunctionCall:
			calls = append(calls, codexhistory.CachedCall{CallID: callID, Item: map[string]any{
				"type":      "function_call",
				"call_id":   callID,
				"name":      output.Name,
				"arguments": output.ArgumentsString(),
			}})
		case responsesOutputTypeCustomToolCall:
			calls = append(calls, codexhistory.CachedCall{CallID: callID, Item: map[string]any{
				"type":    "custom_tool_call",
				"call_id": callID,
				"name":    output.Name,
				"input":   output.Input,
			}})
		case responsesOutputTypeToolSearchCall:
			calls = append(calls, codexhistory.CachedCall{CallID: callID, Item: map[string]any{
				"type":      "tool_search_call",
				"call_id":   callID,
				"arguments": output.ArgumentsString(),
			}})
		}
	}
	codexhistory.Record(resp.ID, calls)
}

func ResponsesStatusFromChatFinishReason(finishReason string) (string, *dto.IncompleteDetails) {
	switch strings.TrimSpace(finishReason) {
	case chatFinishReasonLength:
		return "incomplete", &dto.IncompleteDetails{Reason: responsesIncompleteReasonMaxTokens}
	case chatFinishReasonContentFilter:
		return "incomplete", &dto.IncompleteDetails{Reason: responsesIncompleteReasonContentFilter}
	default:
		return "completed", nil
	}
}

func UsageFromChatUsage(src *dto.Usage) *dto.Usage {
	usage := &dto.Usage{}
	if src == nil {
		return usage
	}
	usage.UsageSemantic = src.UsageSemantic
	usage.UsageSource = src.UsageSource
	usage.BillingUsage = dto.CloneBillingUsage(src.BillingUsage)
	if usage.BillingUsage == nil {
		usage.BillingUsage = dto.NewOpenAIChatBillingUsage(src)
	}
	usage.Cost = src.Cost
	if src.PromptTokens != 0 {
		usage.PromptTokens = src.PromptTokens
		usage.InputTokens = src.PromptTokens
	}
	if src.CompletionTokens != 0 {
		usage.CompletionTokens = src.CompletionTokens
		usage.OutputTokens = src.CompletionTokens
	}
	if src.TotalTokens != 0 {
		usage.TotalTokens = src.TotalTokens
	} else {
		usage.TotalTokens = usage.InputTokens + usage.OutputTokens
	}
	if src.PromptTokensDetails.CachedTokens != 0 ||
		src.PromptTokensDetails.ImageTokens != 0 ||
		src.PromptTokensDetails.AudioTokens != 0 ||
		src.PromptTokensDetails.CachedCreationTokens != 0 ||
		src.PromptTokensDetails.CacheWriteTokens != 0 ||
		src.PromptTokensDetails.TextTokens != 0 {
		details := src.PromptTokensDetails
		usage.InputTokensDetails = &details
	}
	if src.CompletionTokenDetails.ReasoningTokens != 0 ||
		src.CompletionTokenDetails.TextTokens != 0 ||
		src.CompletionTokenDetails.AudioTokens != 0 ||
		src.CompletionTokenDetails.ImageTokens != 0 {
		usage.CompletionTokenDetails = src.CompletionTokenDetails
	}

	// Ensure reasoning_tokens is captured from both sources:
	// 1. Top-level reasoning_tokens (e.g. Moonshot/Kimi upstream)
	// 2. completion_tokens_details.reasoning_tokens (standard OpenAI)
	reasoningTokens := src.ReasoningTokens
	if reasoningTokens == 0 {
		reasoningTokens = src.CompletionTokenDetails.ReasoningTokens
	}
	usage.ReasoningTokens = reasoningTokens
	// output_tokens_details is always emitted (reasoning_tokens may be 0) to
	// match upstream OpenAI behavior; strict Responses clients require the field.
	if src.OutputTokensDetails != nil {
		usage.OutputTokensDetails = src.OutputTokensDetails
	} else {
		usage.OutputTokensDetails = &dto.OutputTokenDetails{
			ReasoningTokens: reasoningTokens,
		}
	}

	usage.ClaudeCacheCreation5mTokens = src.ClaudeCacheCreation5mTokens
	usage.ClaudeCacheCreation1hTokens = src.ClaudeCacheCreation1hTokens
	return usage
}

func responseOutputStatus(resp *dto.OpenAIResponsesResponse) string {
	if resp == nil || responseStatusString(resp) != "incomplete" {
		return "completed"
	}
	return "incomplete"
}

func responseStatusString(resp *dto.OpenAIResponsesResponse) string {
	if resp == nil || len(resp.Status) == 0 {
		return ""
	}
	var status string
	_ = kitutil.Unmarshal(resp.Status, &status)
	return strings.TrimSpace(status)
}

func chatToolCallToResponsesOutput(toolCall dto.ToolCallRequest, responseID string, index int, status string, customTools map[string]bool, toolSearchEnabled bool) (dto.ResponsesOutput, error) {
	callID := strings.TrimSpace(toolCall.ID)
	if callID == "" {
		callID = fmt.Sprintf("%s_call_%d", responseID, index)
	}
	if toolCall.Type == "" || toolCall.Type == "function" {
		name := toolCall.Function.Name
		if toolSearchEnabled && name == "tool_search" {
			// 合成的 tool_search function 被调用：还原为 tool_search_call，
			// arguments 以对象形式携带，execution: client 由 Codex 本地执行。
			return dto.ResponsesOutput{
				Type:      responsesOutputTypeToolSearchCall,
				ID:        callID,
				Status:    status,
				CallId:    callID,
				Execution: "client",
				Arguments: chatArgumentsObjectRawMessage(toolCall.Function.Arguments),
			}, nil
		}
		if customTools[name] {
			// 伪装成 function 的 freeform 工具：还原为 custom_tool_call，
			// arguments {"input": "..."} 解包回 input 字符串。
			return dto.ResponsesOutput{
				Type:   "custom_tool_call",
				ID:     callID,
				Status: status,
				CallId: callID,
				Name:   name,
				Input:  kitutil.UnwrapCustomToolInput(toolCall.Function.Arguments),
			}, nil
		}
		return dto.ResponsesOutput{
			Type:      responsesOutputTypeFunctionCall,
			ID:        callID,
			Status:    status,
			CallId:    callID,
			Name:      name,
			Arguments: chatArgumentsRawMessage(toolCall.Function.Arguments),
		}, nil
	}
	return dto.ResponsesOutput{
		Type:      toolCall.Type,
		ID:        callID,
		Status:    status,
		CallId:    callID,
		Arguments: toolCall.Custom,
	}, nil
}

func chatArgumentsRawMessage(arguments string) []byte {
	raw, err := kitutil.Marshal(arguments)
	if err != nil {
		return []byte(`""`)
	}
	return raw
}

// chatArgumentsObjectRawMessage returns arguments as a raw JSON object;
// falls back to the quoted string when it is not valid JSON.
func chatArgumentsObjectRawMessage(arguments string) []byte {
	var obj map[string]any
	if err := kitutil.UnmarshalJsonStr(arguments, &obj); err == nil && obj != nil {
		if raw, err := kitutil.Marshal(obj); err == nil {
			return raw
		}
	}
	return chatArgumentsRawMessage(arguments)
}

func chatCreatedAt(created any) int {
	switch v := created.(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	case float32:
		return int(v)
	case string:
		if parsed := kitutil.String2Int(v); parsed != 0 {
			return parsed
		}
	}
	return int(time.Now().Unix())
}

func responsesStreamEvent(eventType string, payload dto.ResponsesStreamResponse) ChatToResponsesStreamEvent {
	payload.Type = eventType
	return ChatToResponsesStreamEvent{
		Type:    eventType,
		Payload: payload,
	}
}

func intPtr(v int) *int {
	return &v
}
