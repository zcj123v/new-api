package oaichat

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/QuantumNous/new-api/relaykit/dto"
	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
)

type ChatToResponsesStreamEvent struct {
	Type    string
	Payload dto.ResponsesStreamResponse
}

type ChatToResponsesStreamState struct {
	ID      string
	Model   string
	Created int64
	Usage   *dto.Usage

	// CustomTools 记录请求侧被伪装成 function 的 freeform 工具名；
	// 命中名字的 tool call 还原为 custom_tool_call 事件序列。
	CustomTools map[string]bool
	// ToolSearchEnabled 标记请求侧声明了 tool_search（被合成为同名
	// function），其调用还原为 tool_search_call（execution: client）。
	ToolSearchEnabled bool

	status            string
	incompleteDetails *dto.IncompleteDetails
	sentCreated       bool
	textOutputIndex   int
	textStarted       bool
	textDone          bool
	reasoningIndex    int
	reasoningStarted  bool
	reasoningDone     bool
	finalized         bool
	nextOutputIndex   int
	toolsByIndex      map[int]*chatToResponsesStreamTool
	droppedToolCalls  int
	outputOrder       []chatToResponsesOutputRef
	text              strings.Builder
	reasoning         strings.Builder
}

type chatToResponsesStreamTool struct {
	ChatIndex    int
	OutputIndex  int
	ID           string
	Name         string
	IsCustom     bool
	IsToolSearch bool
	Arguments    strings.Builder
	Done         bool
}

type chatToResponsesOutputRef struct {
	Kind      string
	ToolIndex int
}

func NewChatToResponsesStreamState(id string, model string) *ChatToResponsesStreamState {
	return &ChatToResponsesStreamState{
		ID:              id,
		Model:           model,
		Created:         time.Now().Unix(),
		Usage:           &dto.Usage{},
		status:          "completed",
		textOutputIndex: -1,
		reasoningIndex:  -1,
		toolsByIndex:    make(map[int]*chatToResponsesStreamTool),
	}
}

func ChatCompletionsStreamChunkToResponsesEvents(chunk *dto.ChatCompletionsStreamResponse, state *ChatToResponsesStreamState) ([]ChatToResponsesStreamEvent, error) {
	if chunk == nil || state == nil {
		return nil, nil
	}
	if state.ID == "" {
		state.ID = chunk.Id
	}
	if state.Model == "" {
		state.Model = chunk.Model
	}
	if state.Created == 0 {
		state.Created = chunk.Created
	}
	if chunk.Usage != nil {
		state.Usage = UsageFromChatUsage(chunk.Usage)
	}

	events := make([]ChatToResponsesStreamEvent, 0)
	if !state.sentCreated {
		state.sentCreated = true
		events = append(events, responsesStreamEvent(responsesEventCreated, dto.ResponsesStreamResponse{
			Type:     responsesEventCreated,
			Response: state.createdResponse(),
		}))
	}
	for _, choice := range chunk.Choices {
		if choice.Delta.GetReasoningContent() != "" {
			events = append(events, state.appendReasoningDelta(choice.Delta.GetReasoningContent())...)
		}
		if choice.Delta.GetContentString() != "" {
			events = append(events, state.appendTextDelta(choice.Delta.GetContentString())...)
		}
		for _, toolCall := range choice.Delta.ToolCalls {
			toolEvents, err := state.appendToolCallDelta(toolCall)
			if err != nil {
				return nil, err
			}
			events = append(events, toolEvents...)
		}
		if choice.FinishReason != nil && strings.TrimSpace(*choice.FinishReason) != "" {
			state.applyFinishReason(*choice.FinishReason)
			events = append(events, state.doneDeltaEvents()...)
		}
	}
	return events, nil
}

func FinalizeChatCompletionsStreamToResponses(state *ChatToResponsesStreamState) []ChatToResponsesStreamEvent {
	if state == nil || state.finalized {
		return nil
	}
	events := state.doneDeltaEvents()
	state.finalized = true
	resp := state.finalResponse()
	eventType := responsesEventCompleted
	switch {
	case state.status == "incomplete":
		eventType = responsesEventIncomplete
	case state.droppedToolCalls > 0 && len(state.toolsByIndex) == 0:
		// 上游给出 tool_calls 但所有调用名皆为空：与非流式一致转 failed，
		// 防 Codex agent loop 拿到空 output 静默终止。
		eventType = responsesEventFailed
		resp.Status = []byte(`"failed"`)
		resp.Error = map[string]any{
			"code":    "upstream_tool_call_dropped",
			"message": "upstream returned tool calls but every tool call had an empty name",
		}
	}
	events = append(events, responsesStreamEvent(eventType, dto.ResponsesStreamResponse{
		Type:     eventType,
		Response: resp,
	}))
	recordResponsesToolCallHistory(resp)
	return events
}

func (s *ChatToResponsesStreamState) UsageText() string {
	if s == nil {
		return ""
	}
	return s.text.String()
}

func (s *ChatToResponsesStreamState) appendTextDelta(delta string) []ChatToResponsesStreamEvent {
	events := make([]ChatToResponsesStreamEvent, 0, 2)
	if !s.textStarted {
		s.textStarted = true
		s.textOutputIndex = s.nextIndex("message", -1)
		events = append(events, responsesStreamEvent(responsesEventOutputItemAdded, dto.ResponsesStreamResponse{
			Type:        responsesEventOutputItemAdded,
			OutputIndex: intPtr(s.textOutputIndex),
			Item: &dto.ResponsesOutput{
				Type:    responsesOutputTypeMessage,
				ID:      s.messageID(),
				Status:  "in_progress",
				Role:    "assistant",
				Content: []dto.ResponsesOutputContent{},
			},
		}))
	}
	s.text.WriteString(delta)
	events = append(events, responsesStreamEvent(responsesEventOutputTextDelta, dto.ResponsesStreamResponse{
		Type:         responsesEventOutputTextDelta,
		OutputIndex:  intPtr(s.textOutputIndex),
		ContentIndex: intPtr(0),
		Delta:        delta,
		ItemID:       s.messageID(),
	}))
	return events
}

func (s *ChatToResponsesStreamState) appendReasoningDelta(delta string) []ChatToResponsesStreamEvent {
	events := make([]ChatToResponsesStreamEvent, 0, 2)
	if !s.reasoningStarted {
		s.reasoningStarted = true
		s.reasoningIndex = s.nextIndex("reasoning", -1)
		events = append(events, responsesStreamEvent(responsesEventOutputItemAdded, dto.ResponsesStreamResponse{
			Type:        responsesEventOutputItemAdded,
			OutputIndex: intPtr(s.reasoningIndex),
			Item: &dto.ResponsesOutput{
				Type:    responsesOutputTypeReasoning,
				ID:      s.reasoningID(),
				Status:  "in_progress",
				Content: []dto.ResponsesOutputContent{},
			},
		}))
	}
	s.reasoning.WriteString(delta)
	events = append(events, responsesStreamEvent(responsesEventReasoningSummaryDelta, dto.ResponsesStreamResponse{
		Type:         responsesEventReasoningSummaryDelta,
		OutputIndex:  intPtr(s.reasoningIndex),
		SummaryIndex: intPtr(0),
		Delta:        delta,
		ItemID:       s.reasoningID(),
	}))
	return events
}

func (s *ChatToResponsesStreamState) appendToolCallDelta(toolCall dto.ToolCallResponse) ([]ChatToResponsesStreamEvent, error) {
	chatIndex := 0
	if toolCall.Index != nil {
		chatIndex = *toolCall.Index
	}
	tool := s.toolsByIndex[chatIndex]
	events := make([]ChatToResponsesStreamEvent, 0, 2)
	if tool == nil {
		toolName := strings.TrimSpace(toolCall.Function.Name)
		if toolName == "" {
			// 丢弃无名 tool call 并计数；若回合最终一个可用调用都不剩，
			// finalize 时统一转 failed（防 Codex agent loop 静默终止）。
			s.droppedToolCalls++
			return events, nil
		}
		tool = &chatToResponsesStreamTool{
			ChatIndex:   chatIndex,
			OutputIndex: s.nextIndex("tool", chatIndex),
			ID:          strings.TrimSpace(toolCall.ID),
			Name:        toolName,
		}
		if tool.ID == "" {
			tool.ID = fmt.Sprintf("%s_call_%d", s.ID, chatIndex)
		}
		s.toolsByIndex[chatIndex] = tool
		tool.IsCustom = s.CustomTools[tool.Name]
		tool.IsToolSearch = s.ToolSearchEnabled && tool.Name == "tool_search"
		switch {
		case tool.IsToolSearch:
			// tool_search：arguments 流暂存，完成时一次性发出。
			events = append(events, responsesStreamEvent(responsesEventOutputItemAdded, dto.ResponsesStreamResponse{
				Type:        responsesEventOutputItemAdded,
				OutputIndex: intPtr(tool.OutputIndex),
				ItemID:      tool.ID,
				Item: &dto.ResponsesOutput{
					Type:      responsesOutputTypeToolSearchCall,
					ID:        tool.ID,
					Status:    "in_progress",
					CallId:    tool.ID,
					Execution: "client",
				},
			}))
		case tool.IsCustom:
			// freeform 工具：item 类型为 custom_tool_call；arguments 流暂存，
			// 完成时解包 {"input": "..."} 一次性发出，不做半成品 JSON 解包。
			events = append(events, responsesStreamEvent(responsesEventOutputItemAdded, dto.ResponsesStreamResponse{
				Type:        responsesEventOutputItemAdded,
				OutputIndex: intPtr(tool.OutputIndex),
				ItemID:      tool.ID,
				Item: &dto.ResponsesOutput{
					Type:   responsesOutputTypeCustomToolCall,
					ID:     tool.ID,
					Status: "in_progress",
					CallId: tool.ID,
					Name:   tool.Name,
					Input:  "",
				},
			}))
		default:
			events = append(events, responsesStreamEvent(responsesEventOutputItemAdded, dto.ResponsesStreamResponse{
				Type:        responsesEventOutputItemAdded,
				OutputIndex: intPtr(tool.OutputIndex),
				ItemID:      tool.ID,
				Item: &dto.ResponsesOutput{
					Type:      responsesOutputTypeFunctionCall,
					ID:        tool.ID,
					Status:    "in_progress",
					CallId:    tool.ID,
					Name:      tool.Name,
					Arguments: []byte(`""`),
				},
			}))
		}
	}
	if strings.TrimSpace(toolCall.ID) != "" {
		tool.ID = strings.TrimSpace(toolCall.ID)
	}
	if strings.TrimSpace(toolCall.Function.Name) != "" {
		tool.Name = strings.TrimSpace(toolCall.Function.Name)
		tool.IsCustom = s.CustomTools[tool.Name]
		tool.IsToolSearch = s.ToolSearchEnabled && tool.Name == "tool_search"
	}
	if toolCall.Function.Arguments != "" {
		tool.Arguments.WriteString(toolCall.Function.Arguments)
		if !tool.IsCustom && !tool.IsToolSearch {
			events = append(events, responsesStreamEvent(responsesEventFunctionArgsDelta, dto.ResponsesStreamResponse{
				Type:        responsesEventFunctionArgsDelta,
				OutputIndex: intPtr(tool.OutputIndex),
				ItemID:      tool.ID,
				Delta:       toolCall.Function.Arguments,
			}))
		}
	}
	return events, nil
}

func (s *ChatToResponsesStreamState) doneDeltaEvents() []ChatToResponsesStreamEvent {
	events := make([]ChatToResponsesStreamEvent, 0)
	status := s.outputStatus()
	if s.textStarted && !s.textDone {
		s.textDone = true
		events = append(events, responsesStreamEvent("response.output_text.done", dto.ResponsesStreamResponse{
			Type:         "response.output_text.done",
			OutputIndex:  intPtr(s.textOutputIndex),
			ContentIndex: intPtr(0),
			ItemID:       s.messageID(),
		}))
		events = append(events, responsesStreamEvent(responsesEventOutputItemDone, dto.ResponsesStreamResponse{
			Type:        responsesEventOutputItemDone,
			OutputIndex: intPtr(s.textOutputIndex),
			Item:        s.messageOutput(status),
		}))
	}
	if s.reasoningStarted && !s.reasoningDone {
		s.reasoningDone = true
		events = append(events, responsesStreamEvent(responsesEventReasoningSummaryDone, dto.ResponsesStreamResponse{
			Type:         responsesEventReasoningSummaryDone,
			OutputIndex:  intPtr(s.reasoningIndex),
			SummaryIndex: intPtr(0),
			ItemID:       s.reasoningID(),
			Part: &dto.ResponsesReasoningSummaryPart{
				Type: "summary_text",
				Text: s.reasoning.String(),
			},
		}))
		events = append(events, responsesStreamEvent(responsesEventOutputItemDone, dto.ResponsesStreamResponse{
			Type:        responsesEventOutputItemDone,
			OutputIndex: intPtr(s.reasoningIndex),
			Item:        s.reasoningOutput(status),
		}))
	}
	for _, tool := range s.sortedTools() {
		if tool.Done {
			continue
		}
		tool.Done = true
		if tool.IsToolSearch {
			// tool_search_call 无参数增量事件，output_item.done 携带完整 arguments。
		} else if tool.IsCustom {
			events = append(events, responsesStreamEvent(responsesEventCustomToolInputDone, dto.ResponsesStreamResponse{
				Type:        responsesEventCustomToolInputDone,
				OutputIndex: intPtr(tool.OutputIndex),
				ItemID:      tool.ID,
				Input:       kitutil.UnwrapCustomToolInput(tool.Arguments.String()),
			}))
		} else {
			events = append(events, responsesStreamEvent(responsesEventFunctionArgsDone, dto.ResponsesStreamResponse{
				Type:        responsesEventFunctionArgsDone,
				OutputIndex: intPtr(tool.OutputIndex),
				ItemID:      tool.ID,
			}))
		}
		events = append(events, responsesStreamEvent(responsesEventOutputItemDone, dto.ResponsesStreamResponse{
			Type:        responsesEventOutputItemDone,
			OutputIndex: intPtr(tool.OutputIndex),
			Item:        s.toolOutput(tool, status),
		}))
	}
	return events
}

func (s *ChatToResponsesStreamState) applyFinishReason(finishReason string) {
	if status, details := ResponsesStatusFromChatFinishReason(finishReason); status != "" {
		s.status = status
		s.incompleteDetails = details
	}
}

func (s *ChatToResponsesStreamState) finalResponse() *dto.OpenAIResponsesResponse {
	output := make([]dto.ResponsesOutput, 0, len(s.outputOrder))
	status := s.outputStatus()
	for _, ref := range s.outputOrder {
		switch ref.Kind {
		case "message":
			output = append(output, *s.messageOutput(status))
		case "reasoning":
			output = append(output, *s.reasoningOutput(status))
		case "tool":
			if tool := s.toolsByIndex[ref.ToolIndex]; tool != nil {
				output = append(output, *s.toolOutput(tool, status))
			}
		}
	}
	return &dto.OpenAIResponsesResponse{
		ID:                s.ID,
		Object:            "response",
		CreatedAt:         int(s.Created),
		Status:            []byte(fmt.Sprintf("%q", s.status)),
		IncompleteDetails: s.incompleteDetails,
		Model:             s.Model,
		Output:            output,
		Usage:             s.Usage,
	}
}

func (s *ChatToResponsesStreamState) createdResponse() *dto.OpenAIResponsesResponse {
	return &dto.OpenAIResponsesResponse{
		ID:        s.ID,
		Object:    "response",
		CreatedAt: int(s.Created),
		Status:    []byte(`"in_progress"`),
		Model:     s.Model,
		Output:    []dto.ResponsesOutput{},
	}
}

func (s *ChatToResponsesStreamState) nextIndex(kind string, toolIndex int) int {
	index := s.nextOutputIndex
	s.nextOutputIndex++
	s.outputOrder = append(s.outputOrder, chatToResponsesOutputRef{Kind: kind, ToolIndex: toolIndex})
	return index
}

func (s *ChatToResponsesStreamState) sortedTools() []*chatToResponsesStreamTool {
	indexes := make([]int, 0, len(s.toolsByIndex))
	for index := range s.toolsByIndex {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	tools := make([]*chatToResponsesStreamTool, 0, len(indexes))
	for _, index := range indexes {
		tools = append(tools, s.toolsByIndex[index])
	}
	return tools
}

func (s *ChatToResponsesStreamState) outputStatus() string {
	if s.status == "incomplete" {
		return "incomplete"
	}
	return "completed"
}

func (s *ChatToResponsesStreamState) messageID() string {
	return fmt.Sprintf("%s_msg_0", s.ID)
}

func (s *ChatToResponsesStreamState) reasoningID() string {
	return fmt.Sprintf("%s_reasoning_0", s.ID)
}

func (s *ChatToResponsesStreamState) messageOutput(status string) *dto.ResponsesOutput {
	return &dto.ResponsesOutput{
		Type:   responsesOutputTypeMessage,
		ID:     s.messageID(),
		Status: status,
		Role:   "assistant",
		Content: []dto.ResponsesOutputContent{
			{
				Type:        "output_text",
				Text:        s.text.String(),
				Annotations: []interface{}{},
			},
		},
	}
}

func (s *ChatToResponsesStreamState) reasoningOutput(status string) *dto.ResponsesOutput {
	return &dto.ResponsesOutput{
		Type:   responsesOutputTypeReasoning,
		ID:     s.reasoningID(),
		Status: status,
		Content: []dto.ResponsesOutputContent{
			{
				Type: "summary_text",
				Text: s.reasoning.String(),
			},
		},
	}
}

func (s *ChatToResponsesStreamState) toolOutput(tool *chatToResponsesStreamTool, status string) *dto.ResponsesOutput {
	if tool.IsToolSearch {
		return &dto.ResponsesOutput{
			Type:      responsesOutputTypeToolSearchCall,
			ID:        tool.ID,
			Status:    status,
			CallId:    tool.ID,
			Execution: "client",
			Arguments: chatArgumentsObjectRawMessage(tool.Arguments.String()),
		}
	}
	if tool.IsCustom {
		return &dto.ResponsesOutput{
			Type:   responsesOutputTypeCustomToolCall,
			ID:     tool.ID,
			Status: status,
			CallId: tool.ID,
			Name:   tool.Name,
			Input:  kitutil.UnwrapCustomToolInput(tool.Arguments.String()),
		}
	}
	return &dto.ResponsesOutput{
		Type:      responsesOutputTypeFunctionCall,
		ID:        tool.ID,
		Status:    status,
		CallId:    tool.ID,
		Name:      tool.Name,
		Arguments: chatArgumentsRawMessage(tool.Arguments.String()),
	}
}
