package oairesponses

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/relayconvert/codexhistory"
	kitutil "github.com/QuantumNous/new-api/relaykit/relayconvert/kitutil"
)

const (
	responsesInputTypeFunctionCall       = "function_call"
	responsesInputTypeFunctionCallOutput = "function_call_output"
	responsesInputTypeCustomToolCall     = "custom_tool_call"
	responsesInputTypeCustomToolOutput   = "custom_tool_call_output"
	responsesInputTypeToolSearchCall     = "tool_search_call"
	responsesInputTypeToolSearchOutput   = "tool_search_output"
)

// toolSearchProxyName 是 tool_search 在 chat 侧合成的 function 名
//（对齐 CC Switch 的 TOOL_SEARCH_PROXY_NAME）。
const toolSearchProxyName = "tool_search"

const (
	ResponsesInputTypeFunctionCall       = responsesInputTypeFunctionCall
	ResponsesInputTypeFunctionCallOutput = responsesInputTypeFunctionCallOutput
	ResponsesInputTypeCustomToolCall     = responsesInputTypeCustomToolCall
	ResponsesInputTypeCustomToolOutput   = responsesInputTypeCustomToolOutput
)

func ResponsesRequestToChatCompletionsRequest(req *dto.OpenAIResponsesRequest) (*dto.GeneralOpenAIRequest, error) {
	if req == nil {
		return nil, errors.New("request is nil")
	}
	if req.Model == "" {
		return nil, errors.New("model is required")
	}
	if err := validateResponsesRequestChatUnsupportedFields(req); err != nil {
		return nil, err
	}

	// previous_response_id 续轮：Codex 可能只带新的 *_output 项而省略对应的
	// call 项，先用历史缓存补回（参考 CC Switch 的历史恢复）。
	workReq := req
	if strings.TrimSpace(req.PreviousResponseID) != "" {
		if restored := restoreCallsFromHistory(strings.TrimSpace(req.PreviousResponseID), req.Input); restored != nil {
			copied := *req
			copied.Input = restored
			workReq = &copied
		}
	}

	messages, err := responsesRequestMessagesToChat(workReq)
	if err != nil {
		return nil, err
	}

	tools, err := responsesRequestToolsToChat(workReq.Tools)
	if err != nil {
		return nil, err
	}
	// Codex 会把 exec/collaboration 等工具放在 input 的 additional_tools 项里，
	// tool_search_output 项（客户端工具搜索的结果）里也有工具定义，
	// 而不是顶层 tools。提取出来一并转换，消息流里跳过这些项。
	if extraRaw := responsesInputEmbeddedToolsRaw(workReq.Input); len(extraRaw) > 0 {
		extraTools, err := responsesRequestToolsToChat(extraRaw)
		if err != nil {
			return nil, err
		}
		tools = append(tools, extraTools...)
	}

	toolChoice, err := responsesRequestToolChoiceToChat(req.ToolChoice)
	if err != nil {
		return nil, err
	}

	responseFormat, err := responsesRequestTextToChatResponseFormat(req.Text)
	if err != nil {
		return nil, err
	}

	out := &dto.GeneralOpenAIRequest{
		Model:                req.Model,
		Messages:             messages,
		Stream:               req.Stream,
		StreamOptions:        req.StreamOptions,
		MaxCompletionTokens:  req.MaxOutputTokens,
		Temperature:          req.Temperature,
		TopP:                 req.TopP,
		TopLogProbs:          req.TopLogProbs,
		ResponseFormat:       responseFormat,
		Tools:                tools,
		ToolChoice:           toolChoice,
		User:                 req.User,
		Store:                req.Store,
		Metadata:             req.Metadata,
		SafetyIdentifier:     req.SafetyIdentifier,
		PromptCacheRetention: req.PromptCacheRetention,
		EnableThinking:       req.EnableThinking,
		ThinkingBudget:       req.ThinkingBudget,
	}

	out.FrequencyPenalty, err = responsesRawFloat(req.FrequencyPenalty)
	if err != nil {
		return nil, fmt.Errorf("invalid frequency_penalty: %w", err)
	}
	out.PresencePenalty, err = responsesRawFloat(req.PresencePenalty)
	if err != nil {
		return nil, fmt.Errorf("invalid presence_penalty: %w", err)
	}

	if req.Reasoning != nil {
		out.ReasoningEffort = req.Reasoning.Effort
	}
	if req.ServiceTier != "" {
		out.ServiceTier, _ = kitutil.Marshal(req.ServiceTier)
	}
	if len(req.ParallelToolCalls) > 0 && kitutil.GetJsonType(req.ParallelToolCalls) == "boolean" {
		var parallelToolCalls bool
		if err := kitutil.Unmarshal(req.ParallelToolCalls, &parallelToolCalls); err == nil {
			out.ParallelTooCalls = &parallelToolCalls
		}
	}
	if len(req.PromptCacheKey) > 0 && kitutil.GetJsonType(req.PromptCacheKey) == "string" {
		var promptCacheKey string
		if err := kitutil.Unmarshal(req.PromptCacheKey, &promptCacheKey); err == nil {
			out.PromptCacheKey = promptCacheKey
		}
	}

	// kimi/Moonshot、DeepSeek 等 thinking 模型要求带 tool_calls 的 assistant
	// 消息必须携带非空 reasoning_content，缺失会 400。管线末端统一补占位
	// （参考 CC Switch ensure_tool_call_reasoning_content）。
	backfillToolCallReasoningPlaceholders(out.Messages)

	// 严格上游拒绝 tools 为空但带 tool_choice/parallel_tool_calls 的请求。
	if len(out.Tools) == 0 {
		out.ToolChoice = nil
		out.ParallelTooCalls = nil
	}
	// OpenAI 兼容上游流式默认不吐 usage chunk，必须显式 include_usage，
	// 否则 kimi/MiniMax 等流式请求 token 全部漏记。
	if out.Stream != nil && *out.Stream && out.StreamOptions == nil {
		out.StreamOptions = &dto.StreamOptions{IncludeUsage: true}
	}

	return out, nil
}

// backfillToolCallReasoningPlaceholders 给仍缺 reasoning_content 的
// assistant tool-call 消息补占位字符串。
func backfillToolCallReasoningPlaceholders(messages []dto.Message) {
	for i := range messages {
		msg := &messages[i]
		if msg.Role != "assistant" {
			continue
		}
		if len(msg.ParseToolCalls()) == 0 {
			continue
		}
		if msg.ReasoningContent != nil && strings.TrimSpace(*msg.ReasoningContent) != "" {
			continue
		}
		placeholder := "tool call"
		msg.ReasoningContent = &placeholder
	}
}

func validateResponsesRequestChatUnsupportedFields(req *dto.OpenAIResponsesRequest) error {
	unsupported := make([]string, 0, 4)
	if rawJSONPresent(req.Conversation) {
		unsupported = append(unsupported, "conversation")
	}
	if rawJSONPresent(req.Prompt) {
		unsupported = append(unsupported, "prompt")
	}
	if rawJSONPresent(req.ContextManagement) {
		unsupported = append(unsupported, "context_management")
	}
	if len(unsupported) > 0 {
		return fmt.Errorf("responses to chat conversion does not support stateful fields: %s", strings.Join(unsupported, ", "))
	}
	return nil
}

func ValidateRequestChatUnsupportedFields(req *dto.OpenAIResponsesRequest) error {
	return validateResponsesRequestChatUnsupportedFields(req)
}

func responsesRequestMessagesToChat(req *dto.OpenAIResponsesRequest) ([]dto.Message, error) {
	messages := make([]dto.Message, 0)
	if rawJSONPresent(req.Instructions) {
		instructions, err := responsesJSONString(req.Instructions)
		if err != nil {
			return nil, fmt.Errorf("invalid instructions: %w", err)
		}
		if strings.TrimSpace(instructions) != "" {
			messages = append(messages, dto.Message{Role: "system", Content: instructions})
		}
	}

	if !rawJSONPresent(req.Input) {
		return messages, nil
	}

	switch kitutil.GetJsonType(req.Input) {
	case "string":
		input, err := responsesJSONString(req.Input)
		if err != nil {
			return nil, fmt.Errorf("invalid input string: %w", err)
		}
		messages = append(messages, dto.Message{Role: "user", Content: input})
		return messages, nil
	case "array":
		var items []map[string]any
		if err := kitutil.Unmarshal(req.Input, &items); err != nil {
			return nil, fmt.Errorf("invalid input array: %w", err)
		}
		for _, item := range items {
			nextMessages, err := responsesInputItemToChatMessages(item, messages)
			if err != nil {
				return nil, err
			}
			messages = nextMessages
		}
		return messages, nil
	default:
		return nil, fmt.Errorf("unsupported responses input type %q", kitutil.GetJsonType(req.Input))
	}
}

func responsesInputItemToChatMessages(item map[string]any, messages []dto.Message) ([]dto.Message, error) {
	itemType := strings.TrimSpace(kitutil.Interface2String(item["type"]))
	switch itemType {
	case responsesInputTypeFunctionCall:
		toolCall, err := responsesFunctionCallItemToChatToolCall(item)
		if err != nil {
			return nil, err
		}
		return appendToolCallToLastAssistant(messages, toolCall), nil
	case responsesInputTypeCustomToolCall:
		toolCall, err := responsesCustomToolCallItemToChatToolCall(item)
		if err != nil {
			return nil, err
		}
		return appendToolCallToLastAssistant(messages, toolCall), nil
	case responsesInputTypeToolSearchCall:
		toolCall, err := responsesToolSearchCallItemToChatToolCall(item)
		if err != nil {
			return nil, err
		}
		return appendToolCallToLastAssistant(messages, toolCall), nil
	case responsesInputTypeFunctionCallOutput, responsesInputTypeCustomToolOutput, responsesInputTypeToolSearchOutput:
		callID := strings.TrimSpace(kitutil.Interface2String(item["call_id"]))
		content := responseToolOutputToChatContent(item["output"])
		return append(messages, dto.Message{Role: "tool", ToolCallId: callID, Content: content}), nil
	}

	role := strings.TrimSpace(kitutil.Interface2String(item["role"]))
	if itemType == "additional_tools" {
		// 工具清单/工具搜索结果项，不是消息：其中的 tools 已在
		// ResponsesRequestToChatCompletionsRequest 里单独提取转换。
		return messages, nil
	}
	if itemType == "reasoning" {
		// reasoning 项对 chat 上游无意义（encrypted_content 无法解密），
		// 直接跳过，避免退化成空 user 消息。
		return messages, nil
	}
	if role == "" {
		role = "user"
	}
	if role == "developer" {
		// chat 上游无 developer 角色（Kimi 等会 400 role not allowed），
		// 语义上等同系统指令，映射为 system。
		role = "system"
	}
	content, err := responsesInputContentToChatContent(item["content"])
	if err != nil {
		return nil, err
	}
	return append(messages, dto.Message{Role: role, Content: content}), nil
}

func responsesInputContentToChatContent(content any) (any, error) {
	if content == nil {
		return "", nil
	}

	switch value := content.(type) {
	case string:
		return value, nil
	case []any:
		return responsesContentPartsToChatContent(value)
	case []map[string]any:
		parts := make([]any, 0, len(value))
		for _, part := range value {
			parts = append(parts, part)
		}
		return responsesContentPartsToChatContent(parts)
	default:
		return content, nil
	}
}

func responsesContentPartsToChatContent(parts []any) (any, error) {
	chatParts := make([]any, 0, len(parts))
	var textOnly strings.Builder
	onlyText := true

	for _, rawPart := range parts {
		part, ok := rawPart.(map[string]any)
		if !ok {
			onlyText = false
			chatParts = append(chatParts, rawPart)
			continue
		}

		partType := strings.TrimSpace(kitutil.Interface2String(part["type"]))
		switch partType {
		case "input_text", "output_text", "text":
			text := kitutil.Interface2String(part["text"])
			textOnly.WriteString(text)
			chatParts = append(chatParts, map[string]any{
				"type": dto.ContentTypeText,
				"text": text,
			})
		case "input_image":
			onlyText = false
			chatParts = append(chatParts, map[string]any{
				"type":      dto.ContentTypeImageURL,
				"image_url": responsesImagePartToChatImageURL(part),
			})
		case "input_file":
			onlyText = false
			chatParts = append(chatParts, map[string]any{
				"type": dto.ContentTypeFile,
				"file": responsesFilePartToChatFile(part),
			})
		case "input_audio":
			onlyText = false
			chatParts = append(chatParts, map[string]any{
				"type":        dto.ContentTypeInputAudio,
				"input_audio": responsesPartPayload(part, "input_audio"),
			})
		case "input_video":
			onlyText = false
			chatParts = append(chatParts, map[string]any{
				"type":      dto.ContentTypeVideoUrl,
				"video_url": responsesVideoPartToChatVideoURL(part),
			})
		default:
			onlyText = false
			chatParts = append(chatParts, part)
		}
	}

	if onlyText {
		return textOnly.String(), nil
	}
	return chatParts, nil
}

func responsesFunctionCallItemToChatToolCall(item map[string]any) (dto.ToolCallRequest, error) {
	name := strings.TrimSpace(kitutil.Interface2String(item["name"]))
	if name == "" {
		return dto.ToolCallRequest{}, errors.New("function_call item is missing name")
	}
	return dto.ToolCallRequest{
		ID:   responsesCallID(item),
		Type: "function",
		Function: dto.FunctionRequest{
			Name:      name,
			Arguments: responsesArgumentsString(item["arguments"]),
		},
	}, nil
}

func responsesToolSearchCallItemToChatToolCall(item map[string]any) (dto.ToolCallRequest, error) {
	// tool_search_call → tool_search function 调用；arguments 规范化为 JSON 字符串。
	arguments := "{}"
	switch v := item["arguments"].(type) {
	case string:
		if strings.TrimSpace(v) != "" {
			arguments = v
		}
	case nil:
	default:
		if raw, err := kitutil.Marshal(v); err == nil {
			arguments = string(raw)
		}
	}
	return dto.ToolCallRequest{
		ID:   responsesCallID(item),
		Type: "function",
		Function: dto.FunctionRequest{
			Name:      toolSearchProxyName,
			Arguments: arguments,
		},
	}, nil
}

func responsesCustomToolCallItemToChatToolCall(item map[string]any) (dto.ToolCallRequest, error) {	name := strings.TrimSpace(kitutil.Interface2String(item["name"]))
	if name == "" {
		return dto.ToolCallRequest{}, errors.New("custom_tool_call item is missing name")
	}
	// 伪装成普通 function：freeform input 包成 {"input": "..."}，与
	// responsesRequestToolsToChat 的 custom 工具声明伪装保持一致。
	return dto.ToolCallRequest{
		ID:   responsesCallID(item),
		Type: "function",
		Function: dto.FunctionRequest{
			Name:      name,
			Arguments: kitutil.WrapCustomToolInput(responsesArgumentsString(item["input"])),
		},
	}, nil
}

func appendToolCallToLastAssistant(messages []dto.Message, toolCall dto.ToolCallRequest) []dto.Message {
	if len(messages) == 0 || messages[len(messages)-1].Role != "assistant" {
		messages = append(messages, dto.Message{Role: "assistant"})
	}

	idx := len(messages) - 1
	toolCalls := messages[idx].ParseToolCalls()
	toolCalls = append(toolCalls, toolCall)
	toolCallsRaw, _ := kitutil.Marshal(toolCalls)
	messages[idx].ToolCalls = toolCallsRaw
	return messages
}

func responsesRequestToolsToChat(raw json.RawMessage) ([]dto.ToolCallRequest, error) {
	if !rawJSONPresent(raw) {
		return nil, nil
	}

	var tools []map[string]any
	if err := kitutil.Unmarshal(raw, &tools); err != nil {
		return nil, fmt.Errorf("invalid tools: %w", err)
	}

	out := make([]dto.ToolCallRequest, 0, len(tools))
	for _, tool := range tools {
		toolType := strings.TrimSpace(kitutil.Interface2String(tool["type"]))
		if toolType == "function" {
			out = append(out, dto.ToolCallRequest{
				Type: "function",
				Function: dto.FunctionRequest{
					Name:        strings.TrimSpace(kitutil.Interface2String(tool["name"])),
					Description: kitutil.Interface2String(tool["description"]),
					Parameters:  tool["parameters"],
				},
			})
			continue
		}

		if toolType == "tool_search" {
			// tool_search 由 Codex 客户端本地执行，但上游模型需要"看到"它
			// 才会发起调用。合成同名 function（对齐 CC Switch），响应方向再
			// 还原为 tool_search_call（execution: client）。
			out = append(out, dto.ToolCallRequest{
				Type: "function",
				Function: dto.FunctionRequest{
					Name:        toolSearchProxyName,
					Description: "Search and load Codex tools, plugins, connectors, and MCP namespaces for the current task.",
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"query": map[string]any{
								"type":        "string",
								"description": "Search query for tools or connectors to load.",
							},
							"limit": map[string]any{
								"type":        "integer",
								"description": "Maximum number of tool groups to return.",
							},
						},
						"required": []string{"query"},
					},
				},
			})
			continue
		}

		if toolType == dto.CustomType {
			// freeform 工具（如 Codex 的 apply_patch）：chat 上游只认 function，
			// 伪装成 {"input": "..."} 单参数函数；响应方向按工具名还原。
			name := strings.TrimSpace(kitutil.Interface2String(tool["name"]))
			if name == "" {
				continue
			}
			description := kitutil.Interface2String(tool["description"])
			if format, ok := tool["format"].(map[string]any); ok {
				if formatDesc := strings.TrimSpace(kitutil.Interface2String(format["description"])); formatDesc != "" {
					if description != "" {
						description += "\n\n"
					}
					description += "Input format: " + formatDesc
				}
			}
			// 把原始工具定义嵌入描述，保住 freeform 工具的 format/grammar
			// 约束，减少 apply_patch 类工具的格式漂移（参考 CC Switch）。
			if rawDef, err := kitutil.Marshal(tool); err == nil {
				description += "\n\nOriginal tool definition:\n```json\n" + string(rawDef) + "\n```"
			}
			out = append(out, dto.ToolCallRequest{
				Type: "function",
				Function: dto.FunctionRequest{
					Name:        name,
					Description: description,
					Parameters: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"input": map[string]any{
								"type":        "string",
								"description": "The complete freeform input for this tool.",
							},
						},
						"required": []string{"input"},
					},
				},
			})
			continue
		}

		if toolType == "namespace" {
			// namespace 信封（Codex collaboration/MCP 等）：chat 上游不认识，
			// 展平为内部嵌套的工具定义（内部 custom 递归伪装）。
			nestedRaw, err := kitutil.Marshal(tool["tools"])
			if err != nil {
				return nil, err
			}
			nested, err := responsesRequestToolsToChat(nestedRaw)
			if err != nil {
				return nil, err
			}
			out = append(out, nested...)
			continue
		}

		// 其余 Responses 专属类型（tool_search、web_search、mcp、
		// image_generation、code_interpreter 等）在 chat 上游没有等价物，
		// 原样透传只会被上游拒绝（unknown tool type: ...），直接丢弃。
		// tool_search 由客户端本地执行，丢弃不影响已展平工具的使用。
		kitutil.LogInfo(fmt.Sprintf("responses->chat: dropping unsupported tool type %q (name=%q)",
			toolType, kitutil.Interface2String(tool["name"])))
	}
	return out, nil
}

// CollectResponsesCustomToolNames returns the names of freeform (custom) tools
// declared in a Responses request. The chat-side response converter uses the
// names to map disguised function calls back to custom_tool_call items.
func CollectResponsesCustomToolNames(raw json.RawMessage) []string {
	if !rawJSONPresent(raw) {
		return nil
	}
	var tools []map[string]any
	if err := kitutil.Unmarshal(raw, &tools); err != nil {
		return nil
	}
	return collectCustomToolNamesFromTools(tools)
}

func collectCustomToolNamesFromTools(tools []map[string]any) []string {
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		toolType := strings.TrimSpace(kitutil.Interface2String(tool["type"]))
		if toolType == "namespace" {
			// 递归收集 namespace 信封内的 custom 工具
			var nested []map[string]any
			nestedRaw, err := kitutil.Marshal(tool["tools"])
			if err == nil && rawJSONPresent(nestedRaw) {
				if err := kitutil.Unmarshal(nestedRaw, &nested); err == nil {
					names = append(names, collectCustomToolNamesFromTools(nested)...)
				}
			}
			continue
		}
		if toolType != dto.CustomType {
			continue
		}
		if name := strings.TrimSpace(kitutil.Interface2String(tool["name"])); name != "" {
			names = append(names, name)
		}
	}
	return names
}

// ResponsesRequestHasToolSearch reports whether the request declares a
// tool_search tool (top-level tools or embedded in input items).
func ResponsesRequestHasToolSearch(req *dto.OpenAIResponsesRequest) bool {
	if req == nil {
		return false
	}
	hasToolSearch := func(raw json.RawMessage) bool {
		if !rawJSONPresent(raw) {
			return false
		}
		var tools []map[string]any
		if err := kitutil.Unmarshal(raw, &tools); err != nil {
			return false
		}
		for _, tool := range tools {
			if strings.TrimSpace(kitutil.Interface2String(tool["type"])) == "tool_search" {
				return true
			}
		}
		return false
	}
	if hasToolSearch(req.Tools) {
		return true
	}
	return hasToolSearch(responsesInputEmbeddedToolsRaw(req.Input))
}

// CollectResponsesCustomToolNamesFromRequest collects custom tool names from
// both the top-level tools and input additional_tools items.
func CollectResponsesCustomToolNamesFromRequest(req *dto.OpenAIResponsesRequest) []string {
	if req == nil {
		return nil
	}
	names := CollectResponsesCustomToolNames(req.Tools)
	if extraRaw := responsesInputEmbeddedToolsRaw(req.Input); len(extraRaw) > 0 {
		names = append(names, CollectResponsesCustomToolNames(extraRaw)...)
	}
	return names
}

// responsesInputEmbeddedToolsRaw extracts tool definitions embedded in input
// items: additional_tools items (Codex sends exec/collaboration etc. there)
// and tool_search_output items (client-side tool search results carry the
// discovered tool definitions).
func responsesInputEmbeddedToolsRaw(input json.RawMessage) json.RawMessage {
	if kitutil.GetJsonType(input) != "array" {
		return nil
	}
	var items []map[string]any
	if err := kitutil.Unmarshal(input, &items); err != nil {
		return nil
	}
	out := make([]any, 0)
	for _, item := range items {
		itemType := strings.TrimSpace(kitutil.Interface2String(item["type"]))
		if itemType != "additional_tools" && itemType != "tool_search_output" {
			continue
		}
		var tools []any
		toolsRaw, err := kitutil.Marshal(item["tools"])
		if err != nil || !rawJSONPresent(toolsRaw) {
			continue
		}
		if err := kitutil.Unmarshal(toolsRaw, &tools); err != nil {
			continue
		}
		out = append(out, tools...)
	}
	if len(out) == 0 {
		return nil
	}
	raw, err := kitutil.Marshal(out)
	if err != nil {
		return nil
	}
	return raw
}

// restoreCallsFromHistory 用历史缓存补回 previous_response_id 续轮中缺失的
// 工具调用项：input 里出现了 *_output 但对应 call 项缺失时，在该 output 之前
// 插入缓存的 call 项。返回 nil 表示无需修改。
func restoreCallsFromHistory(previousResponseID string, input json.RawMessage) json.RawMessage {
	if kitutil.GetJsonType(input) != "array" {
		return nil
	}
	var items []map[string]any
	if err := kitutil.Unmarshal(input, &items); err != nil {
		return nil
	}

	existing := make(map[string]bool)
	requested := make(map[string]bool)
	for _, item := range items {
		callID := strings.TrimSpace(kitutil.Interface2String(item["call_id"]))
		if callID == "" {
			callID = strings.TrimSpace(kitutil.Interface2String(item["id"]))
		}
		if callID == "" {
			continue
		}
		switch strings.TrimSpace(kitutil.Interface2String(item["type"])) {
		case responsesInputTypeFunctionCall, responsesInputTypeCustomToolCall, responsesInputTypeToolSearchCall:
			existing[callID] = true
			requested[callID] = true
		case responsesInputTypeFunctionCallOutput, responsesInputTypeCustomToolOutput, responsesInputTypeToolSearchOutput:
			requested[callID] = true
		}
	}
	cached := codexhistory.Lookup(previousResponseID, requested)
	if len(cached) == 0 {
		return nil
	}
	byCallID := make(map[string]map[string]any, len(cached))
	for _, call := range cached {
		byCallID[call.CallID] = call.Item
	}

	seen := make(map[string]bool)
	for id := range existing {
		seen[id] = true
	}
	restored := false
	out := make([]map[string]any, 0, len(items)+len(cached))
	for _, item := range items {
		itemType := strings.TrimSpace(kitutil.Interface2String(item["type"]))
		callID := strings.TrimSpace(kitutil.Interface2String(item["call_id"]))
		switch itemType {
		case responsesInputTypeFunctionCall, responsesInputTypeCustomToolCall, responsesInputTypeToolSearchCall:
			if callID == "" {
				callID = strings.TrimSpace(kitutil.Interface2String(item["id"]))
			}
			if callID != "" {
				seen[callID] = true
			}
		case responsesInputTypeFunctionCallOutput, responsesInputTypeCustomToolOutput, responsesInputTypeToolSearchOutput:
			if callID != "" && !seen[callID] {
				if cachedItem, ok := byCallID[callID]; ok {
					out = append(out, cachedItem)
					restored = true
				}
				seen[callID] = true
			}
		}
		out = append(out, item)
	}
	if !restored {
		return nil
	}
	raw, err := kitutil.Marshal(out)
	if err != nil {
		return nil
	}
	kitutil.LogInfo(fmt.Sprintf("responses->chat: restored %d tool call(s) from history for previous_response_id=%s", len(cached), previousResponseID))
	return raw
}

func responsesRequestToolChoiceToChat(raw json.RawMessage) (any, error) {
	if !rawJSONPresent(raw) {
		return nil, nil
	}
	if kitutil.GetJsonType(raw) == "string" {
		var choice string
		if err := kitutil.Unmarshal(raw, &choice); err != nil {
			return nil, fmt.Errorf("invalid tool_choice: %w", err)
		}
		return choice, nil
	}

	var choice map[string]any
	if err := kitutil.Unmarshal(raw, &choice); err != nil {
		return nil, fmt.Errorf("invalid tool_choice: %w", err)
	}
	if kitutil.Interface2String(choice["type"]) == "function" {
		name := strings.TrimSpace(kitutil.Interface2String(choice["name"]))
		if name != "" {
			return map[string]any{
				"type": "function",
				"function": map[string]any{
					"name": name,
				},
			}, nil
		}
	}
	return choice, nil
}

func RequestToolChoiceToChat(raw json.RawMessage) (any, error) {
	return responsesRequestToolChoiceToChat(raw)
}

func responsesRequestTextToChatResponseFormat(raw json.RawMessage) (*dto.ResponseFormat, error) {
	if !rawJSONPresent(raw) {
		return nil, nil
	}

	var textConfig map[string]any
	if err := kitutil.Unmarshal(raw, &textConfig); err != nil {
		return nil, fmt.Errorf("invalid text config: %w", err)
	}
	format, ok := textConfig["format"].(map[string]any)
	if !ok {
		return nil, nil
	}

	formatType := strings.TrimSpace(kitutil.Interface2String(format["type"]))
	if formatType == "" {
		return nil, nil
	}

	out := &dto.ResponseFormat{Type: formatType}
	if formatType == "json_schema" {
		schemaRaw, err := kitutil.Marshal(format)
		if err != nil {
			return nil, err
		}
		out.JsonSchema = schemaRaw
	}
	return out, nil
}

func RequestTextToChatResponseFormat(raw json.RawMessage) (*dto.ResponseFormat, error) {
	return responsesRequestTextToChatResponseFormat(raw)
}

func responsesImagePartToChatImageURL(part map[string]any) any {
	if imageURL, ok := part["image_url"]; ok {
		return imageURL
	}
	imageURL := map[string]any{}
	for _, key := range []string{"url", "file_id", "detail"} {
		if value, ok := part[key]; ok {
			imageURL[key] = value
		}
	}
	if len(imageURL) == 0 {
		return part
	}
	return imageURL
}

func responsesFilePartToChatFile(part map[string]any) any {
	if file, ok := part["file"]; ok {
		return file
	}
	file := map[string]any{}
	for _, key := range []string{"file_id", "file_data", "filename", "file_url"} {
		if value, ok := part[key]; ok {
			file[key] = value
		}
	}
	if len(file) == 0 {
		return part
	}
	return file
}

func responsesVideoPartToChatVideoURL(part map[string]any) any {
	if videoURL, ok := part["video_url"]; ok {
		if videoURLMap, ok := videoURL.(map[string]any); ok {
			if url := kitutil.Interface2String(videoURLMap["url"]); url != "" {
				return url
			}
		}
		return videoURL
	}
	if url := kitutil.Interface2String(part["url"]); url != "" {
		return url
	}
	return responsesPartPayload(part, "video_url")
}

func responsesPartPayload(part map[string]any, key string) any {
	if value, ok := part[key]; ok {
		return value
	}
	payload := make(map[string]any, len(part))
	for k, value := range part {
		if k == "type" {
			continue
		}
		payload[k] = value
	}
	return payload
}

func responsesCallID(item map[string]any) string {
	callID := strings.TrimSpace(kitutil.Interface2String(item["call_id"]))
	if callID != "" {
		return callID
	}
	return strings.TrimSpace(kitutil.Interface2String(item["id"]))
}

func CallID(item map[string]any) string {
	return responsesCallID(item)
}

func responsesArgumentsString(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	default:
		raw, err := kitutil.Marshal(v)
		if err != nil {
			return kitutil.Interface2String(v)
		}
		return string(raw)
	}
}

func responseToolOutputToChatContent(value any) any {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	default:
		raw, err := kitutil.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(raw)
	}
}

func responsesRawFloat(raw json.RawMessage) (*float64, error) {
	if !rawJSONPresent(raw) {
		return nil, nil
	}
	var value float64
	if err := kitutil.Unmarshal(raw, &value); err != nil {
		return nil, err
	}
	return &value, nil
}

func responsesJSONString(raw json.RawMessage) (string, error) {
	if kitutil.GetJsonType(raw) != "string" {
		return string(raw), nil
	}
	var value string
	if err := kitutil.Unmarshal(raw, &value); err != nil {
		return "", err
	}
	return value, nil
}

func rawJSONPresent(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	return kitutil.GetJsonType(raw) != "null"
}

func JSONString(raw json.RawMessage) (string, error) {
	return responsesJSONString(raw)
}

func RawJSONPresent(raw json.RawMessage) bool {
	return rawJSONPresent(raw)
}
