package openai

import (
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	"github.com/QuantumNous/new-api/logger"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relay/helper"
	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/types"
	"github.com/QuantumNous/new-api/service"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func OaiResponsesHandler(c *gin.Context, info *relaycommon.RelayInfo, resp *http.Response) (*dto.Usage, *types.NewAPIError) {
	defer service.CloseResponseBodyGracefully(resp)

	// read response body
	var responsesResponse dto.OpenAIResponsesResponse
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, types.NewOpenAIError(err, types.ErrorCodeReadResponseBodyFailed, http.StatusInternalServerError)
	}
	err = common.Unmarshal(responseBody, &responsesResponse)
	if err != nil {
		return nil, types.NewOpenAIError(err, types.ErrorCodeBadResponseBody, http.StatusInternalServerError)
	}
	if oaiError := responsesResponse.GetOpenAIError(); oaiError != nil && oaiError.Type != "" {
		return nil, types.WithOpenAIError(*oaiError, resp.StatusCode)
	}

	info.ObserveResponseModel(responsesResponse.Model)

	// 默认原始透传响应体：typed struct 重序列化会丢失上游字段（如
	// custom_tool_call 的 input、reasoning 的 encrypted_content）。仅在缺
	// output_tokens_details 时做 map 级补丁（保留所有未知字段）。
	// Some upstreams (e.g. Moonshot/Kimi) report reasoning_tokens at the top
	// level of usage rather than inside output_tokens_details, while strict
	// clients require the field.
	outBody := rewriteSGLangResponsesCreatedAt(info, responseBody, "created_at", responsesResponse.CreatedAt)
	if responsesResponse.Usage != nil && responsesResponse.Usage.OutputTokensDetails == nil {
		outBody = patchResponsesBodyUsage(outBody)
	}
	service.IOCopyBytesGracefully(c, resp, outBody)

	// compute usage
	usage := &dto.Usage{}
	service.ApplyResponsesUsage(usage, responsesResponse.Usage)
	// Count actual tool invocations from Output (not tool declarations).
	for _, output := range responsesResponse.Output {
		switch output.Type {
		case dto.BuildInCallWebSearchCall:
			info.CountBillableToolCall(dto.BuildInCallWebSearchCall, "")
		case dto.BuildInCallFileSearchCall:
			info.CountBillableToolCall(dto.BuildInCallFileSearchCall, "")
		case dto.BuildInCallFunctionCall:
			info.CountBillableToolCall(dto.BuildInCallFunctionCall, output.Name)
		}
	}

	imageCounter := &relaycommon.ImageGenerationCallCounter{}
	if !relaycommon.IsNonBillableResponsesStatus(responsesResponse.Status) {
		for i := range responsesResponse.Output {
			idx := i
			imageCounter.Observe(&responsesResponse.Output[i], &idx)
		}
	}
	imageCounter.Commit(info)

	return usage, nil
}

func OaiResponsesStreamHandler(c *gin.Context, info *relaycommon.RelayInfo, resp *http.Response) (*dto.Usage, *types.NewAPIError) {
	if resp == nil || resp.Body == nil {
		logger.LogError(c, "invalid response or response body")
		return nil, types.NewError(fmt.Errorf("invalid response"), types.ErrorCodeBadResponse)
	}

	defer service.CloseResponseBodyGracefully(resp)

	accumulator := service.NewResponsesUsageAccumulator(info)

	// 终止事件跟踪：上游流可能在没有 response.completed/failed/incomplete
	// 的情况下直接 EOF（半死连接、上游崩溃等）。严格客户端（如 Codex）会
	// 一直等待终止事件而表现为"卡死"。流结束时若未见到终止事件，兜底合成
	// response.incomplete 让客户端干净地失败并重试。
	terminalSeen := false
	// contentSeen 记录是否收到过除 created/in_progress 外的任何实质事件；
	// EOF 兜底时零输出合成 response.failed（让客户端报错而非静默重试），
	// 有输出则仍合成 response.incomplete。
	contentSeen := false
	var responseSnapshot map[string]any

	helper.StreamScannerHandler(c, resp, info, func(data string, sr *helper.StreamResult) {

		// 检查当前数据是否包含 completed 状态和 usage 信息
		var streamResponse dto.ResponsesStreamResponse
		if err := common.UnmarshalJsonStr(data, &streamResponse); err != nil {
			// 解析失败时原样透传而不是丢事件（例如 created_at 为浮点的上游），
			// 仅无法做 usage 记账与补丁。
			logger.LogError(c, "failed to unmarshal stream response, relay raw data: "+err.Error())
			if doc := parseResponsesStreamEventDoc(data); doc != nil {
				if t, _ := doc["type"].(string); isResponsesTerminalEventType(t) {
					terminalSeen = true
				}
				if t, _ := doc["type"].(string); t != "" && t != "response.created" && t != "response.in_progress" {
					contentSeen = true
				}
				if responseSnapshot == nil {
					if r, ok := doc["response"].(map[string]any); ok {
						responseSnapshot = r
					}
				}
			}
			sendResponsesStreamData(c, streamResponse, patchResponsesStreamEventUsage(data))
			return
		}
		// 事件默认原始透传：typed struct 重序列化会丢失上游字段（如
		// custom_tool_call 的 input、reasoning 的 encrypted_content、
		// sequence_number），导致严格客户端（如 Codex）拿不到工具调用。
		// 仅 completed/done 事件在缺 output_tokens_details 时做 map 级补丁。
		// Some upstreams (e.g. Moonshot/Kimi) report reasoning_tokens at the top
		// level of usage rather than inside output_tokens_details, while strict
		// clients require the field.
		if streamResponse.Type == "response.completed" || streamResponse.Type == "response.done" {
			data = patchResponsesStreamEventUsage(data)
		}
		if isResponsesTerminalEventType(streamResponse.Type) {
			terminalSeen = true
		}
		if streamResponse.Type != "" && streamResponse.Type != "response.created" && streamResponse.Type != "response.in_progress" {
			contentSeen = true
		}
		if responseSnapshot == nil {
			if doc := parseResponsesStreamEventDoc(data); doc != nil {
				if r, ok := doc["response"].(map[string]any); ok {
					responseSnapshot = r
				}
			}
		}
		if streamResponse.Response != nil {
			data = string(rewriteSGLangResponsesCreatedAt(info, []byte(data), "response.created_at", streamResponse.Response.CreatedAt))
		}
		sendResponsesStreamData(c, streamResponse, data)
		accumulator.Observe(&streamResponse)
	})

	common.SetContextKey(c, constant.ContextKeyResponseStreamStatus, info.StreamStatus)
	info.StreamStatus.RequireTerminal()

	// 上游流结束但未发任何终止事件（半死连接、上游崩溃、超时）：给客户端
	// 合成终止事件，让严格客户端（如 Codex）干净地失败，而不是无限等待。
	// 零实质输出（连一个 delta 都没有）时合成 response.failed——这种情况
	// 多半是上游转换层根本没产出内容，incomplete 会让客户端无谓重试；
	// 已有部分输出时仍合成 response.incomplete 让客户端重试续传。
	// 客户端已断开（client_gone）时跳过——写了也收不到。
	if !terminalSeen && info.StreamStatus != nil &&
		info.StreamStatus.EndReason != relaycommon.StreamEndReasonClientGone {
		if contentSeen {
			logger.LogWarn(c, fmt.Sprintf("responses stream ended without terminal event (reason=%s, received=%d), synthesizing response.incomplete",
				info.StreamStatus.EndReason, info.ReceivedResponseCount))
			if eventData := buildResponsesIncompleteEvent(responseSnapshot, info.StreamStatus.EndReason); eventData != "" {
				_ = helper.ResponseChunkData(c, dto.ResponsesStreamResponse{Type: "response.incomplete"}, eventData)
			}
		} else {
			logger.LogWarn(c, fmt.Sprintf("responses stream ended without terminal event and zero content (reason=%s, received=%d), synthesizing response.failed",
				info.StreamStatus.EndReason, info.ReceivedResponseCount))
			if eventData := buildResponsesFailedEvent(responseSnapshot, info.StreamStatus.EndReason); eventData != "" {
				_ = helper.ResponseChunkData(c, dto.ResponsesStreamResponse{Type: "response.failed"}, eventData)
			}
		}
	}

	helper.Done(c)

	return accumulator.Finish(), nil
}

func rewriteSGLangResponsesCreatedAt(info *relaycommon.RelayInfo, payload []byte, path string, createdAt dto.IntValue) []byte {
	if info.GetChannelType() != constant.ChannelTypeSGLang {
		return payload
	}
	if !gjson.GetBytes(payload, path).Exists() {
		return payload
	}
	patched, err := sjson.SetBytes(payload, path, int(createdAt))
	if err != nil {
		return payload
	}
	return patched
}

// isResponsesTerminalEventType reports whether a Responses stream event type
// terminates the response lifecycle.
func isResponsesTerminalEventType(eventType string) bool {
	switch eventType {
	case "response.completed", "response.done", "response.failed",
		"response.incomplete", "response.cancelled", "response.canceled":
		return true
	}
	return false
}

// parseResponsesStreamEventDoc decodes a raw stream event into a map,
// preserving unknown fields. Returns nil when the data is not valid JSON.
func parseResponsesStreamEventDoc(data string) map[string]any {
	var doc map[string]any
	if err := common.UnmarshalJsonStr(data, &doc); err != nil {
		return nil
	}
	return doc
}

// buildResponsesIncompleteEvent builds a synthetic response.incomplete event
// for streams that ended without a terminal event. It reuses the last seen
// upstream response object (preserving id/model/etc.) and fills the fields a
// strict client requires: integer created_at, status/incomplete_details, and
// a zero usage with both detail objects. Returns "" when serialization fails.
func buildResponsesIncompleteEvent(snapshot map[string]any, endReason relaycommon.StreamEndReason) string {
	resp := make(map[string]any, len(snapshot)+6)
	for k, v := range snapshot {
		resp[k] = v
	}
	if _, ok := resp["id"]; !ok || resp["id"] == "" {
		resp["id"] = "resp_" + common.GetRandomString(24)
	}
	resp["object"] = "response"
	if _, ok := resp["created_at"]; !ok {
		resp["created_at"] = time.Now().Unix()
	}
	resp["status"] = "incomplete"
	reason := "upstream_eof"
	if endReason == relaycommon.StreamEndReasonTimeout {
		reason = "upstream_timeout"
	}
	resp["incomplete_details"] = map[string]any{"reason": reason}
	if _, ok := resp["usage"]; !ok {
		resp["usage"] = map[string]any{
			"input_tokens":          0,
			"input_tokens_details":  map[string]any{"cached_tokens": 0},
			"output_tokens":         0,
			"output_tokens_details": map[string]any{"reasoning_tokens": 0},
			"total_tokens":          0,
		}
	}
	// 复用现有补丁：created_at 取整 + usage 补 output_tokens_details。
	patchResponsesUsageDoc(resp)
	doc := map[string]any{
		"type":     "response.incomplete",
		"response": resp,
	}
	patched, err := common.Marshal(doc)
	if err != nil {
		return ""
	}
	return string(patched)
}

// buildResponsesFailedEvent builds a synthetic response.failed event for
// streams that ended without any terminal event AND without any content
// (the upstream produced nothing at all). Shape mirrors
// buildResponsesIncompleteEvent but carries an error object instead of
// incomplete_details. Returns "" when serialization fails.
func buildResponsesFailedEvent(snapshot map[string]any, endReason relaycommon.StreamEndReason) string {
	resp := make(map[string]any, len(snapshot)+6)
	for k, v := range snapshot {
		resp[k] = v
	}
	if _, ok := resp["id"]; !ok || resp["id"] == "" {
		resp["id"] = "resp_" + common.GetRandomString(24)
	}
	resp["object"] = "response"
	if _, ok := resp["created_at"]; !ok {
		resp["created_at"] = time.Now().Unix()
	}
	resp["status"] = "failed"
	code := "stream_truncated"
	if endReason == relaycommon.StreamEndReasonTimeout {
		code = "upstream_timeout"
	}
	resp["error"] = map[string]any{
		"code":    code,
		"message": fmt.Sprintf("upstream stream ended with no content (reason=%s)", endReason),
	}
	if _, ok := resp["usage"]; !ok {
		resp["usage"] = map[string]any{
			"input_tokens":          0,
			"input_tokens_details":  map[string]any{"cached_tokens": 0},
			"output_tokens":         0,
			"output_tokens_details": map[string]any{"reasoning_tokens": 0},
			"total_tokens":          0,
		}
	}
	patchResponsesUsageDoc(resp)
	doc := map[string]any{
		"type":     "response.failed",
		"response": resp,
	}
	patched, err := common.Marshal(doc)
	if err != nil {
		return ""
	}
	return string(patched)
}

// patchResponsesUsageDoc patches a decoded response object in place: rounds
// created_at to an integer and populates usage.output_tokens_details when the
// upstream omitted it. Unknown fields are preserved because it operates on the
// raw map.
func patchResponsesUsageDoc(resp map[string]any) {
	if v, ok := resp["created_at"].(float64); ok {
		resp["created_at"] = int64(v)
	}
	usage, ok := resp["usage"].(map[string]any)
	if !ok || usage == nil {
		return
	}
	if _, exists := usage["output_tokens_details"]; exists {
		return
	}
	reasoningTokens := 0
	if v, ok := usage["reasoning_tokens"].(float64); ok {
		reasoningTokens = int(v)
	}
	if reasoningTokens == 0 {
		if details, ok := usage["completion_tokens_details"].(map[string]any); ok {
			if v, ok := details["reasoning_tokens"].(float64); ok {
				reasoningTokens = int(v)
			}
		}
	}
	usage["output_tokens_details"] = map[string]any{
		"reasoning_tokens": reasoningTokens,
	}
}

// patchResponsesBodyUsage patches a non-streaming Responses response body.
// Returns the original body when it cannot be parsed or re-serialized.
func patchResponsesBodyUsage(body []byte) []byte {
	var doc map[string]any
	if err := common.Unmarshal(body, &doc); err != nil {
		return body
	}
	patchResponsesUsageDoc(doc)
	patched, err := common.Marshal(doc)
	if err != nil {
		return body
	}
	return patched
}

// patchResponsesStreamEventUsage patches a response.completed/done stream
// event. Returns the original data when it cannot be parsed or re-serialized.
func patchResponsesStreamEventUsage(data string) string {
	var doc map[string]any
	if err := common.UnmarshalJsonStr(data, &doc); err != nil {
		return data
	}
	eventType, _ := doc["type"].(string)
	if eventType != "response.completed" && eventType != "response.done" {
		return data
	}
	resp, ok := doc["response"].(map[string]any)
	if !ok {
		return data
	}
	patchResponsesUsageDoc(resp)
	patched, err := common.Marshal(doc)
	if err != nil {
		return data
	}
	return string(patched)
}
