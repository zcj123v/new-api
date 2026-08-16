package openai

import (
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/logger"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relay/helper"
	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/types"
	"github.com/QuantumNous/new-api/service"

	"github.com/gin-gonic/gin"
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

	// 默认原始透传响应体：typed struct 重序列化会丢失上游字段（如
	// custom_tool_call 的 input、reasoning 的 encrypted_content）。仅在缺
	// output_tokens_details 时做 map 级补丁（保留所有未知字段）。
	// Some upstreams (e.g. Moonshot/Kimi) report reasoning_tokens at the top
	// level of usage rather than inside output_tokens_details, while strict
	// clients require the field.
	outBody := responseBody
	if responsesResponse.Usage != nil && responsesResponse.Usage.OutputTokensDetails == nil {
		outBody = patchResponsesBodyUsage(responseBody)
	}
	service.IOCopyBytesGracefully(c, resp, outBody)

	// compute usage
	usage := dto.Usage{}
	if responsesResponse.Usage != nil {
		usage.PromptTokens = responsesResponse.Usage.InputTokens
		usage.CompletionTokens = responsesResponse.Usage.OutputTokens
		usage.TotalTokens = responsesResponse.Usage.TotalTokens
		if responsesResponse.Usage.InputTokensDetails != nil {
			usage.PromptTokensDetails.CachedTokens = responsesResponse.Usage.InputTokensDetails.CachedTokens
			usage.PromptTokensDetails.CacheWriteTokens = responsesResponse.Usage.InputTokensDetails.CacheWriteTokens
		}
	}
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

	return &usage, nil
}

func OaiResponsesStreamHandler(c *gin.Context, info *relaycommon.RelayInfo, resp *http.Response) (*dto.Usage, *types.NewAPIError) {
	if resp == nil || resp.Body == nil {
		logger.LogError(c, "invalid response or response body")
		return nil, types.NewError(fmt.Errorf("invalid response"), types.ErrorCodeBadResponse)
	}

	defer service.CloseResponseBodyGracefully(resp)

	var usage = &dto.Usage{}
	var responseTextBuilder strings.Builder
	imageCounter := &relaycommon.ImageGenerationCallCounter{}
	imageCommitted := false

	helper.StreamScannerHandler(c, resp, info, func(data string, sr *helper.StreamResult) {

		// 检查当前数据是否包含 completed 状态和 usage 信息
		var streamResponse dto.ResponsesStreamResponse
		if err := common.UnmarshalJsonStr(data, &streamResponse); err != nil {
			// 解析失败时原样透传而不是丢事件（例如 created_at 为浮点的上游），
			// 仅无法做 usage 记账与补丁。
			logger.LogError(c, "failed to unmarshal stream response, relay raw data: "+err.Error())
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
		sendData := data
		if streamResponse.Type == "response.completed" || streamResponse.Type == "response.done" {
			sendData = patchResponsesStreamEventUsage(data)
		}
		sendResponsesStreamData(c, streamResponse, sendData)
		switch streamResponse.Type {
		case "response.completed", "response.done":
			if streamResponse.Response != nil {
				if streamResponse.Response.Usage != nil {
					if streamResponse.Response.Usage.InputTokens != 0 {
						usage.PromptTokens = streamResponse.Response.Usage.InputTokens
					}
					if streamResponse.Response.Usage.OutputTokens != 0 {
						usage.CompletionTokens = streamResponse.Response.Usage.OutputTokens
					}
					if streamResponse.Response.Usage.TotalTokens != 0 {
						usage.TotalTokens = streamResponse.Response.Usage.TotalTokens
					}
					if streamResponse.Response.Usage.InputTokensDetails != nil {
						usage.PromptTokensDetails.CachedTokens = streamResponse.Response.Usage.InputTokensDetails.CachedTokens
						usage.PromptTokensDetails.CacheWriteTokens = streamResponse.Response.Usage.InputTokensDetails.CacheWriteTokens
					}
					// Ensure reasoning_tokens is captured from top-level or output_tokens_details
					rt := streamResponse.Response.Usage.ReasoningTokens
					if rt == 0 && streamResponse.Response.Usage.OutputTokensDetails != nil {
						rt = streamResponse.Response.Usage.OutputTokensDetails.ReasoningTokens
					}
					if rt != 0 {
						usage.ReasoningTokens = rt
						usage.CompletionTokenDetails.ReasoningTokens = rt
					}
				}
				if !imageCommitted {
					if relaycommon.IsNonBillableResponsesStatus(streamResponse.Response.Status) {
						imageCounter.Reset()
						imageCounter.Commit(info)
						imageCommitted = true
					} else {
						for i := range streamResponse.Response.Output {
							idx := i
							imageCounter.Observe(&streamResponse.Response.Output[i], &idx)
						}
						imageCounter.Commit(info)
						imageCommitted = true
					}
				}
			} else if !imageCommitted {
				imageCounter.Commit(info)
				imageCommitted = true
			}
		case "response.failed", "response.incomplete", "response.cancelled", "response.canceled":
			if !imageCommitted {
				imageCounter.Reset()
				imageCounter.Commit(info)
				imageCommitted = true
			}
		case "response.output_text.delta":
			// 处理输出文本
			responseTextBuilder.WriteString(streamResponse.Delta)
		case dto.ResponsesOutputTypeItemDone:
			if streamResponse.Item != nil {
				switch streamResponse.Item.Type {
				case dto.BuildInCallWebSearchCall:
					info.CountBillableToolCall(dto.BuildInCallWebSearchCall, "")
				case dto.BuildInCallFileSearchCall:
					info.CountBillableToolCall(dto.BuildInCallFileSearchCall, "")
				case dto.BuildInCallFunctionCall:
					info.CountBillableToolCall(dto.BuildInCallFunctionCall, streamResponse.Item.Name)
				case dto.ResponsesOutputTypeImageGenerationCall:
					if !imageCommitted {
						imageCounter.Observe(streamResponse.Item, streamResponse.OutputIndex)
					}
				}
			}
		}
	})

	if usage.CompletionTokens == 0 {
		// 计算输出文本的 token 数量
		tempStr := responseTextBuilder.String()
		if len(tempStr) > 0 {
			// 非正常结束，使用输出文本的 token 数量
			completionTokens := service.CountTextToken(tempStr, info.UpstreamModelName)
			usage.CompletionTokens = completionTokens
		}
	}

	if usage.PromptTokens == 0 && usage.CompletionTokens != 0 {
		usage.PromptTokens = info.GetEstimatePromptTokens()
	}

	usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens

	helper.Done(c)

	return usage, nil
}

// patchResponsesUsageDoc patches a decoded response object in place: rounds
// created_at to an integer and populates usage.output_tokens_details when the
// upstream omitted it. Unknown fields are preserved because it operates on the
// raw map.
func patchResponsesUsageDoc(resp map[string]interface{}) {
	if v, ok := resp["created_at"].(float64); ok {
		resp["created_at"] = int64(v)
	}
	usage, ok := resp["usage"].(map[string]interface{})
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
		if details, ok := usage["completion_tokens_details"].(map[string]interface{}); ok {
			if v, ok := details["reasoning_tokens"].(float64); ok {
				reasoningTokens = int(v)
			}
		}
	}
	usage["output_tokens_details"] = map[string]interface{}{
		"reasoning_tokens": reasoningTokens,
	}
}

// patchResponsesBodyUsage patches a non-streaming Responses response body.
// Returns the original body when it cannot be parsed or re-serialized.
func patchResponsesBodyUsage(body []byte) []byte {
	var doc map[string]interface{}
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
	var doc map[string]interface{}
	if err := common.UnmarshalJsonStr(data, &doc); err != nil {
		return data
	}
	eventType, _ := doc["type"].(string)
	if eventType != "response.completed" && eventType != "response.done" {
		return data
	}
	resp, ok := doc["response"].(map[string]interface{})
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
