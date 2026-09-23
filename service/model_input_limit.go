package service

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/constant"
	"github.com/QuantumNous/new-api/logger"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relaykit/dto"
	"github.com/QuantumNous/new-api/relaykit/types"
	"github.com/QuantumNous/new-api/setting"

	"github.com/gin-gonic/gin"
)

// CheckModelMaxInputTokens rejects text requests whose input exceeds the limit
// configured for the requested model, before the request is dispatched upstream
// or billed. The wording matches what OpenAI-compatible clients expect for a
// context overflow, so they can run their own recovery path, and the error is
// marked as non-retryable because the limit belongs to the model rather than to
// one channel.
func CheckModelMaxInputTokens(c *gin.Context, info *relaycommon.RelayInfo) *types.NewAPIError {
	if info == nil {
		return nil
	}
	limit, limited := resolveMaxInputTokens(c, info.OriginModelName)
	if !limited {
		return nil
	}
	estimated, ok := estimateRequestInputTokens(info)
	if !ok {
		logger.LogDebug(c, "max input tokens check skipped: unrecognized request shape, model=%s", info.OriginModelName)
		return nil
	}
	if estimated <= limit {
		return nil
	}
	message := fmt.Sprintf("This model's maximum context length is %d tokens, however you requested %d tokens. Please reduce the length of your input.", limit, estimated)
	return types.NewError(
		errors.New(message),
		types.ErrorCodeContextLengthExceeded,
		types.ErrOptionWithSkipRetry(),
		types.ErrOptionWithStatusCode(http.StatusBadRequest),
	)
}

// resolveMaxInputTokens returns the effective input limit for the model on the
// channel selected for this request: a positive channel max_input_tokens setting
// wins over the model-level option, a negative one disables the limit for that
// channel, and 0/unset inherits the model-level option.
func resolveMaxInputTokens(c *gin.Context, model string) (int, bool) {
	if model == "" {
		return 0, false
	}
	if channelSetting, ok := common.GetContextKeyType[dto.ChannelSettings](c, constant.ContextKeyChannelSetting); ok {
		if channelSetting.MaxInputTokens > 0 {
			return channelSetting.MaxInputTokens, true
		}
		if channelSetting.MaxInputTokens < 0 {
			return 0, false
		}
	}
	limit := setting.GetModelMaxInputTokens(model)
	return limit, limit > 0
}

// estimateRequestInputTokens estimates the model-visible input of the two text
// request shapes the limit covers: chat "messages" and responses "input", both
// including their tool definitions. File-like content (images, audio, files) is
// not counted here; those relay formats are skipped by the caller anyway.
func estimateRequestInputTokens(info *relaycommon.RelayInfo) (int, bool) {
	var meta *types.TokenCountMeta
	switch request := info.Request.(type) {
	case *dto.GeneralOpenAIRequest:
		if len(request.Messages) == 0 {
			return 0, false
		}
		meta = request.GetTokenCountMeta()
	case *dto.OpenAIResponsesRequest:
		if len(request.Input) == 0 {
			return 0, false
		}
		meta = request.GetTokenCountMeta()
	default:
		return 0, false
	}
	if meta == nil {
		return 0, false
	}
	tokens := CountTokenInput(meta.CombineText, info.OriginModelName)
	if info.RelayFormat == types.RelayFormatOpenAI {
		// Mirror CountRequestToken's per-message/name/tool overhead so this check
		// and the pre-consume estimate do not disagree.
		tokens += meta.ToolsCount*8 + meta.MessagesCount*3 + meta.NameCount*3 + 3
	}
	return tokens, true
}
