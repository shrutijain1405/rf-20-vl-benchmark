from vllm.v1.sample.logits_processor.interface import LogitsProcessor
import torch

class VLLMThinkingTokenProcessor(LogitsProcessor):
    """
    Mimics the original Transformers ThinkingTokenBudgetProcessor:
    - Nudges </think> and newline tokens near 95% of max tokens.
    - Forces </think> at the end.
    """
    def __init__(self, tokenizer, max_thinking_tokens=None):
        self.tokenizer = tokenizer
        self.max_thinking_tokens = max_thinking_tokens
        self.tokens_generated = 0
        self.stopped_thinking = False

        self.think_end_token = tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.nl_token = tokenizer.encode("\n", add_special_tokens=False)[0]
        self.neg_inf = float('-inf')

    # vLLM abstract method
    def is_argmax_invariant(self) -> bool:
        return True

    # vLLM abstract method
    def update_state(self, new_token_ids: torch.Tensor):
        self.tokens_generated += new_token_ids.shape[0]

    # Core logic mimicking the Transformers version
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        # self.tokens_generated += 0  # update_state already called separately

        # Case: max_thinking_tokens = 0
        if self.max_thinking_tokens == 0 and not self.stopped_thinking:
            logits[:] = self.neg_inf
            logits[0, self.nl_token] = 0
            logits[0, self.think_end_token] = 0
            self.stopped_thinking = True
            return logits

        if self.max_thinking_tokens is not None and not self.stopped_thinking:
            progress = self.tokens_generated / self.max_thinking_tokens

            # Nudging near 95% of budget
            if progress > 0.95:
                logits[0, self.nl_token] = logits[0, self.think_end_token] * (1 + progress)
                logits[0, self.think_end_token] = logits[0, self.think_end_token] * (1 + progress)

            # Force end of thinking
            if self.tokens_generated >= self.max_thinking_tokens - 1:
                if self.tokens_generated == self.max_thinking_tokens - 1:
                    logits[:] = self.neg_inf
                    logits[0, self.nl_token] = 0
                else:
                    logits[:] = self.neg_inf
                    logits[0, self.think_end_token] = 0
                    self.stopped_thinking = True

        return logits
