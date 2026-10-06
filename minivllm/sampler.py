import torch

from minivllm.sequence import SamplingParams


class Sampler:
    """Per-request temperature, top-k and top-p; temperature 0 is greedy.

    Sampling uses argmax(probs / E) with E ~ Exponential(1), which picks token i with probability probs[i]
    and needs no CPU-GPU synchronization.
    """

    def __init__(self, vocab_limit=None):
        self.vocab_limit = vocab_limit  # ids from here up are padding and never sampled

    @torch.no_grad()
    def __call__(self, logits, params: list[SamplingParams]):
        logits = logits.float()
        if self.vocab_limit is not None:
            logits[:, self.vocab_limit :] = float("-inf")
        greedy = logits.argmax(dim=-1)
        temps = [p.temperature for p in params]
        if all(t == 0 for t in temps):
            return greedy
        device = logits.device
        t = torch.tensor([t if t > 0 else 1.0 for t in temps], device=device)
        logits = logits / t[:, None]
        if any(p.top_k > 0 or p.top_p < 1.0 for p in params):
            logits = self.top_k_top_p(logits, params)
        probs = torch.softmax(logits, dim=-1)
        sampled = probs.div_(torch.empty_like(probs).exponential_()).argmax(dim=-1)
        is_greedy = torch.tensor([t == 0 for t in temps], device=device)
        return torch.where(is_greedy, greedy, sampled)

    @staticmethod
    def top_k_top_p(logits, params):
        V = logits.shape[-1]
        device = logits.device
        k = torch.tensor([p.top_k if p.top_k > 0 else V for p in params], device=device)
        p = torch.tensor([p.top_p for p in params], device=device)
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        rank = torch.arange(V, device=device)
        sorted_logits[rank[None, :] >= k[:, None]] = float("-inf")
        probs = torch.softmax(sorted_logits, dim=-1)
        # drop a token once the more likely tokens already reach top_p; the most likely one always stays
        sorted_logits[probs.cumsum(dim=-1) - probs >= p[:, None]] = float("-inf")
        return logits.scatter(-1, order, sorted_logits)
