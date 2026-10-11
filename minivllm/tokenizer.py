"""Text in and out of the engine, for the HTTP server (minivllm/server.py).

Qwen3 uses its Hugging Face tokenizer (tokenizer.json, and the chat template in tokenizer_config.json); GPT-2
uses tiktoken's "gpt2" encoding, as nanoGPT did, and has no chat template. Both are byte-level BPE, so one
character can be split across tokens: streamed text is decoded incrementally (IncrementalDetokenizer), and
a stop string can arrive in pieces (StopChecker).
"""


class Tokenizer:
    def __init__(self, model: str, path: str):
        if model == "qwen3":
            from transformers import AutoTokenizer

            self.hf = AutoTokenizer.from_pretrained(path)
            self.special = set(self.hf.all_special_ids)
            self.vocab_size = len(self.hf)
            self.has_chat_template = True
        elif model == "gpt2":
            import tiktoken

            self.tt = tiktoken.get_encoding("gpt2")
            self.special = {self.tt.eot_token}
            self.vocab_size = self.tt.n_vocab
            self.has_chat_template = False
        else:
            raise ValueError(f"no tokenizer for model {model}")
        self.model = model

    def encode(self, text: str) -> list[int]:
        if self.model == "qwen3":
            return self.hf.encode(text, add_special_tokens=False)
        return self.tt.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, ids) -> str:
        """Special tokens (EOS) are left out; an incomplete character at the end becomes U+FFFD."""
        if self.model == "qwen3":
            return self.hf.decode(ids, skip_special_tokens=True)
        return self.tt.decode([i for i in ids if i not in self.special])

    def apply_chat_template(self, messages, **kwargs) -> list[int]:
        out = self.hf.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)
        return list(out if isinstance(out, list) else out["input_ids"])


class IncrementalDetokenizer:
    """Turns a stream of token ids into text deltas.

    Each call decodes the tokens from the start of the previous delta's tokens (prefix) to the end, and
    returns what goes beyond the text of the previous delta's tokens alone (prefix..read), but only once it
    does not end in U+FFFD, the decoder's mark for a character still missing bytes. finish() returns the
    rest. For byte-level BPE the deltas add up to the decoding of all the tokens.
    """

    def __init__(self, decode):
        self.decode = decode
        self.ids: list[int] = []
        self.prefix = 0
        self.read = 0

    def add(self, ids) -> str:
        self.ids.extend(ids)
        return self._delta(final=False)

    def finish(self) -> str:
        return self._delta(final=True)

    def _delta(self, final):
        if self.read == len(self.ids):
            return ""
        before = self.decode(self.ids[self.prefix : self.read])
        text = self.decode(self.ids[self.prefix :])
        if not final and (len(text) <= len(before) or text.endswith("�")):
            return ""
        self.prefix, self.read = self.read, len(self.ids)
        return text[len(before) :]


class StopChecker:
    """Cuts the output before the first stop string. Until the output finishes, it holds back the last
    (longest stop string - 1) characters, which could be the start of a stop string."""

    def __init__(self, stop: list[str]):
        self.stop = [s for s in stop if s]
        self.hold = max(map(len, self.stop)) - 1
        self.text = ""
        self.sent = 0

    def add(self, delta: str, final: bool) -> tuple[str, bool]:
        """Returns the text to send now and whether a stop string ended the output. A stop string starts at or
        after self.sent: text before it was sent only once it could no longer begin one."""
        self.text += delta
        hits = [i for i in (self.text.find(s, self.sent) for s in self.stop) if i >= 0]
        if hits:
            end = min(hits)
            out, self.sent = self.text[self.sent : end], end
            return out, True
        end = len(self.text) if final else max(self.sent, len(self.text) - self.hold)
        out, self.sent = self.text[self.sent : end], end
        return out, False
