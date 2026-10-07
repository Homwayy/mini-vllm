import itertools
from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class SamplingParams:
    temperature: float = 1.0  # 0 means greedy
    top_k: int = 0  # 0 means no top-k cut
    top_p: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False


class Status(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    _ids = itertools.count()

    def __init__(self, prompt_token_ids, params: SamplingParams, arrival_time: float):
        self.seq_id = next(Sequence._ids)
        self.token_ids = list(prompt_token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.params = params
        self.status = Status.WAITING
        self.block_table: list[int] = []
        self.num_cached_tokens = 0  # leading tokens whose K/V came from the prefix cache at admission
        self.num_computed_tokens = 0  # leading tokens whose K/V is in the pool
        self.block_hashes: list[int] = []  # prefix hashes of the full blocks, as far as computed
        self.arrival_time = arrival_time
        self.first_token_time = None
        self.token_times: list[float] = []  # when each completion token was produced
        self.finish_time = None
        self.num_preemptions = 0

    def __len__(self):
        return len(self.token_ids)

    @property
    def num_completion_tokens(self):
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def max_len(self):
        return self.num_prompt_tokens + self.params.max_tokens
