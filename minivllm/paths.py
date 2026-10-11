"""Where the nanoGPT project is. Its GPT-2 checkpoint is this repo's second model, and its FineWeb-Edu data
gives the real-text workload W4, the draft models' training data and the GPT-2 correctness test.

By default it is a sibling of this repo (../nanoGPT); set the environment variable NANOGPT_DIR to use
another location.
"""

import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NANOGPT_DIR = os.environ.get("NANOGPT_DIR") or os.path.join(os.path.dirname(REPO_DIR), "nanoGPT")
GPT2_CKPT = os.path.join(NANOGPT_DIR, "out", "final", "ckpt.pt")
FINEWEB_EDU_DIR = os.path.join(NANOGPT_DIR, "data", "fineweb_edu")  # train.bin and val.bin: GPT-2 tokens as uint16
