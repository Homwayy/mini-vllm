import torch


def cap_gpu_memory(headroom_gib=0.5):
    """Cap PyTorch's caching allocator a little below the currently free GPU memory.

    Without a cap, workloads whose tensor shapes keep changing let the allocator cache blocks until it
    holds the whole GPU. A later CUDA allocation that bypasses the allocator then fails with
    "CUDA error: out of memory", which the allocator cannot recover from. With the cap, the allocator
    frees its cache and retries when it reaches the limit.
    """
    free, total = torch.cuda.mem_get_info()
    cap = torch.cuda.memory_reserved() + free - int(headroom_gib * 2**30)
    torch.cuda.set_per_process_memory_fraction(cap / total)
    return cap
