"""Minimal multi-GPU support (single or multiple nodes via torchrun).

Training: every rank holds a full copy of the (frozen) base model + the LoRA adapter on its own GPU, processes its own slice of the
data / its own share of the GRPO questions, and the gradients of the trainable LoRA parameters are all-reduced (averaged) before
each optimizer step. Identical to DDP for LoRA, without wrapper/gradient-checkpointing interactions.
Evaluation: each rank (or each SLURM array task) is one shard of the test instances.
"""
import os
import datetime


def world():
    return int(os.environ.get("WORLD_SIZE", "1"))

def rank():
    return int(os.environ.get("RANK", "0"))

def local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))

def is_main():
    return rank() == 0


def init():
    """Call once at start-up. Sets the CUDA device of this process; initialises NCCL when launched with torchrun (WORLD_SIZE > 1)."""
    import torch
    if torch.cuda.is_available(): torch.cuda.set_device(local_rank())
    if world() > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3))


def barrier():
    if world() > 1:
        import torch.distributed as dist
        dist.barrier()


def all_reduce_sum(t):
    if world() > 1:
        import torch.distributed as dist
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_reduce_mean_grads(params):
    """Average .grad of the given parameters across ranks (bucketed into one flat tensor for speed)."""
    if world() == 1: return
    import torch
    grads = [p.grad for p in params if p.grad is not None]
    if not grads: return
    flat = torch.cat([g.reshape(-1) for g in grads]); all_reduce_sum(flat); flat /= world()
    off = 0
    for g in grads:
        n = g.numel(); g.copy_(flat[off:off + n].view_as(g)); off += n


def scalar_sum(x):
    """Sum a python float across ranks."""
    if world() == 1: return float(x)
    import torch
    t = torch.tensor([float(x)], device="cuda"); all_reduce_sum(t); return float(t.item())


def finish():
    if world() > 1:
        import torch.distributed as dist
        if dist.is_initialized(): dist.barrier(); dist.destroy_process_group()
