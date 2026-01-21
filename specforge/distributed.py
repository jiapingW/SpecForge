from datetime import timedelta
from typing import Any, Optional

import torch
import torch.distributed as dist
from yunchang.globals import PROCESS_GROUP, set_seq_parallel_pg

from specforge.utils import print_with_rank

_TARGET_DEVICE_MESH = None
_TARGET_TP_GROUP = None
_TARGET_TP_DEVICE_MESH = None
_TARGET_DP_GROUP = None
_TARGET_DP_DEVICE_MESH = None

_DRAFT_DEVICE_MESH = None
_DRAFT_DP_GROUP = None
_DRAFT_DP_DEVICE_MESH = None
_DRAFT_TP_GROUP = None
_DRAFT_TP_DEVICE_MESH = None
_DRAFT_SP_GROUP = None

_SP_ULYSSES_GROUP = None
_SP_RING_GROUP = None


def get_target_tp_group():
    global _TARGET_TP_GROUP
    return _TARGET_TP_GROUP


def get_target_dp_group():
    global _TARGET_DP_GROUP
    return _TARGET_DP_GROUP

def get_target_device_mesh():
    global _TARGET_DEVICE_MESH
    return _TARGET_DEVICE_MESH


def get_target_tp_device_mesh():
    global _TARGET_TP_DEVICE_MESH
    return _TARGET_TP_DEVICE_MESH

def get_target_dp_device_mesh():
    global _TARGET_DP_DEVICE_MESH
    return _TARGET_DP_DEVICE_MESH


def get_draft_dp_group():
    global _DRAFT_DP_GROUP
    return _DRAFT_DP_GROUP


def get_draft_sp_group():
    global _DRAFT_SP_GROUP
    return _DRAFT_SP_GROUP

def get_draft_tp_group():
    global _DRAFT_TP_GROUP
    return _DRAFT_TP_GROUP

def get_draft_device_mesh():
    global _DRAFT_DEVICE_MESH
    return _DRAFT_DEVICE_MESH

def get_draft_dp_device_mesh():
    global _DRAFT_DP_DEVICE_MESH
    return _DRAFT_DP_DEVICE_MESH

def get_draft_tp_device_mesh():
    global _DRAFT_TP_DEVICE_MESH
    return _DRAFT_TP_DEVICE_MESH

def get_draft_sp_ulysses_group():
    global _SP_ULYSSES_GROUP
    return _SP_ULYSSES_GROUP


def get_draft_sp_ring_group():
    global _SP_RING_GROUP
    return _SP_RING_GROUP


def init_distributed(
    timeout: int = 10, target_tp_size: int = 1, tp_size: int = 1, sp_ulysses_size: int = 1, sp_ring_size: int = 1
):
    """Initialize distributed training.

    Args:
        timeout(int): Timeout for collective communication in minutes
        target_tp_size(int): The degree of tensor parallelism of target model
        tp_size(int): The degree of tensor parallelism of darft model
        sp_ulysses_size(int): The degree of sequence parallelism of ulysses of draft model
        sp_ring_size(int): The degree of sequence parallelism of ring of draft model
    """
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print_with_rank(f"bind to device {local_rank}")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    target_dp_size = world_size // target_tp_size
    assert (
        world_size == target_tp_size * target_dp_size
    ), (
        f"world size must be divisible by target_tp_size, "
        f"now {world_size=}, {target_tp_size=}, {target_dp_size=}"
    )

    # Create Target Model's device mesh: (dp, tp)
    target_device_mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        (target_dp_size, target_tp_size),
        mesh_dim_names=("target_dp", "target_tp"),
    )
    print_with_rank(f"Target device mesh: {target_device_mesh}")

    target_tp_group = target_device_mesh.get_group("target_tp")
    target_dp_group = target_device_mesh.get_group("target_dp")
    target_tp_device_mesh = dist.DeviceMesh.from_group(target_tp_group, device_type="cuda")
    target_dp_device_mesh = dist.DeviceMesh.from_group(target_dp_group, device_type="cuda")

    # ============================================================
    # 2. Draft Model 并行配置（SP + TP + DP）
    # ============================================================
    total_draft_sp_size = sp_ulysses_size * sp_ring_size
    
    assert (
        world_size % (total_draft_sp_size * tp_size) == 0
    ), (
        f"World size ({world_size}) cannot be evenly divided by "
        f"draft_tp_size ({tp_size}) * total_sp_size ({total_draft_sp_size})"
    )

    draft_dp_size = world_size // (total_draft_sp_size * tp_size)
    
    # 创建Draft Model的device mesh: (draft_dp, draft_tp, sp) 或 (draft_dp, sp)
    if tp_size > 1:
        draft_device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            (draft_dp_size, tp_size, total_draft_sp_size),
            mesh_dim_names=("draft_dp", "draft_tp", "sp"),
        )
    else:
        draft_device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            (draft_dp_size, total_draft_sp_size),
            mesh_dim_names=("draft_dp", "sp"),
        )

    print_with_rank(f"Draft device mesh: {draft_device_mesh}")

    # 设置sequence parallelism进程组
    set_seq_parallel_pg(sp_ulysses_size, sp_ring_size, rank, world_size)

    # 获取Draft Model的各个进程组
    draft_dp_group = draft_device_mesh.get_group("draft_dp")
    draft_sp_group = draft_device_mesh.get_group("sp")
    draft_dp_device_mesh = dist.DeviceMesh.from_group(draft_dp_group, device_type="cuda")
    
    if tp_size > 1:
        draft_tp_group = draft_device_mesh.get_group("draft_tp")
        draft_tp_device_mesh = dist.DeviceMesh.from_group(draft_tp_group, device_type="cuda")
    else:
        draft_tp_group = None
        draft_tp_device_mesh = None

    # 获取SP进程组（来自PROCESS_GROUP模块）
    sp_ulysses_group = PROCESS_GROUP.ULYSSES_PG
    sp_ring_group = PROCESS_GROUP.RING_PG

    # ============================================================
    # 3. 计算并验证并行度配置
    # ============================================================
    print_with_rank("=" * 80)
    print_with_rank("Distributed Parallelism Configuration:")
    print_with_rank(f"  World Size: {world_size}")
    print_with_rank(f"  Target Model:")
    print_with_rank(f"    TP Size: {target_tp_size}, DP Size: {target_dp_size}")
    print_with_rank(f"  Draft Model:")
    print_with_rank(f"    TP Size: {tp_size}, DP Size: {draft_dp_size}")
    print_with_rank(f"    SP (Ulysses x Ring): {sp_ulysses_size} x {sp_ring_size} = {total_draft_sp_size}")
    print_with_rank(f"  Verification: {target_tp_size} * {target_dp_size} = {target_tp_size * target_dp_size}")
    print_with_rank(f"  Verification: {tp_size} * {draft_dp_size} * {total_draft_sp_size} = {tp_size * draft_dp_size * total_draft_sp_size}")
    print_with_rank("=" * 80)

    # ============================================================
    # 4. 全局变量设置
    # ============================================================
    global _TARGET_DEVICE_MESH, _TARGET_TP_GROUP,_TARGET_TP_DEVICE_MESH,_TARGET_DP_GROUP,_TARGET_DP_DEVICE_MESH,_DRAFT_DEVICE_MESH, \
        _DRAFT_DP_GROUP,_DRAFT_DP_DEVICE_MESH,_DRAFT_TP_GROUP,_DRAFT_TP_DEVICE_MESH,_DRAFT_SP_GROUP,_SP_ULYSSES_GROUP,_SP_RING_GROUP

    # Target Model 相关全局变量
    _TARGET_DEVICE_MESH = target_device_mesh
    _TARGET_TP_GROUP = target_tp_group
    _TARGET_TP_DEVICE_MESH = target_tp_device_mesh
    _TARGET_DP_GROUP = target_dp_group
    _TARGET_DP_DEVICE_MESH = target_dp_device_mesh

    # Draft Model 相关全局变量
    _DRAFT_DEVICE_MESH = draft_device_mesh
    _DRAFT_DP_GROUP = draft_dp_group
    _DRAFT_DP_DEVICE_MESH = draft_dp_device_mesh
    _DRAFT_TP_GROUP = draft_tp_group
    _DRAFT_TP_DEVICE_MESH = draft_tp_device_mesh
    _DRAFT_SP_GROUP = draft_sp_group

    # Sequence Parallelism 相关全局变量
    _SP_ULYSSES_GROUP = sp_ulysses_group
    _SP_RING_GROUP = sp_ring_group

    print_with_rank("Distributed environment initialized successfully")


def destroy_distributed():
    global _TARGET_DEVICE_MESH,_TARGET_TP_GROUP,_TARGET_TP_DEVICE_MESH,_TARGET_DP_GROUP,_TARGET_DP_DEVICE_MESH,_DRAFT_DEVICE_MESH, \
        _DRAFT_DP_GROUP,_DRAFT_DP_DEVICE_MESH,_DRAFT_TP_GROUP,_DRAFT_TP_DEVICE_MESH,_DRAFT_SP_GROUP,_SP_ULYSSES_GROUP,_SP_RING_GROUP

    # 销毁Target Model相关进程组
    if _TARGET_TP_GROUP is not None:
        try:
            dist.destroy_process_group(_TARGET_TP_GROUP)
            print_with_rank("Destroyed target TP group")
        except Exception as e:
            print_with_rank(f"Warning: Failed to destroy target TP group: {e}")
    
    if _TARGET_DP_GROUP is not None:
        try:
            dist.destroy_process_group(_TARGET_DP_GROUP)
            print_with_rank("Destroyed target DP group")
        except Exception as e:
            print_with_rank(f"Warning: Failed to destroy target DP group: {e}")

    # 销毁Draft Model相关进程组
    if _DRAFT_TP_GROUP is not None:
        try:
            dist.destroy_process_group(_DRAFT_TP_GROUP)
            print_with_rank("Destroyed draft TP group")
        except Exception as e:
            print_with_rank(f"Warning: Failed to destroy draft TP group: {e}")
    
    if _DRAFT_DP_GROUP is not None:
        try:
            dist.destroy_process_group(_DRAFT_DP_GROUP)
            print_with_rank("Destroyed draft DP group")
        except Exception as e:
            print_with_rank(f"Warning: Failed to destroy draft DP group: {e}")
    
    if _DRAFT_SP_GROUP is not None:
        try:
            dist.destroy_process_group(_DRAFT_SP_GROUP)
            print_with_rank("Destroyed draft SP group")
        except Exception as e:
            print_with_rank(f"Warning: Failed to destroy draft SP group: {e}")

    # 销毁Sequence Parallelism相关进程组
    if _SP_ULYSSES_GROUP is not None:
        try:
            dist.destroy_process_group(_SP_ULYSSES_GROUP)
            print_with_rank("Destroyed SP Ulysses group")
        except Exception as e:
            print_with_rank(f"Warning: Failed to destroy SP Ulysses group: {e}")
    
    if _SP_RING_GROUP is not None:
        try:
            dist.destroy_process_group(_SP_RING_GROUP)
            print_with_rank("Destroyed SP Ring group")
        except Exception as e:
            print_with_rank(f"Warning: Failed to destroy SP Ring group: {e}")

    # 重置全局变量
    _TARGET_DEVICE_MESH = None
    _TARGET_TP_GROUP = None
    _TARGET_TP_DEVICE_MESH = None
    _TARGET_DP_GROUP = None
    _TARGET_DP_DEVICE_MESH = None
    _DRAFT_DEVICE_MESH = None
    _DRAFT_DP_GROUP = None
    _DRAFT_DP_DEVICE_MESH = None
    _DRAFT_TP_GROUP = None
    _DRAFT_TP_DEVICE_MESH = None
    _DRAFT_SP_GROUP = None
    _SP_ULYSSES_GROUP = None
    _SP_RING_GROUP = None

    # 最后销毁默认进程组
    try:
        dist.destroy_process_group()
        print_with_rank("Destroyed default process group")
    except Exception as e:
        print_with_rank(f"Warning: Failed to destroy default process group: {e}")


def shard_tensor(
    tensor: torch.Tensor, process_group: dist.ProcessGroup = None, dim: int = -1
) -> torch.Tensor:
    rank = dist.get_rank(process_group)
    size = dist.get_world_size(process_group)
    return tensor.chunk(size, dim=dim)[rank].contiguous()


def gather_tensor(
    tensor: torch.Tensor, process_group: dist.ProcessGroup = None, dim: int = -1
) -> torch.Tensor:
    size = dist.get_world_size(process_group)
    obj_list = [torch.empty_like(tensor) for _ in range(size)]
    dist.all_gather(obj_list, tensor, group=process_group)
    gathered_tensor = torch.cat(obj_list, dim=dim)
    return gathered_tensor


def all_gather_tensor(
    local_tensor: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    sp_world_size = dist.get_world_size(group=group)
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * sp_world_size
    output = torch.empty(
        output_shape, dtype=local_tensor.dtype, device=local_tensor.device
    )
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    return output


# Adapted from https://github.com/volcengine/verl/blob/a0e8e4472b8b472409defb0c8fcc5162301450af/verl/utils/ulysses.py#L194
class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_tensor: torch.Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op=False,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        sp_world_size = dist.get_world_size(group=group)
        ctx.sp_world_size = sp_world_size

        sp_rank = dist.get_rank(group=group)
        ctx.sp_rank = sp_rank

        local_shape = list(local_tensor.size())
        split_size = local_shape[0]
        part_size = local_shape[gather_dim]  # store original size
        ctx.part_size = part_size

        output = all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.sp_world_size
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[
                ctx.sp_rank
            ].contiguous(),
            None,
            None,
            None,
            None,
        )


def gather_outputs_and_unpad(
    x: torch.Tensor,
    gather_dim: int,
    grad_scaler: bool = True,
    group: Optional[dist.ProcessGroup] = None,
):
    """
    Gather a tensor across a process group and optionally unpad its padded elements.

    Args:
        x (Tensor): Input tensor to gather.
        gather_dim (int): Dimension along which to gather across ranks.
        grad_scaler (bool): Whether to apply gradient scaling during gather. Defaults to True.
        group (ProcessGroup, optional): Process group for gathering. If None, uses
            `get_ulysses_sequence_parallel_group()`. If still None, returns `x` unchanged.

    Returns:
        Tensor: The gathered tensor, with padding removed if requested.
    """
    if not group:
        group = get_draft_sp_group()
    if torch.distributed.get_world_size(group) == 1:
        return x
    x = Gather.apply(group, x, gather_dim, grad_scaler)
    return x


def is_tp_rank_0():
    """Return True if current process is rank 0 in its TP group."""
    tp_group = get_tp_group()
    if tp_group is None:
        return True
    return dist.get_rank(group=tp_group) == 0
