"""
ddp_smoke.py

学習コードを通さずに NCCL だけを試す最小テスト。
train_ecg.py が trainer.train() で固まるとき、原因が NCCL 側か
こちらのコード側かを切り分けるために使う。

    torchrun --standalone --nproc_per_node=2 ddp_smoke.py

各段階の前後にログを出すので、どこで止まるかが分かる。
30 秒で応答がない段階が原因。
"""

import datetime
import os
import socket
import sys
import time

import torch
import torch.distributed as dist


def log(msg):
    rank = os.environ.get("RANK", "?")
    print(f"[rank {rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    log(f"host={socket.gethostname()} local_rank={local_rank} world_size={world}")
    log(f"visible devices = {torch.cuda.device_count()}")
    if world < 2:
        log("WORLD_SIZE < 2。torchrun --nproc_per_node=2 で起動してください")
        return 1

    log("1) set_device")
    torch.cuda.set_device(local_rank)
    log(f"   -> {torch.cuda.get_device_name(local_rank)}")

    log("2) init_process_group (ここで止まるなら NCCL の初期化)")
    dist.init_process_group(backend="nccl",
                            timeout=datetime.timedelta(seconds=60))
    log("   -> OK")

    log("3) barrier")
    dist.barrier()
    log("   -> OK")

    log("4) all_reduce (ここで止まるなら P2P / shm)")
    x = torch.full((1024, 1024), float(rank + 1), device=local_rank)
    dist.all_reduce(x)
    expected = sum(range(1, world + 1))
    ok = abs(x[0, 0].item() - expected) < 1e-3
    log(f"   -> sum={x[0,0].item():.1f} expected={expected} ok={ok}")

    log("5) 大きめの all_reduce (100 MB)")
    t0 = time.time()
    big = torch.randn(25 * 1024 * 1024, device=local_rank)
    for _ in range(5):
        dist.all_reduce(big)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / 5
    log(f"   -> {dt*1000:.1f} ms/回, 実効 {100/dt/1024:.2f} GB/s")

    log("6) DDP でラップした 1 ステップ")
    from torch.nn.parallel import DistributedDataParallel as DDP
    m = torch.nn.Sequential(torch.nn.Linear(512, 512),
                            torch.nn.GELU(),
                            torch.nn.Linear(512, 512)).to(local_rank)
    m = DDP(m, device_ids=[local_rank])
    y = m(torch.randn(8, 512, device=local_rank)).sum()
    y.backward()
    log("   -> OK")

    dist.barrier()
    log("all checks passed")
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
