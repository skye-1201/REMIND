import argparse
import json
import os
import random
import time

import numpy as np
import torch

from config import cfg
from data import make_dataloader
from engine.processor import do_inference
from modeling import make_model
from modeling.MemoryBank.MemoryBank import MemoryBank
from utils.logger import setup_logger


MEMORY_KEYS = ['rgb', 'nir', 'tir', 'sr', 'sn', 'st']


class ForwardLatencyMeter:
    """
    Measure model.forward latency only.

    - warmup_calls: number of initial forward calls ignored from statistics.
    - max_record_calls: optional cap to reduce bookkeeping; None means record all.
    """

    def __init__(self, device, warmup_calls=10, max_record_calls=None):
        self.device = device
        self.warmup_calls = max(0, int(warmup_calls))
        self.max_record_calls = None if max_record_calls is None or int(max_record_calls) <= 0 else int(max_record_calls)
        self.forward_times = []
        self.forward_batch_sizes = []
        self._num_calls = 0
        self._orig_forward = None

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _infer_batch_size(first_arg):
        if isinstance(first_arg, dict):
            for v in first_arg.values():
                if torch.is_tensor(v):
                    return int(v.size(0))
        if torch.is_tensor(first_arg):
            return int(first_arg.size(0))
        return 1

    def attach(self, model):
        self._orig_forward = model.forward
        meter = self

        def wrapped_forward(*args, **kwargs):
            meter._sync()
            start = time.perf_counter()
            out = meter._orig_forward(*args, **kwargs)
            meter._sync()
            elapsed = time.perf_counter() - start

            meter._num_calls += 1
            if meter._num_calls > meter.warmup_calls:
                if meter.max_record_calls is None or len(meter.forward_times) < meter.max_record_calls:
                    batch_size = meter._infer_batch_size(args[0]) if len(args) > 0 else 1
                    meter.forward_times.append(elapsed)
                    meter.forward_batch_sizes.append(batch_size)
            return out

        model.forward = wrapped_forward

    def detach(self, model):
        if self._orig_forward is not None:
            model.forward = self._orig_forward
            self._orig_forward = None

    def summary(self):
        if len(self.forward_times) == 0:
            return {
                'num_recorded_calls': 0,
                'num_recorded_samples': 0,
                'mean_batch_latency_ms': None,
                'mean_sample_latency_ms': None,
                'p50_batch_latency_ms': None,
                'p90_batch_latency_ms': None,
                'forward_fps': None,
            }

        times = np.array(self.forward_times, dtype=np.float64)
        batch_sizes = np.array(self.forward_batch_sizes, dtype=np.float64)
        total_samples = int(batch_sizes.sum())
        total_time = float(times.sum())
        per_sample_latency_ms = (total_time / max(total_samples, 1)) * 1000.0
        return {
            'num_recorded_calls': int(len(self.forward_times)),
            'num_recorded_samples': total_samples,
            'mean_batch_latency_ms': float(times.mean() * 1000.0),
            'mean_sample_latency_ms': float(per_sample_latency_ms),
            'p50_batch_latency_ms': float(np.percentile(times, 50) * 1000.0),
            'p90_batch_latency_ms': float(np.percentile(times, 90) * 1000.0),
            'forward_fps': float(total_samples / max(total_time, 1e-12)),
        }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _memory_to_tensor(obj, device):
    if isinstance(obj, MemoryBank):
        return obj.features.detach().to(device)
    if torch.is_tensor(obj):
        if obj.dim() != 2:
            raise ValueError(f"Expected memory tensor [N, C], got {tuple(obj.shape)}")
        return obj.detach().to(device)
    if hasattr(obj, 'features') and torch.is_tensor(obj.features):
        return obj.features.detach().to(device)
    raise TypeError(f"Unsupported memory object type: {type(obj)}")


def build_eval_memories(memory_data, device, memory_n=-1, subset_mode='first', seed=1234):
    base = _memory_to_tensor(memory_data[MEMORY_KEYS[0]], device)
    total_n = int(base.size(0))

    if memory_n is None or int(memory_n) <= 0:
        effective_n = total_n
    else:
        effective_n = min(int(memory_n), total_n)

    if subset_mode == 'first':
        indices = torch.arange(effective_n, device=device, dtype=torch.long)
    elif subset_mode == 'random':
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        indices = torch.randperm(total_n, generator=g, device=device)[:effective_n]
    else:
        raise ValueError(f"Unsupported subset mode: {subset_mode}")

    bank_list = []
    tensor_list = []
    for key in MEMORY_KEYS:
        feats = _memory_to_tensor(memory_data[key], device)
        if feats.size(0) != total_n:
            raise ValueError(f"Memory size mismatch for key={key}: {feats.size(0)} vs {total_n}")
        selected = feats.index_select(0, indices).contiguous()
        tensor_list.append(selected)
        bank_list.append(MemoryBank.from_tensor(selected, device=device))

    meta = {
        'total_memory_size': total_n,
        'effective_memory_size': effective_n,
        'subset_mode': subset_mode,
        'seed': int(seed),
        'selected_indices_preview': indices[: min(20, indices.numel())].detach().cpu().tolist(),
    }
    return bank_list, tensor_list, meta


def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    param_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'non_trainable_params': int(non_trainable_params),
        'param_size_mb': float(param_size_bytes / (1024 ** 2)),
    }


def count_memory_storage_mb(memory_tensors):
    total_bytes = 0
    for t in memory_tensors:
        if torch.is_tensor(t):
            total_bytes += t.numel() * t.element_size()
    return float(total_bytes / (1024 ** 2))


def get_eval_sample_count(val_loader):
    try:
        return len(val_loader.dataset)
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REMIND Testing with configurable memory size and latency")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument(
        "--weight_path",
        type=str,
        default=None,
        help="path to trained model weights; default is TEST.WEIGHT_PATH in yml",
    )
    parser.add_argument(
        "--memory_path",
        type=str,
        default=None,
        help="path to saved memory bank file; default is TEST.MEMORY_PATH in yml",
    )
    parser.add_argument("--memory_n", type=int, default=None, help="test-time memory size N; default is TEST.MEMORY_N in yml")
    parser.add_argument("--memory_subset", type=str, default=None, choices=["first", "random"], help="default is TEST.MEMORY_SUBSET in yml")
    parser.add_argument("--seed", type=int, default=None, help="default is TEST.MEMORY_SEED in yml")
    parser.add_argument("--latency_warmup_calls", type=int, default=None, help="default is TEST.LATENCY_WARMUP_CALLS in yml")
    parser.add_argument("--latency_max_record_calls", type=int, default=None, help="default is TEST.LATENCY_MAX_RECORD_CALLS in yml")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    weight_path = args.weight_path or cfg.TEST.WEIGHT_PATH
    memory_path = args.memory_path or cfg.TEST.MEMORY_PATH
    memory_n = cfg.TEST.MEMORY_N if args.memory_n is None else args.memory_n
    memory_subset = cfg.TEST.MEMORY_SUBSET if args.memory_subset is None else args.memory_subset
    memory_seed = cfg.TEST.MEMORY_SEED if args.seed is None else args.seed
    latency_warmup_calls = cfg.TEST.LATENCY_WARMUP_CALLS if args.latency_warmup_calls is None else args.latency_warmup_calls
    latency_max_record_calls = cfg.TEST.LATENCY_MAX_RECORD_CALLS if args.latency_max_record_calls is None else args.latency_max_record_calls
    set_seed(memory_seed)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("REMIND", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    logger.info(f"Loading memory from: {memory_path}")
    memory_data = torch.load(memory_path, map_location=device)

    eval_memories, eval_memory_tensors, memory_meta = build_eval_memories(
        memory_data=memory_data,
        device=device,
        memory_n=memory_n,
        subset_mode=memory_subset,
        seed=memory_seed,
    )
    memory_storage_mb = count_memory_storage_mb(eval_memory_tensors)
    memory_meta['storage_mb'] = memory_storage_mb
    logger.info(f"Memory setup: {memory_meta}")

    model = make_model(
        cfg,
        num_class=num_classes,
        camera_num=camera_num,
        view_num=view_num,
        memory=eval_memories,
    )
    model.eval()
    model.load_param(weight_path)

    param_summary = count_model_parameters(model)

    meter = ForwardLatencyMeter(
        device=device,
        warmup_calls=latency_warmup_calls,
        max_record_calls=latency_max_record_calls,
    )
    meter.attach(model)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    wall_start = time.perf_counter()
    do_inference(
        cfg,
        model,
        val_loader,
        num_query,
        *eval_memories,
        return_pattern=1,
    )
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    wall_time = time.perf_counter() - wall_start

    meter.detach(model)
    latency_summary = meter.summary()

    eval_samples = get_eval_sample_count(val_loader)
    if eval_samples is not None:
        end_to_end_latency_ms = wall_time / max(eval_samples, 1) * 1000.0
        end_to_end_fps = eval_samples / max(wall_time, 1e-12)
    else:
        end_to_end_latency_ms = None
        end_to_end_fps = None

    peak_mem_mb = None
    if device.type == 'cuda':
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    summary = {
        'memory': memory_meta,
        'params': param_summary,
        'latency': {
            'forward_only': latency_summary,
            'end_to_end': {
                'wall_time_s': float(wall_time),
                'num_eval_samples': int(eval_samples) if eval_samples is not None else None,
                'mean_sample_latency_ms': float(end_to_end_latency_ms) if end_to_end_latency_ms is not None else None,
                'fps': float(end_to_end_fps) if end_to_end_fps is not None else None,
            },
            'peak_gpu_memory_mb': peak_mem_mb,
        },
    }

    logger.info("=" * 60)
    logger.info(f"Effective memory size N: {memory_meta['effective_memory_size']} / {memory_meta['total_memory_size']}")
    logger.info(f"Memory storage: {memory_meta['storage_mb']:.2f} MB")
    logger.info(
        "Params: "
        f"total={param_summary['total_params']:,}, "
        f"trainable={param_summary['trainable_params']:,}, "
        f"non-trainable={param_summary['non_trainable_params']:,}, "
        f"size={param_summary['param_size_mb']:.2f} MB"
    )
    if latency_summary['mean_sample_latency_ms'] is not None:
        logger.info(
            "Forward-only latency: "
            f"{latency_summary['mean_sample_latency_ms']:.3f} ms/sample, "
            f"FPS={latency_summary['forward_fps']:.3f}"
        )
        logger.info(
            "Forward-only batch latency: "
            f"mean={latency_summary['mean_batch_latency_ms']:.3f} ms, "
            f"p50={latency_summary['p50_batch_latency_ms']:.3f} ms, "
            f"p90={latency_summary['p90_batch_latency_ms']:.3f} ms"
        )
    if end_to_end_latency_ms is not None:
        logger.info(
            f"End-to-end latency: {end_to_end_latency_ms:.3f} ms/sample, FPS={end_to_end_fps:.3f}, wall_time={wall_time:.3f}s"
        )
    if peak_mem_mb is not None:
        logger.info(f"Peak GPU memory: {peak_mem_mb:.2f} MB")

    save_path = os.path.join(
        output_dir,
        f"test_memoryN_{memory_meta['effective_memory_size']}_latency.json",
    )
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved latency summary to {save_path}")
