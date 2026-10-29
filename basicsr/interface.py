import logging
import os
from os import path as osp
import sys
import statistics
import time

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def benchmark_inference_time(
        model,
        logger,
        input_size=(1, 3, 180, 320),   # (B, C, H, W), LR input size
        warmup=50,
        repeat=200):
    """
    Benchmark pure forward inference time with a fixed random input.

    Args:
        model: BasicSR model object.
        logger: logger.
        input_size: tuple, input tensor shape (B, C, H, W).
        warmup: int, number of warm-up iterations.
        repeat: int, number of timed iterations.

    Returns:
        avg_time: float, average latency in ms/image.
        fps: float, FPS computed from avg_time.
        peak_memory: float, peak GPU memory in MB.
    """
    model.net_g.eval()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Fixed random input
    x = torch.randn(*input_size, device=device)

    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # Clear the cache first to avoid any residual impact statistics
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    times = []

    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = model.net_g(x)

        if use_cuda:
            # Restart the peak memory count after warming up
            torch.cuda.reset_peak_memory_stats()

        # timed runs
        for _ in range(repeat):
            if use_cuda:
                starter.record()
                _ = model.net_g(x)
                ender.record()
                torch.cuda.synchronize()
                elapsed = starter.elapsed_time(ender)  # ms
            else:
                start_time = time.time()
                _ = model.net_g(x)
                end_time = time.time()
                elapsed = (end_time - start_time) * 1000.0

            times.append(elapsed)

    avg_time = sum(times) / len(times)
    fps = 1000.0 / avg_time if avg_time > 0 else 0.0

    if use_cuda:
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    else:
        peak_memory = 0.0

    logger.info(
        f'Run result -> {avg_time:.3f} ms/image | '
        f'FPS: {fps:.3f} | '
        f'Peak Memory: {peak_memory:.1f} MB | '
        f'(LR input: {input_size[2]}x{input_size[3]}, batch={input_size[0]}, '
        f'warmup={warmup}, repeat={repeat})'
    )

    return avg_time, fps, peak_memory


def benchmark_repeat(
        model,
        logger,
        runs=3,
        input_size=(1, 3, 180, 320),
        warmup=50,
        repeat=200):
    """
    Run benchmark multiple times and report mean/std latency, mean FPS, and mean peak memory.

    Args:
        model: BasicSR model object.
        logger: logger.
        runs: int, number of independent benchmark runs.
        input_size: tuple, input tensor shape (B, C, H, W).
        warmup: int, number of warm-up iterations per run.
        repeat: int, number of timed iterations per run.

    Returns:
        mean_time: float, mean latency in ms/image.
        std_time: float, std latency in ms/image.
        mean_fps: float, mean FPS across runs.
        mean_memory: float, mean peak memory in MB across runs.
    """
    times = []
    fps_list = []
    memory_list = []

    logger.info('========================================')
    logger.info(f'Benchmarking fixed-input inference time for {runs} runs...')
    logger.info('========================================')

    for i in range(runs):
        logger.info(f'Run {i + 1}/{runs}...')

        avg_time, fps, peak_memory = benchmark_inference_time(
            model=model,
            logger=logger,
            input_size=input_size,
            warmup=warmup,
            repeat=repeat
        )

        times.append(avg_time)
        fps_list.append(fps)
        memory_list.append(peak_memory)

    mean_time = sum(times) / len(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0
    mean_fps = sum(fps_list) / len(fps_list)
    mean_memory = sum(memory_list) / len(memory_list)

    logger.info('========================================')
    logger.info('Final Benchmark Result')
    logger.info('========================================')
    logger.info(f'All latency runs (ms/image): {[round(t, 3) for t in times]}')
    logger.info(f'All FPS runs: {[round(f, 3) for f in fps_list]}')
    logger.info(f'All peak memory runs (MB): {[round(m, 1) for m in memory_list]}')
    logger.info(
        f'Latency: {mean_time:.3f} ± {std_time:.3f} ms/image | '
        f'FPS: {mean_fps:.3f} | '
        f'Peak Memory: {mean_memory:.1f} MB'
    )

    return mean_time, std_time, mean_fps, mean_memory


def main(root_path):
    # parse options
    opt, _ = parse_options(root_path, is_train=False)
    torch.backends.cudnn.benchmark = True

    # mkdir and logger
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"latency_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr',
        log_level=logging.INFO,
        log_file=log_file
    )

    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create model
    model = build_model(opt)

    # benchmark
    mean_time, std_time, mean_fps, mean_memory = benchmark_repeat(
        model=model,
        logger=logger,
        runs=5,                       
        input_size=(1, 3, 180, 320),  # x4 -> approx 1280x720 output
        warmup=50,
        repeat=200
    )

    logger.info(
        f'Final Summary -> {mean_time:.3f} ± {std_time:.3f} ms/image, '
        f'FPS: {mean_fps:.3f}, Peak Memory: {mean_memory:.1f} MB'
    )


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    main(root_path)