#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import warnings

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from natten.flops import add_natten_handle

from basicsr.build_models import build_models

warnings.filterwarnings("ignore", category=UserWarning)


def pick_device(device_str: str) -> torch.device:
    s = device_str.lower()
    if s == "cpu":
        return torch.device("cpu")
    if s == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but --device cuda was requested.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_threads(n: int) -> None:
    try:
        torch.set_num_threads(max(1, int(n)))
        torch.set_num_interop_threads(min(4, max(1, int(n))))
    except Exception:
        pass


def fmt_params(num_params: int) -> str:
    return f"{num_params / 1e6:.4f} M"


def fmt_flops(flops: float) -> str:
    return f"{flops / 1e9:.3f} G"


def calc_lr_from_hr(hr_w: int, hr_h: int, scale: int):
    """ceil LR so that upsampled HR >= target HR, then crop to exact."""
    lr_w = int(math.ceil(hr_w / scale))
    lr_h = int(math.ceil(hr_h / scale))
    up_hr_w = lr_w * scale
    up_hr_h = lr_h * scale
    crop_w = up_hr_w - hr_w
    crop_h = up_hr_h - hr_h
    return lr_w, lr_h, up_hr_w, up_hr_h, crop_w, crop_h


@torch.no_grad()
def run_one_scale(args, scale: int):
    # build model
    model, _ = build_models(args.model_name, args.resolution, scale)

    # net name
    net_cls_str = f"{model.__class__.__name__}"
    if "-" in args.model_name:
        net_cls_str += "-" + args.model_name.split("-")[1]

    device = pick_device(args.device)
    model = model.to(device).eval()

    # params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print params-only
    if not args.multi_adds:
        print(f"x{scale}: Network={net_cls_str}, Params={fmt_params(params)}")
        return

    # FLOPs input derived from target HR
    lr_w, lr_h, up_hr_w, up_hr_h, crop_w, crop_h = calc_lr_from_hr(args.hr_w, args.hr_h, scale)
    inp = torch.randn(args.batch, 3, lr_h, lr_w, device=device)

    flop_ctr = FlopCountAnalysis(model, inp)

    # natten handle for MAT-like models
    name_l = args.model_name.lower()
    if "hmfan" in name_l or "mat" in name_l or "mrat" in name_l:
        flop_ctr = add_natten_handle(flop_ctr)

    # suppress unsupported warnings
    flop_ctr = flop_ctr.unsupported_ops_warnings(False)

    # ✅ module/op breakdown table (same as your original)
    if args.print_table:
        print(f"\n========== FLOPs breakdown table (x{scale}) ==========")
        print(f"[Input]  LR {lr_w}x{lr_h}  ->  HR {up_hr_w}x{up_hr_h}"
              + (f"  (crop {crop_w}x{crop_h} -> {args.hr_w}x{args.hr_h})" if (crop_w or crop_h) else ""))
        print(flop_count_table(flop_ctr))

    total_flops = float(flop_ctr.total())  # fvcore convention: report as FLOPs
    crop_note = ""
    if crop_w != 0 or crop_h != 0:
        crop_note = f", crop {crop_w}x{crop_h} -> {args.hr_w}x{args.hr_h}"

    print(
        f"x{scale}: Network={net_cls_str}, Params={fmt_params(params)}, "
        f"FLOPs={fmt_flops(total_flops)} "
        f"(LR {lr_w}x{lr_h} -> HR {up_hr_w}x{up_hr_h}{crop_note})"
    )


def main(args):
    device = pick_device(args.device)
    if device.type == "cpu":
        set_threads(args.cpu_threads)

    scales = [int(s.strip()) for s in args.scales.split(",") if s.strip()]
    if not scales:
        raise ValueError("Empty --scales. Example: --scales 2,3,4")

    print(f"[Info] Model={args.model_name} | resolution tag={args.resolution}")
    print(f"[Info] Device={device} | batch={args.batch}")
    print(f"[Info] Target HR for FLOPs reporting: {args.hr_w}x{args.hr_h}")
    print(f"[Info] Scales={scales}")
    print(f"[Info] FLOPs convention: using fvcore total() directly as FLOPs (no x2).")
    print("")

    for s in scales:
        run_one_scale(args, s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Params + FLOPs (fvcore) on HR=1280x720 for x2/x3/x4, with breakdown table.")
    parser.add_argument("-m", "--model_name", type=str, default="hmfan")
    parser.add_argument("-r", "--resolution", type=str, default="720p")

    parser.add_argument("--scales", type=str, default="2,3,4")
    parser.add_argument("--hr_w", type=int, default=1280)
    parser.add_argument("--hr_h", type=int, default=720)

    parser.add_argument("--multi_adds", action="store_true", help="calculate FLOPs via fvcore (runs forward)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--batch", type=int, default=1)

    # ✅ print breakdown table (like your original code)
    parser.add_argument("--print_table", action="store_true",
                        help="print flop_count_table (module/op breakdown) for each scale")

    parser.add_argument("--cpu_threads", type=int, default=int(os.getenv("OMP_NUM_THREADS", "16")))
    args = parser.parse_args()
    main(args)

# Examples:
# python basicsr/calculate_params_flops_cpu.py -m hmfan -r 720p --multi_adds --device cuda --scales 2,3,4 --print_table
# python basicsr/calculate_params_flops_cpu.py -m hmfan -r 720p --multi_adds --device cuda --scales 2,3,4