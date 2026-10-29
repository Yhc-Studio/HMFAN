import os
import argparse
import random

import cv2
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from basicsr.archs import build_network


def load_bsr_opt(opt_path):
    """
    Prefer BasicSR's option parser and fall back to ``yaml.safe_load``
    when the local BasicSR version is incompatible.
    """
    try:
        from basicsr.utils.options import parse
        opt = parse(opt_path, is_train=False)
        print("Loaded yaml by BasicSR options.parse")
    except Exception as e:
        print("BasicSR options.parse failed, fallback to yaml.safe_load")
        print("Reason:", repr(e))
        with open(opt_path, "r", encoding="utf-8") as f:
            opt = yaml.safe_load(f)

    return opt


def set_random_seed(seed):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataset(data_dir, img_ext=None):
    if img_ext is None:
        img_ext = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    paths = []

    for imgname in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(imgname)[-1].lower()
        if ext in img_ext:
            paths.append(os.path.join(data_dir, imgname))

    return paths


def img_read(path):
    print("read:", path)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert img is not None, f"Cannot read image: {path}"

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = torch.from_numpy(np.ascontiguousarray(img))
    img = img.permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    return img


def select_dataset(opt, dataset_key=None, dataset_name=None):
    """
    Select a dataset from a BasicSR test YAML either by its key
    (for example ``test_1``) or by its configured dataset name
    (for example ``Urban100``).
    """

    datasets = opt.get("datasets", None)
    assert datasets is not None, "No datasets field found in yaml."

    if dataset_key is not None:
        assert dataset_key in datasets, f"{dataset_key} not found in datasets. Available: {list(datasets.keys())}"
        return dataset_key, datasets[dataset_key]

    if dataset_name is not None:
        for key, dataset_opt in datasets.items():
            if dataset_opt.get("name", None) == dataset_name:
                return key, dataset_opt

        raise ValueError(
            f"Dataset name {dataset_name} not found. "
            f"Available names: {[v.get('name', None) for v in datasets.values()]}"
        )

    # Default to the first configured test dataset.
    first_key = list(datasets.keys())[0]
    return first_key, datasets[first_key]


def get_lq_root_from_dataset(dataset_opt):
    """
    Prefer ``dataroot_lq`` and fall back to ``dataroot_gt`` when needed.
    """

    candidate_keys = [
        "dataroot_lq",
        "dataroot_lq1",
        "dataroot_lq2",
        "dataroot_gt"
    ]

    for key in candidate_keys:
        if key in dataset_opt and dataset_opt[key] is not None:
            return dataset_opt[key]

    raise ValueError(
        "Cannot find dataroot_lq or dataroot_gt in selected dataset. "
        f"Dataset keys: {list(dataset_opt.keys())}"
    )


def load_pretrained(model, opt, device):
    path_opt = opt.get("path", {})
    weight_path = path_opt.get("pretrain_network_g", None)

    assert weight_path is not None, "path.pretrain_network_g is missing in yaml."
    assert os.path.isfile(weight_path), f"Checkpoint not found: {weight_path}"

    strict = path_opt.get("strict_load_g", True)
    param_key = path_opt.get("param_key_g", None)

    checkpoint = torch.load(weight_path, map_location=device)

    if isinstance(checkpoint, dict):
        if param_key is not None and param_key in checkpoint:
            state_dict = checkpoint[param_key]
            print(f"Load checkpoint key from yaml: {param_key}")
        elif "params_ema" in checkpoint:
            state_dict = checkpoint["params_ema"]
            print("Load checkpoint key: params_ema")
        elif "params" in checkpoint:
            state_dict = checkpoint["params"]
            print("Load checkpoint key: params")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print("Load checkpoint key: state_dict")
        else:
            state_dict = checkpoint
            print("Load checkpoint as raw state_dict")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded pretrained model from: {weight_path}")
    print(f"strict_load_g: {strict}")

    return model


def get_input_grad(model, samples, grad_postprocess="relu"):
    outputs = model(samples)

    out_h, out_w = outputs.size(2), outputs.size(3)

    central_point = outputs[:, :, out_h // 2, out_w // 2].sum()

    grad = torch.autograd.grad(
        outputs=central_point,
        inputs=samples,
        retain_graph=False,
        create_graph=False
    )[0]

    if grad_postprocess == "relu":
        grad = torch.relu(grad)
    elif grad_postprocess == "abs":
        grad = torch.abs(grad)
    elif grad_postprocess == "raw":
        pass
    else:
        raise ValueError(f"Unsupported grad_postprocess: {grad_postprocess}")

    grad_map = grad.sum(dim=(0, 1))
    grad_map = grad_map.detach().cpu().numpy()

    return grad_map


def save_erf_heatmap(data, save_path, cmap="RdYlGn", dpi=400, show_colorbar=True):
    plt.figure(figsize=(10, 10.75), dpi=40)
    ax = plt.gca()

    im = ax.imshow(data, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])

    if show_colorbar:
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("top", size="5%", pad="2%")
        plt.colorbar(im, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("top")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    print("save at:", save_path)


def run_erf(
    model,
    device,
    data_dir,
    save_dir,
    save_name,
    input_size=96,
    interpolation="nearest",
    grad_postprocess="relu",
    log_scale=True,
    normalize=True,
    cmap="RdYlGn",
    dpi=400
):
    h, w = input_size, input_size

    dataset = get_dataset(data_dir)
    assert len(dataset) > 0, f"No images found in: {data_dir}"

    meter = np.zeros((h, w), dtype=np.float64)

    model.eval()

    for i, img_path in enumerate(dataset):
        img_lr = img_read(img_path)

        img_lr = F.interpolate(
            img_lr,
            size=(h, w),
            mode=interpolation
        )

        img_lr = img_lr.to(device)
        img_lr.requires_grad_(True)

        model.zero_grad(set_to_none=True)

        contribution_scores = get_input_grad(
            model=model,
            samples=img_lr,
            grad_postprocess=grad_postprocess
        )

        if np.isnan(np.sum(contribution_scores)):
            print("got NAN, next image")
            continue

        meter += contribution_scores

        print(f"{i + 1}/{len(dataset)} meter.shape is {meter.shape}")

    # Save the accumulated raw ERF map before log scaling or normalization.
    raw_save_path = os.path.join(save_dir, f"{save_name}_erf_raw.npy")
    np.save(raw_save_path, meter)
    print("raw ERF saved at:", raw_save_path)

    data = meter.copy()

    if log_scale:
        data = np.log10(data + 1)

    # Normalize only for visualization when requested.
    if normalize:
        max_value = np.max(data)
        if max_value > 0:
            data = data / max_value

    save_path = os.path.join(save_dir, f"{save_name}_erf.png")

    save_erf_heatmap(
        data=data,
        save_path=save_path,
        cmap=cmap,
        dpi=dpi,
        show_colorbar=True
    )


def main():
    parser = argparse.ArgumentParser("ERF from BasicSR test yaml")

    parser.add_argument(
        "-opt",
        type=str,
        required=True,
        help="Path to BasicSR test yaml."
    )

    parser.add_argument(
        "--dataset_key",
        type=str,
        default=None,
        help="Dataset key in yaml, e.g. test_1, test_2, test_3."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name in yaml, e.g. Urban100, Manga109."
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=96,
        help="ERF input size. Default: 96."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="plt/erf",
        help="Save directory for ERF figure."
    )

    parser.add_argument(
        "--grad_postprocess",
        type=str,
        default="relu",
        choices=["relu", "abs", "raw"],
        help="Postprocess gradient map."
    )

    parser.add_argument(
        "--interpolation",
        type=str,
        default="nearest",
        choices=["nearest", "bilinear", "bicubic"],
        help="Resize mode for LR input."
    )

    parser.add_argument(
        "--cmap",
        type=str,
        default="RdYlGn",
        help="Matplotlib colormap."
    )

    args = parser.parse_args()

    opt = load_bsr_opt(args.opt)

    seed = opt.get("manual_seed", 0)
    set_random_seed(seed)

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Select a dataset from the test YAML.
    dataset_key, dataset_opt = select_dataset(
        opt,
        dataset_key=args.dataset_key,
        dataset_name=args.dataset_name
    )

    dataset_name = dataset_opt.get("name", dataset_key)
    data_dir = get_lq_root_from_dataset(dataset_opt)

    print(f"Selected dataset key: {dataset_key}")
    print(f"Selected dataset name: {dataset_name}")
    print(f"ERF data dir: {data_dir}")

    # 2. Build the network from ``network_g``.
    model = build_network(opt["network_g"])
    model = model.to(device)

    # 3. Load the checkpoint from ``path.pretrain_network_g``.
    model = load_pretrained(model, opt, device)

    # 4. Build a descriptive output name.
    model_name = opt.get("name", model.__class__.__name__)
    save_name = f"{model_name}_{dataset_name}_{args.input_size}"

    run_erf(
        model=model,
        device=device,
        data_dir=data_dir,
        save_dir=args.save_dir,
        save_name=save_name,
        input_size=args.input_size,
        interpolation=args.interpolation,
        grad_postprocess=args.grad_postprocess,
        cmap=args.cmap
    )


if __name__ == "__main__":
    main()