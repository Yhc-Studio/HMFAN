# HMFAN: Hierarchical Memory Fusion with Adaptive Range Modeling for Lightweight Image Super-Resolution

Official PyTorch implementation of **HMFAN**, a lightweight single-image super-resolution (SISR) network with hierarchical memory fusion, adaptive range modeling, and selective Fourier modulation.

HMFAN extends a lightweight multi-range attention backbone with three complementary components:

- **Hierarchical Memory Fusion (HMF)** for controlled intra-group and inter-group feature reuse.
- **RangeGate** for image-conditioned reweighting of existing multi-range spatial responses.
- **Hybrid Fourier Modulated Attention (HFMA)** for selective multi-band Fourier processing in a restricted head subspace.

The code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR) and uses [NATTEN](https://github.com/SHI-Labs/NATTEN) for neighborhood attention.

---

## Contents

- [Environment](#environment)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Pretrained Models](#pretrained-models)
- [Testing](#testing)
- [Additional Evaluations](#additional-evaluations)
- [Training](#training)
- [Efficiency and Model Complexity](#efficiency-and-model-complexity)
- [Effective Receptive Field](#effective-receptive-field)
- [Results](#results)
- [Code-to-Paper Mapping](#code-to-paper-mapping)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Environment

The manuscript experiments used:

- Linux / Ubuntu
- Python 3.10
- CUDA 12.1
- PyTorch 2.2.2
- torchvision 0.17.2
- NATTEN 0.17.x
- BasicSR 1.4.2

Other compatible PyTorch/CUDA combinations may also work.

---

## Installation

Clone the repository and create the environment:

```bash
git clone https://github.com/Yhc-Studio/HMFAN.git
cd HMFAN
conda create -n hmfan python=3.10 -y
conda activate hmfan
```

Install PyTorch:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

Install a NATTEN build compatible with your PyTorch and CUDA versions. See the official [NATTEN releases](https://github.com/SHI-Labs/NATTEN/releases).

Then install the remaining dependencies and BasicSR in development mode:

```bash
pip install -r requirements.txt
python setup.py develop
```

---

## Dataset Preparation

Training uses **DF2K**, i.e. DIV2K + Flickr2K.

Evaluation uses:

- Set5
- Set14
- BSD100 / B100
- Urban100
- Manga109

Additional configurations are included for:

- DIV2K validation set
- Test2K
- RealSRSet

Please refer to [`datasets/README.md`](datasets/README.md) and update dataset paths in the YAML files under:

```text
options/train/HMFAN/
options/test/HMFAN/
```

The default public configurations use repository-relative paths under `datasets/`.

---

## Pretrained Models

Released checkpoints are stored in:

```text
experiments/pretrained_models/HMFAN/
├── HMFAN_x2.pth
├── HMFAN_x3.pth
└── HMFAN_x4.pth
```

Each checkpoint contains both `params` and `params_ema`; testing uses `params_ema`.

---

## Testing

### Standard benchmarks

×2:

```bash
python basicsr/test.py -opt options/test/HMFAN/test_HMFAN_x2.yml
```

×3:

```bash
python basicsr/test.py -opt options/test/HMFAN/test_HMFAN_x3.yml
```

×4:

```bash
python basicsr/test.py -opt options/test/HMFAN/test_HMFAN_x4.yml
```

The reconstructed images and logs are saved under BasicSR's `results/` directory.

PSNR and SSIM are evaluated on the **Y channel** with scale-dependent border cropping.

---

## Additional Evaluations

### DIV2K validation set and Test2K

Update the dataset paths in:

```text
options/test/HMFAN/test_HMFAN_x4-Test2K&DIV2K.yml
```

Then run:

```bash
python basicsr/test.py \
    -opt "options/test/HMFAN/test_HMFAN_x4-Test2K&DIV2K.yml"
```

### RealSRSet

Update `dataroot_lq` in:

```text
options/test/HMFAN/test_HMFAN_x4-RealSRSet.yml
```

Then run:

```bash
python basicsr/test.py \
    -opt options/test/HMFAN/test_HMFAN_x4-RealSRSet.yml
```

RealSRSet is used for reference-free qualitative evaluation because aligned HR references are unavailable.

---

## Training

The public training configurations reproduce the manuscript schedule:

```text
options/train/HMFAN/
├── train_HMFAN_x2.yml
├── train_HMFAN_x3_finetune.yml
└── train_HMFAN_x4_finetune.yml
```

| Scale | Initialization | GT crop | Initial LR | Iterations |
|---|---|---:|---:|---:|
| ×2 | From scratch | 128 × 128 | 5e-4 | 500K |
| ×3 | Pretrained ×2 | 192 × 192 | 2e-4 | 250K |
| ×4 | Pretrained ×2 | 256 × 256 | 2e-4 | 250K |

The manuscript uses Adam with `beta1=0.9`, `beta2=0.99`, L1 reconstruction loss, EMA decay `0.999`, and a total batch size of 32 on four GPUs. The provided configurations therefore use `batch_size_per_gpu: 8`.

### Single-GPU training

```bash
python basicsr/train.py -opt options/train/HMFAN/train_HMFAN_x2.yml
```

For a different number of GPUs, adjust `batch_size_per_gpu` if you want to preserve the same total batch size.

### Distributed training

Example with four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/dist_train.sh 4 options/train/HMFAN/train_HMFAN_x2.yml
```

The ×3 and ×4 configurations initialize the shared body from the released ×2 checkpoint and ignore the scale-specific upsampling layer:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/dist_train.sh 4 options/train/HMFAN/train_HMFAN_x3_finetune.yml
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/dist_train.sh 4 options/train/HMFAN/train_HMFAN_x4_finetune.yml
```

Training logs and checkpoints are saved under `experiments/`.

---

## Efficiency and Model Complexity

### Latency and peak GPU memory

```bash
python basicsr/Latency.py \
    -opt options/test/HMFAN/test_HMFAN_x4.yml
```

The manuscript protocol uses:

- LR input: `320 × 180`
- Batch size: `1`
- Warm-up: `50` iterations
- Timed iterations: `200`
- Independent runs: `5`
- Data I/O excluded from timing
- CUDA synchronization around timed inference

### Parameters and FLOPs

```bash
python basicsr/calculate_params_flops_cpu.py \
    -m hmfan \
    -r 720p \
    --multi_adds \
    --device cuda \
    --scales 2,3,4
```

For a detailed FLOPs breakdown:

```bash
python basicsr/calculate_params_flops_cpu.py \
    -m hmfan \
    -r 720p \
    --multi_adds \
    --device cuda \
    --scales 2,3,4 \
    --print_table
```

Profiled operation counts are reported for an SR output resolution of approximately **1280 × 720**, using the corresponding LR input size at each scale. The provided `fvcore` profiler does not count `rFFT2/irFFT2` operators; their execution is nevertheless included in the end-to-end latency measurements.

---

## Effective Receptive Field

The ERF analysis script builds the network from a BasicSR test YAML and loads its checkpoint.

Example for ×4 HMFAN on Urban100:

```bash
python erf_from_test_yaml.py \
    -opt options/test/HMFAN/test_HMFAN_x4.yml \
    --dataset_name val_Urban100 \
    --input_size 96
```

Outputs are saved by default under:

```text
plt/erf/
```

including the accumulated ERF map (`.npy`) and rendered heatmap (`.png`).

---

## Results

### Standard benchmark results

PSNR / SSIM on the Y channel.

| Scale | Params | FLOPs | Set5 | Set14 | BSD100 | Urban100 | Manga109 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ×2 | 824K | 213.6G | 38.31 / 0.9617 | 34.23 / 0.9233 | 32.45 / 0.9032 | 33.32 / 0.9389 | 39.66 / 0.9792 |
| ×3 | 832K | 95.9G | 34.80 / 0.9303 | 30.72 / 0.8497 | 29.36 / 0.8121 | 29.21 / 0.8725 | 34.73 / 0.9514 |
| ×4 | 844K | 54.5G | 32.71 / 0.9012 | 29.00 / 0.7908 | 27.82 / 0.7452 | 26.98 / 0.8120 | 31.58 / 0.9210 |

For ×4 SR, HMFAN improves MAT-Light by **0.15 dB on Urban100** and **0.20 dB on Manga109**.

### Additional ×4 evaluation

| Method | DIV2K | Test2K |
|---|---:|---:|
| HMFAN | 30.93 / 0.8484 | 27.86 / 0.7842 |

### Practical ×4 inference efficiency

Under the manuscript's unified benchmark protocol on a single NVIDIA GeForce RTX 4090 D:

| Model | Latency | Peak GPU Memory | Average PSNR / SSIM |
|---|---:|---:|---:|
| HMFAN | 38.945 ms/image | 303.6 MB | 29.62 / 0.8340 |

The average reconstruction score is computed over Set5, Set14, BSD100, Urban100, and Manga109.

---

## Code-to-Paper Mapping

Several internal option names are retained for checkpoint compatibility. Their correspondence to the manuscript is:

| Paper terminology | Public implementation |
|---|---|
| HMFAN | `HMFAN` alias in `basicsr/archs/mat_arch.py` |
| Intra-group Memory Fusion (Intra-MF) | `BDRLite`, `use_bdr`, `memory_indices` |
| Inter-group Memory Fusion (Inter-MF) | `InterGroupStaticFusion`, `use_cgmf`, `cgmf_num_prev` |
| RangeGate | `use_range_gate`, `gate_mode` in `mat_arch.py` |
| Hybrid Fourier Modulated Attention (HFMA) | `HybridMultiBandFMA` in `basicsr/archs/hfma.py` |

The canonical released configuration uses:

```yaml
num_feat: 60
num_block: 4
depth: 4
num_head: 6
kernel_sizes: [7, 9, 11]
dilations: [[1, 1, 1], [9, 7, 5]]
dw_sizes: [1, 3, 5, 7]

use_bdr: true
memory_indices: [0, 1, 2]
routing_reduction: 8
routing_type: static

use_cgmf: true
cgmf_num_prev: 2
cgmf_scale: 1.0

use_range_gate: true
gate_mode: softmax

num_heads: 6
fourier_heads: 2
num_bands: 3
```

The train/test YAML files under `options/` are the canonical reproduction settings for the released checkpoints. `HMFAN_REPRODUCIBILITY_CONFIG.yaml` provides the same core settings in a compact form.

---

## Citation

If you find this repository useful, please consider citing HMFAN:

```bibtex
@misc{liu2026hmfan,
  title  = {Hierarchical Memory Fusion with Adaptive Range Modeling for Lightweight Image Super-Resolution},
  author = {Liu, Wei and Ye, Jianting and Chen, Qing and Sun, Yiming},
  year   = {2026},
  note   = {Manuscript under review}
}
```

The citation will be updated after publication.

---

## Acknowledgements

This repository is built upon [BasicSR](https://github.com/XPixelGroup/BasicSR). We also acknowledge the public implementations and research contributions of MAT and NATTEN.

---

## License

This repository is distributed under the Apache License 2.0. Portions derived from third-party projects remain subject to their respective licenses. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
