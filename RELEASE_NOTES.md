# HMFAN public repository release

This package is the cleaned public HMFAN/BasicSR repository intended for reproducible training, testing, and efficiency evaluation.

Key public-release points:
- HMFAN architecture registered as `HMFAN` in `basicsr/archs/mat_arch.py`.
- HFMA implementation is in `basicsr/archs/hfma.py`.
- Canonical train/test settings are under `options/train/HMFAN/` and `options/test/HMFAN/`.
- Released x2/x3/x4 checkpoints are under `experiments/pretrained_models/HMFAN/`.
- Local dataset paths and Python cache files are excluded.
- The compact `HMFAN_REPRODUCIBILITY_CONFIG.yaml` mirrors the core released settings; full YAMLs under `options/` remain authoritative.
