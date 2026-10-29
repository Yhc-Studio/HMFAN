import warnings

from basicsr.archs.mat_arch import HMFAN

warnings.filterwarnings("ignore", category=UserWarning)


def build_models(model_name, resolution='720p', upscale=4):
    """Build HMFAN for the model-complexity utility.

    Args:
        model_name (str): ``hmfan``. ``mat-light`` is kept as a legacy alias.
        resolution (str): Resolution tag used only to return the nominal LR size.
        upscale (int): Super-resolution scale.

    Returns:
        tuple: ``(model, (lr_height, lr_width))``.
    """
    resolution_map = {
        '540p': (540, 960),
        '720p': (720, 1280),
        '1080p': (1080, 1920),
        '2k': (1440, 2560),
        '4k': (2160, 3840),
    }

    resolution_key = resolution.lower()
    if resolution_key in resolution_map:
        hr_height, hr_width = resolution_map[resolution_key]
    elif resolution.isdigit():
        hr_height = hr_width = int(resolution)
    else:
        raise ValueError(f'Unsupported resolution: {resolution}')

    if model_name.lower() not in ('hmfan', 'mat-light'):
        raise ValueError(
            f'Unsupported model: {model_name}. Use "hmfan" '
            '(or the legacy alias "mat-light").'
        )

    lr_height, lr_width = hr_height // upscale, hr_width // upscale

    model = HMFAN(
        num_feat=60,
        num_block=4,
        expanded_ratio=1,
        squeeze_factor=4,
        depth=4,
        num_head=6,
        kernel_sizes=[7, 9, 11],
        dilations=[[1, 1, 1], [9, 7, 5]],
        rel_pos_bias=True,
        mlp_ratio=2,
        dw_sizes=[1, 3, 5, 7],
        use_bdr=True,
        memory_indices=[0, 1, 2],
        routing_reduction=8,
        routing_type='static',
        use_cgmf=True,
        cgmf_num_prev=2,
        cgmf_scale=1.0,
        use_range_gate=True,
        gate_mode='softmax',
        gate_hidden=None,
        num_heads=6,
        fourier_heads=2,
        num_bands=3,
        upscale=upscale,
        upsampler='pixelshuffledirect',
    )

    return model, (lr_height, lr_width)
