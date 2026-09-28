dataset_hyperparams = {
    "FFHQ256": {
        "patch_size": [4, 4],
        "depths": [2, 2, 4, 2],
        "widths": [64 * 3, 128 * 3, 256 * 3, 512 * 3],
        "joint": [True, True, True, False],
        "self_attns": [
            {"type": "neighborhood", "d_head": 64, "kernel_size": 5},
            {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
            {"type": "global", "d_head": 64},
            {"type": "global", "d_head": 64},
        ],
    },
    "ImageNet256": {
        "patch_size": [4, 4],
        "depths": [2, 2, 16],
        "widths": [384, 768, 1536],
        "joint": [True, True, True],
        "self_attns": [
            {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
            {"type": "neighborhood", "d_head": 64, "kernel_size": 7},
            {"type": "global", "d_head": 64},
        ],
        "dropout_rate": 0.0,
        "mapping_width": 768,
    },
}
