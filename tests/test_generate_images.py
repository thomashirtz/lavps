from pathlib import Path

import pytest
import torch
from hydra import compose, initialize

from lavps.paths import CONFIG_DIRECTORY
from scripts.generate_images import main

OPERATORS_PATH = CONFIG_DIRECTORY / "inverse_problem"

dataset_files = ["ffhq_val", "imagenet_val"]

operator_files = [
    p.stem for p in Path(OPERATORS_PATH).glob("*.yaml")
    if not p.stem.startswith("_")
]

sampler_files = ["mgdm", "dps"]
# sampler_files = ["mgdm", "dps", "lavps"]  # Adding LAVPS, which requires checkpoints for the amortized models


@pytest.mark.parametrize("dataset_name", dataset_files, ids=dataset_files)
@pytest.mark.parametrize("operator_name", operator_files, ids=operator_files)
@pytest.mark.parametrize("sampler_name", sampler_files, ids=sampler_files)
def test_generate_images_does_not_smoke(dataset_name, operator_name, sampler_name):
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="generate_images",
            overrides=[
                f"dataset={dataset_name}",
                f"inverse_problem={operator_name}",
                f"sampler={sampler_name}",
                f"model=hdit_{dataset_name.split('_')[0]}",
                "sampler.nsteps=2",
                "dataset.num_images=1",
                f"experiment.folder=pytest_generate_images_{dataset_name}_{operator_name}_{sampler_name}",
            ],
        )

        try:
            main(cfg)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()