import pytest
import torch
from pathlib import Path
from hydra import compose, initialize
from scripts.generate_observations import main

from lavps.paths import CONFIG_DIRECTORY

OPERATORS_PATH = CONFIG_DIRECTORY / "inverse_problem"

operator_files = [
    p.stem for p in Path(OPERATORS_PATH).glob("*.yaml")
    if not p.stem.startswith('_')
]


@pytest.mark.parametrize("dataset_name", ["ffhq", "imagenet"])
@pytest.mark.parametrize("operator_name", operator_files)
def test_generate_observations_does_not_smoke(dataset_name, operator_name):
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="generate_images",
            overrides=[
                f"dataset={dataset_name}_val",
                f"inverse_problem={operator_name}",
                f"model=hdit_{dataset_name}",
                "sampler.nsteps=2",
                "dataset.num_images=1",
                f"experiment.folder=pytest_generate_observations_{dataset_name}",
            ]
        )
        try:
            main(cfg)
        except Exception as e:
            pytest.fail(f"Production script failed on {dataset_name}/{operator_name}: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
