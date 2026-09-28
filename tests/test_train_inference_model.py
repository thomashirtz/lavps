import pytest
from pathlib import Path
from hydra import compose, initialize
from scripts.train_inference_model import main

from lavps.paths import CONFIG_DIRECTORY, OUTPUT_DIRECTORY

# 1. SETUP PATHS
# Adjust these relative to where this test file is located
OPERATORS_PATH = CONFIG_DIRECTORY / "inverse_problem"

# Get all operators except defaults to test variety
operator_files = [
    p.stem for p in Path(OPERATORS_PATH).glob("*.yaml")
    if not p.stem.startswith('_')
]
operator_files = ['mb_21_0.9']


@pytest.mark.parametrize("dataset_name", ["ffhq_train"])
@pytest.mark.parametrize("operator_name", operator_files)
def test_train_inference_model_does_not_smoke(dataset_name, operator_name):
    """
    Programmatically initializes Hydra and runs the variational sampler.
    """
    test_folder = OUTPUT_DIRECTORY / f"pytest_results_{operator_name}"

    # 2. INITIALIZE HYDRA CONFIG
    # version_base matches your script (1.1)
    with initialize(version_base="1.1", config_path="../configs"):
        cfg = compose(
            config_name="train_inference_model",
            overrides=[
                f"dataset={dataset_name}",
                f"inverse_problem={operator_name}",
                f"output_path={test_folder}",
                "training.num_epochs=1",
                "training.train_split=0.0001",
                "training.val_split=0.0001",
                "checkpoint.enabled=False",
            ]
        )

        # When calling a @hydra.main function manually,
        # we pass the config object directly.
        main(cfg)

        # 4. VERIFY RESULTS
        # save_path = Path(cfg.save_folder)
        # assert save_path.exists(), f"Save folder {save_path} was not created"
        # assert (save_path / "config.yaml").exists(), "Config was not saved in output"
