import os
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from model.full_model import FullModel


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args, device="cpu"):
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [sys.executable, "main.py", *map(str, args)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_missing_subcommand_reports_usage():
    result = run_cli()
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "command" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("command", [None, "organisms", "embeddings", "descriptors", "prediction"])
def test_cli_help(command):
    result = run_cli(*([] if command is None else [command]), "--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prediction_defaults_with_bundled_checkpoint(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    output = tmp_path / "predictions with spaces.csv"
    result = run_cli(
        "prediction", "dataset/Example data.csv", "target_organism", output,
        device=device,
    )
    assert result.returncode == 0, result.stderr

    source = pd.read_csv(ROOT / "dataset/Example data.csv")
    actual = pd.read_csv(output)
    assert list(actual.columns) == [*source.columns[:3], "prediction"]
    pd.testing.assert_frame_equal(actual.iloc[:, :3], source.iloc[:, :3])
    assert np.isfinite(actual["prediction"]).all()

    # Independently prepare the checkpoint's 132 scaled descriptors.
    artifacts = ROOT / "antibiofilm checkpoint"
    features = source.iloc[:, 3:].to_numpy(copy=True)
    scaler = joblib.load(artifacts / "normalizer.pkl")
    assert scaler.n_features_in_ == 132
    features[:, 768:900] = scaler.transform(features[:, 768:900])
    encoder = joblib.load(artifacts / "encoder.pkl")
    organisms = encoder.transform(source[["target_organism"]].to_numpy()).toarray()
    model = FullModel.load(artifacts / "antibiofilm_model.pth").eval()
    assert next(model.parameters()).device.type == "cpu"
    with torch.no_grad():
        expected = model(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(organisms, dtype=torch.float32),
        ).numpy()
    np.testing.assert_allclose(actual["prediction"], expected, rtol=1e-4, atol=1e-5)
