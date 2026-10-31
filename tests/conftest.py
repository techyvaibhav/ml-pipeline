"""Shared fixtures: a tiny synthetic dataset and a quickly-trained model."""
import pytest

from data.download_data import make_synthetic
from src import train


@pytest.fixture(scope="session")
def raw():
    """Raw synthetic df (same format as the real csv)."""
    return make_synthetic(900, seed=0)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, raw):
    """Train once in a temp dir; returns (dir, metrics)."""
    d = tmp_path_factory.mktemp("model")  # temp model dir
    m = train.run(df=raw, model_dir=d, rep_dir=d / "reports", proc_dir=None, quick=True)
    return d, m
