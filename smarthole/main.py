"""CLI entry point for Smarthole."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np

from src.predict import predict_from_csv
from train import run_training_pipeline

np.random.seed(42)
random.seed(42)


def setup_logging(project_root: Path) -> None:
    """Configure console + file logging.

    Parameters
    ----------
    project_root : Path
        Project root directory where log file is stored.
    """
    log_file = project_root / "smarthole.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI args.
    """
    parser = argparse.ArgumentParser(description="Smarthole ML pipeline")
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--input", type=str, help="Input CSV path for predict mode")
    parser.add_argument("--model", type=str, default="Extra Trees", help="Model name for predict mode")
    return parser.parse_args()


def main() -> None:
    """Run selected CLI mode."""
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parent
    setup_logging(project_root)

    if args.mode == "train":
        run_training_pipeline(config_path)
    else:
        if not args.input:
            raise ValueError("--input is required in predict mode")
        predict_from_csv(
            input_csv=Path(args.input).resolve(),
            model_name=args.model,
            config_path=config_path,
        )


if __name__ == "__main__":
    main()

