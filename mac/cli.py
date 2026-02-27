"""
MAC CLI — Multi-Agent Constitution Learning.

Usage:
    mac run configs/run.yaml
    mac run configs/run.yaml --verbose
"""

import argparse
import sys
from pathlib import Path


def cmd_run(args):
    """Run MAC training pipeline."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {config_path}")
        sys.exit(1)

    from .io import ConfigLoader
    from .epoch_pipeline import run_epoch_batch_constitutional_classifier

    config = ConfigLoader.load_config(config_path)

    print("MAC — Multi-Agent Constitution Learning")
    print(f"Config:  {config_path}")
    print(f"Model:   {config.get('model', {}).get('model_name', '?')}")
    print(f"Dataset: {config.get('data', {}).get('active_dataset', '?')}")
    print("=" * 60)

    results = run_epoch_batch_constitutional_classifier(config_path)
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Epochs:       {results.total_epochs}")
    print(f"  Time:         {results.total_training_time:.1f}s")
    print(f"  Constitution: v{results.final_constitution_version:04d}")
    print(f"  Train F1:     {results.training_metrics.get('final_f1', 0.0):.3f}")
    print(f"  Holdout F1:   {results.holdout_metrics.get('f1', 0.0):.3f}")


def main():
    parser = argparse.ArgumentParser(
        prog="mac",
        description="MAC: Multi-Agent Constitution Learning",
    )
    sub = parser.add_subparsers(dest="command")

    # mac run
    run_parser = sub.add_parser("run", help="Run training pipeline")
    run_parser.add_argument("config", help="Path to YAML config file")
    run_parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    try:
        if args.command == "run":
            cmd_run(args)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
