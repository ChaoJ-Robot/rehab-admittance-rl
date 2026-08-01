"""Load all Phase 0 configuration templates and print a short summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from rehab_sim.config import load_config_bundle
from rehab_sim.logging_config import configure_logging


def main() -> int:
    """Validate that all configured YAML documents are loadable."""

    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=project_root / "configs",
        help="directory containing the YAML configuration templates",
    )
    args = parser.parse_args()

    logger = configure_logging()
    bundle = load_config_bundle(args.config_dir)
    logger.info("loaded %d configuration files from %s", len(bundle), args.config_dir)
    for name, data in bundle.items():
        logger.info("config=%s top_level_keys=%s", name, sorted(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
