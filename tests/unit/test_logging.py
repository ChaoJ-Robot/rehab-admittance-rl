import logging

import pytest
from rehab_sim.logging_config import configure_logging


def test_configure_logging_sets_level() -> None:
    logger_name = "rehab.test.logging"
    logger = configure_logging("DEBUG", logger_name)

    assert logger.name == logger_name
    assert logger.level == logging.DEBUG
    assert logger.handlers


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError):
        configure_logging("NOT_A_LEVEL", "rehab.test.invalid")
