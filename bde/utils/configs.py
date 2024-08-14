r"""Configuration settings for the Bayesian Deep Ensembles (BDE) framework.

This module contains configuration settings that are used throughout the BDE framework.
Currently.

Classes
--------
- General: Holds general constants for the framework.
"""

import pytest
from collections.abc import Sized, Iterable, Generator
from typing import Any, Union


class General:
    r"""A class to hold general configuration settings for the BDE framework.
    """
    SEED = 42


if __name__ == '__main__':
    pytest.main()
