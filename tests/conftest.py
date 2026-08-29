import pytest

from brownlow.synthetic import make_synthetic_seasons


@pytest.fixture(scope="session")
def synthetic_seasons():
    """Three fake seasons with a known vote-generating process."""
    return make_synthetic_seasons(seasons=(2020, 2021, 2022), matches_per_season=40, seed=7)


@pytest.fixture(scope="session")
def small_season():
    return make_synthetic_seasons(seasons=(2020,), matches_per_season=12, seed=3)
