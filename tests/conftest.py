"""Shared fixtures.

The invariant tests are parameterised over *every* feature configuration the
project ships. That is deliberate: the bugs this suite was written after all
lived in configurations no test exercised, not in logic no test covered.
"""

from types import SimpleNamespace

import pytest

from brownlow.features import FeatureConfig
from brownlow.model import PlackettLuceModel, WeightedLogisticModel
from brownlow.synthetic import make_synthetic_seasons

#: One entry per meaningfully different way of building features. Anything the
#: shipped experiment configs can switch on should appear here.
FEATURE_CONFIGS = {
    "default": FeatureConfig(),
    "raw-counts-only": FeatureConfig(within_match_stats=(), win_interactions=(),
                                     include_shares=False, include_ranks=False),
    "history": FeatureConfig(include_history=True),
    "form": FeatureConfig(include_form=True),
    "interactions": FeatureConfig(include_interactions=True, include_match_best=True),
    "everything": FeatureConfig(include_history=True, include_form=True,
                                include_interactions=True, include_match_best=True),
}


@pytest.fixture(scope="session")
def synthetic_seasons():
    """Three fake seasons with a known vote-generating process."""
    return make_synthetic_seasons(seasons=(2020, 2021, 2022), matches_per_season=40, seed=7)


@pytest.fixture(scope="session")
def small_season():
    return make_synthetic_seasons(seasons=(2020,), matches_per_season=12, seed=3)


@pytest.fixture(scope="session")
def messy_season():
    """Synthetic data carrying the awkward shapes real data actually has.

    Clean fixtures hide bugs. Two players share a name at different clubs, as
    the AFL has genuinely had, and some rows are missing their player id.
    """
    frame = make_synthetic_seasons(seasons=(2020, 2021), matches_per_season=25, seed=11)
    frame = frame.copy()
    frame["player_id"] = frame.groupby("player", sort=False).ngroup().astype(float)

    teams = frame["team"].unique()[:2]
    for team in teams:
        on_team = frame["team"] == team
        victim = frame.loc[on_team, "player"].iloc[0]
        frame.loc[frame["player"] == victim, "player"] = "Shared Name"

    # A handful of rows with no id at all, as the upstream mirror produces.
    frame.loc[frame.index[::97], "player_id"] = float("nan")
    return frame.reset_index(drop=True)


@pytest.fixture(scope="session", params=list(FEATURE_CONFIGS), ids=list(FEATURE_CONFIGS))
def fitted(request, synthetic_seasons):
    """A fitted model and its predictions, once per feature configuration."""
    config = FEATURE_CONFIGS[request.param]
    model = PlackettLuceModel(alpha=1.0, feature_config=config).fit(synthetic_seasons)
    return SimpleNamespace(
        name=request.param,
        config=config,
        model=model,
        frame=synthetic_seasons,
        predictions=model.predict(synthetic_seasons),
    )


@pytest.fixture(scope="session", params=["plackett_luce", "logistic"])
def fitted_any_model(request, synthetic_seasons):
    """The same checks applied to both model classes."""
    factory = {"plackett_luce": PlackettLuceModel, "logistic": WeightedLogisticModel}
    model = factory[request.param](alpha=1.0).fit(synthetic_seasons)
    return SimpleNamespace(
        name=request.param,
        model=model,
        frame=synthetic_seasons,
        predictions=model.predict(synthetic_seasons),
    )
