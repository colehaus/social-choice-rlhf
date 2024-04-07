from __future__ import annotations

import hypothesis
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.strategies import DrawFn, SearchStrategy, composite

from social_choice_rlhf.lm.rlhf.social_choice.core import (
    borda_count,
    majority_verbose,
    max_utility,
    unanimity_verbose,
    utility_of,
)
from social_choice_rlhf.lm.rlhf.social_choice.tn import *  # noqa: F403
from social_choice_rlhf.lm.rlhf.social_choice.types import *  # noqa: F403


def strict_order_strat(choices: Collection[Choice]) -> SearchStrategy[OrderedPartition[Choice]]:
    return st.permutations(list(choices)).map(OrderedPartition.mk_simple)


@composite
def population_strat(draw: DrawFn, choices: Collection[Choice]) -> Population[Complete, Choice]:
    all_orders = [OrderedPartition.mk_simple(x) for x in list(itertools.permutations(list(choices)))]
    orders = draw(st.lists(st.sampled_from(all_orders), min_size=1, max_size=len(all_orders), unique=True))
    prevalences = draw(
        st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=len(orders) - 1,
            max_size=len(orders) - 1,
        )
        # Ensure that it never sums to 0
        .map(lambda x: [1, *x])
        .map(lambda x: [y / sum(x) for y in x])
    )
    return Population.mk_complete(dict(zip(orders, prevalences)))


@given(population_strat(tn_cities))
def test_erb_unanimity(population: Population[Complete, TnCity]):
    cand, final = unanimity_verbose(borda_count)(population)
    hypothesis.note(f"swf: {borda_count(population)}")
    hypothesis.note(f"candidates: {cand}")
    hypothesis.note(f"final: {final}")
    assert cand.issubset(final)


@given(population_strat(tn_cities))
def test_borda_majority(population: Population[Complete, TnCity]):
    majority_cand, swf_res = majority_verbose(borda_count)(population)
    assert majority_cand == swf_res, (majority_cand, swf_res)


# AssertionError: (Knoxville(), Memphis())
# Falsifying example: test_borda_majority(
#   population=Population(orders={
#     OrderedPartition(value=((Memphis(),), (Chattanooga(),), (Nashville(),), (Knoxville(),))): 0.3333333333333333,
#     OrderedPartition(value=((Knoxville(),), (Memphis(),), (Chattanooga(),), (Nashville(),))): 0.6666666666666666
#   }),
# )


def cardinal_preferences_strat(choices: Collection[Choice]) -> SearchStrategy[CardinalPreferences[Choice]]:
    return st.lists(
        st.integers(min_value=0, max_value=100),
        min_size=len(choices),
        max_size=len(choices),
        unique=True,
    ).map(lambda x: CardinalPreferences.mk(dict(zip(choices, x, strict=True))))


@composite
def cardinal_population_strat(
    draw: DrawFn, num_distinct_profiles: int, choices: Collection[Choice]
) -> CardinalPopulation[Complete, Choice]:
    all_profiles = draw(
        st.lists(
            cardinal_preferences_strat(choices),
            min_size=num_distinct_profiles,
            max_size=num_distinct_profiles,
            unique=True,
        )
    )
    prevalences = draw(
        st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=len(all_profiles) - 1,
            max_size=len(all_profiles) - 1,
        )
        # Ensure that it never sums to 0
        .map(lambda x: [1, *x])
        .map(lambda x: [y / sum(x) for y in x])
    )
    return CardinalPopulation.mk_complete(dict(zip(all_profiles, prevalences)))


@given(cardinal_population_strat(5, tn_cities))
def test_borda_utility(population: CardinalPopulation[Complete, Choice]):
    mu = max_utility(population)
    # print(mu)
    borda_winner = borda_count(population.to_ordinal()).value[0][0]
    bu = utility_of(population, borda_winner)
    # print(borda_winner, bu)
    diff = mu[1] - bu
    hypothesis.target(diff)
    print(diff)
    assert diff < 65  # noqa: PLR2004


def mk_borda_perverse(num_dummies: int):
    """Generate a population with an arbitrarily large divergence between utility of the Borda winner
    and the utility of the max utility candidate
    """
    prevalence = 1 / (num_dummies + 2)
    epsilon = 1e-5
    step_size = 10_000
    ord_order = ["ord_win", *[str(i) for i in range(num_dummies)], "util_win"]
    ord_utils = [1 - i / step_size for i in range(num_dummies + 2)]
    util_order = ["util_win", "ord_win", *[str(i) for i in range(num_dummies)]]
    util_utils = [1.0, *[-1 + i / step_size for i in reversed(range(num_dummies + 1))]]
    return CardinalPopulation.mk_complete(
        {
            CardinalPreferences.mk(dict(zip(ord_order, ord_utils, strict=True))): prevalence + epsilon,
            CardinalPreferences.mk(dict(zip(util_order, util_utils, strict=True))): 1 - prevalence - epsilon,
        }
    )
