from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import hypothesis
import jax.numpy as jnp

from social_choice_rlhf.lm.rlhf.social_choice.types import *  # noqa: F403

A = TypeVar("A")
B = TypeVar("B")


def pairwise_preference(preference_order: OrderedPartition[Choice], a: Choice, b: Choice) -> Preference:
    a_index = next(i for i, e in enumerate(preference_order.value) if a in e)
    b_index = next(i for i, e in enumerate(preference_order.value) if b in e)
    if a_index < b_index:
        return StrictlyPreferred()
    elif a_index > b_index:
        return StrictlyDispreferred()
    else:
        return Indifferent()


def pairwise_contest(population: Population[Complete, Choice], a: Choice, b: Choice) -> Mapping[Preference, float]:
    preferences = [
        (pairwise_preference(order, a, b), prevalence) for order, prevalence in population.orders.items()
    ]
    outcome = {StrictlyPreferred(): 0.0, Indifferent(): 0, StrictlyDispreferred(): 0}
    for preference, prevalence in preferences:
        outcome[preference] += prevalence
    return outcome


# =============================================================================
# SWFs
# =============================================================================


def one_zero_negative_one(x: Preference):
    return {StrictlyPreferred(): 1.0, Indifferent(): 0.0, StrictlyDispreferred(): -1.0}[x]


def copeland(
    population: Population[Complete, Choice], rule: Callable[[Preference], float] = one_zero_negative_one
) -> OrderedPartition[Choice]:
    def copeland_collate(pairwise_result: Mapping[Preference, float]) -> Preference:
        if pairwise_result[StrictlyPreferred()] > pairwise_result[StrictlyDispreferred()]:
            return StrictlyPreferred()
        elif pairwise_result[StrictlyPreferred()] < pairwise_result[StrictlyDispreferred()]:
            return StrictlyDispreferred()
        else:
            return Indifferent()

    choices = flatten(next(iter(population.orders)).value)
    scores: Mapping[Choice, float] = {
        choice: sum(rule(copeland_collate(pairwise_contest(population, choice, other))) for other in choices)
        for choice in choices
    }
    return OrderedPartition.mk_by_key(set(choices), lambda choice: scores[choice])


def kemeny(population: Population[Complete, Choice]) -> OrderedPartition[Choice]:
    choices = flatten(next(iter(population.orders)).value)
    pairwise_scores: Mapping[tuple[Choice, Choice], float] = {
        (a, b): pairwise_contest(population, a, b)[StrictlyPreferred()]
        for a, b in itertools.product(choices, choices)
    }
    sequences = tuple(itertools.permutations(choices))
    sequence_pairs = {sequence: itertools.combinations(sequence, 2) for sequence in sequences}
    sequence_scores = {
        sequence: sum(pairwise_scores[(a, b)] for a, b in pairs) for sequence, pairs in sequence_pairs.items()
    }
    return OrderedPartition(
        tuple((choice,) for choice in max(sequence_scores, key=lambda seq: sequence_scores[seq]))
    )


def minimax(population: Population[Complete, Choice]) -> OrderedPartition[Choice]:
    choices = flatten(next(iter(population.orders)).value)
    pairwise_scores: Mapping[tuple[Choice, Choice], float] = {
        (a, b): pairwise_contest(population, a, b)[StrictlyPreferred()]
        for a, b in itertools.product(choices, choices)
    }
    pairwise_margins = {
        a: {b: pairwise_scores[(a, b)] - pairwise_scores[(b, a)] for b in choices} for a in choices
    }
    worst_matchup: dict[Choice, tuple[Choice, float]] = {
        choice: min(margins.items(), key=lambda x: x[1]) for choice, margins in pairwise_margins.items()
    }
    return OrderedPartition.mk_by_key(set(choices), lambda choice: worst_matchup[choice][1])


def random_ballot_swf(population: Population[PopType, Choice], key: jax.Array) -> OrderedPartition[Choice]:
    orders = list(population.orders.keys())
    cum_probs = jnp.cumsum(jnp.array(list(population.orders.values())))
    r = jax.random.uniform(key)
    index = jnp.searchsorted(cum_probs, r)
    return orders[index]


def borda_count(population: Population[Complete, Choice]):
    """Slightly different phrasing than standard Borda count, but semantically identical.

    This phrasing makes the connection to random ballot a little clearer.
    """

    def regroup(x: Mapping[tuple[A, A], B]) -> Mapping[A, Mapping[A, B]]:
        nested = defaultdict(dict)
        for (a, b), v in x.items():
            nested[a][b] = v
        return nested

    choices = flatten(next(iter(population.orders)).value)
    pairwise_scores_ = regroup(pairwise_scores(population))
    # print({choice: sum(pairwise_scores[choice].values()) for choice in choices})
    return OrderedPartition.mk_by_key(set(choices), lambda choice: sum(pairwise_scores_[choice].values()))


def pairwise_scores(population: Population[Complete, Choice]) -> Mapping[tuple[Choice, Choice], float]:
    choices = flatten(next(iter(population.orders)).value)
    return {
        (a, b): pairwise_contest(population, a, b)[StrictlyPreferred()]
        for a, b in itertools.product(choices, choices)
    }


# =============================================================================
# Properties
# =============================================================================


def powerset(x: Collection[Choice]):
    return itertools.chain.from_iterable(itertools.combinations(x, r) for r in range(len(x) + 1))


class RestrictReturn(NamedTuple, Generic[Choice]):
    pre_restrict: OrderedPartition[Choice]
    post_restrict: OrderedPartition[Choice]


def strong_iia_verbose(
    swf: SocialWelfareFunction[Choice]
) -> Callable[[Population[Complete, Choice]], Mapping[frozenset[Choice], RestrictReturn[Choice]]]:
    """For all subsets of choices,
    if we restrict individual preference orderings to the subset and then run the SWF,
    does this produce the same social order as running the SWF on the full set of choices and then restricting?
    """

    def inner(population: Population[Complete, Choice]) -> Mapping[frozenset[Choice], RestrictReturn[Choice]]:
        choices = flatten(next(iter(population.orders)).value)
        subsets = powerset(choices)
        return {
            k: v
            for k, v in {
                frozenset(subset): (
                    RestrictReturn(
                        swf(population.restrict(subset)),
                        swf(population).restrict(subset),
                    )
                )
                for subset in subsets
            }.items()
            if v.pre_restrict != v.post_restrict
        }

    return inner


def strong_iia(swf: SocialWelfareFunction[Choice]) -> Callable[[Population[Complete, Choice]], bool]:
    def inner(population: Population[Complete, Choice]) -> bool:
        return all(
            pre_restrict == post_restrict
            for pre_restrict, post_restrict in strong_iia_verbose(swf)(population).values()
        )

    return inner


def majority_verbose(
    swf: SocialWelfareFunction[Choice]
) -> Callable[[Population[Complete, Choice]], tuple[Choice, tuple[Choice, ...]]]:
    """If a majority of voters rank A at #1, does the social order rank A at #1?"""

    def inner(population: Population[Complete, Choice]):
        firsts = flatten([[(o, p) for o in order.value[0]] for order, p in population.orders.items()])
        first_totals = defaultdict(float)
        for first, p in firsts:
            first_totals[first] += p
        highest: tuple[Choice, float] = max(first_totals.items(), key=lambda x: x[1])
        hypothesis.target(highest[1])
        hypothesis.assume(highest[1] > 0.5)  # noqa: PLR2004
        return highest[0], swf(population).value[0]

    return inner


def majority(swf: SocialWelfareFunction[Choice]) -> Callable[[Population[Complete, Choice]], bool]:
    """If a majority of voters rank A at #1, does the social order rank A at #1?"""

    def inner(population: Population[Complete, Choice]):
        highest, social_highest = majority_verbose(swf)(population)
        return highest in social_highest

    return inner


class UnanimityReturn(NamedTuple, Generic[Choice]):
    top_candidates: set[tuple[Choice, Choice]]
    social_pairs: set[tuple[Choice, Choice]]


def unanimity_verbose(
    swf: SocialWelfareFunction[Choice]
) -> Callable[[Population[Complete, Choice]], UnanimityReturn[Choice]]:
    """If every voter prefers choice A to choice B, is A preferred to B in the social order?"""

    def inner(population: Population[Complete, Choice]):
        choices = flatten(next(iter(population.orders)).value)
        pairs = list(itertools.product(choices, choices))
        top_candidates = {
            pair
            for pair in pairs
            if all(
                pairwise_preference(order, pair[0], pair[1]) == StrictlyPreferred() for order in population.orders
            )
        }
        if hypothesis.currently_in_test_context():
            hypothesis.target(len(top_candidates))
        hypothesis.assume(len(top_candidates) > 0)
        social_pairs = {
            pair
            for pair in pairs
            if pairwise_preference(swf(population), pair[0], pair[1]) == StrictlyPreferred()
            and pair in top_candidates
        }
        return UnanimityReturn(top_candidates, social_pairs)

    return inner


def unanimity(swf: SocialWelfareFunction[Choice]) -> Callable[[Population[Complete, Choice]], bool]:
    """If every voter prefers choice A to choice B, is A preferred to B in the social order?"""

    def inner(population: Population[Complete, Choice]) -> bool:
        top_candidates, social_pairs = unanimity_verbose(swf)(population)
        return top_candidates.issubset(social_pairs)

    return inner


def utility_of(population: CardinalPopulation[Complete, Choice], candidate: Choice) -> float:
    return sum(prevalence * dict(profile.value)[candidate] for profile, prevalence in population.orders.items())


def max_utility(population: CardinalPopulation[Complete, Choice]) -> tuple[Choice, float]:
    choices = set(dict(next(iter(population.orders)).value).keys())
    return max(((choice, utility_of(population, choice)) for choice in choices), key=lambda x: x[1])
