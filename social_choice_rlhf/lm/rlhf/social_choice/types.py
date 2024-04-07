from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeAlias, TypeVar

import jax

from social_choice_rlhf.util.misc import flatten

Choice = TypeVar("Choice")
Choice2 = TypeVar("Choice2")


@dataclass(frozen=True)
class OrderedPartition(Generic[Choice]):
    """Inner tuple represents equivalence class. Outer tuple represents order."""

    value: tuple[tuple[Choice, ...], ...]

    @staticmethod
    def mk(order: tuple[tuple[Choice, ...], ...]) -> OrderedPartition[Choice]:
        # flattened = flatten(order)
        # assert len(flattened) == len(set(flattened))
        return OrderedPartition(order)

    @staticmethod
    def mk_simple(choices: Sequence[Choice]) -> OrderedPartition[Choice]:
        return OrderedPartition.mk(tuple((c,) for c in choices))

    @staticmethod
    def mk_by_key(choices: set[Choice], key: Callable[[Choice], float]) -> OrderedPartition[Choice]:
        """Higher `key` value means more preferred."""
        return OrderedPartition.mk(
            tuple(
                tuple(choices) for _, choices in itertools.groupby(sorted(choices, key=key, reverse=True), key=key)
            )
        )

    def restrict(self, subset: Collection[Choice]) -> OrderedPartition[Choice]:
        return OrderedPartition.mk(
            tuple(
                x
                for x in (tuple(x for x in equivalence_class if x in subset) for equivalence_class in self.value)
                if len(x) != 0
            )
        )

    def map_(self, f: Callable[[Choice], Choice2]) -> OrderedPartition[Choice2]:
        return OrderedPartition.mk(
            tuple(tuple(f(x) for x in equivalence_class) for equivalence_class in self.value)
        )

    def reverse(self) -> OrderedPartition[Choice]:
        return OrderedPartition(self.value[::-1])

    def pairs(self) -> Sequence[tuple[Choice, Choice]]:
        # ATM, we assume that each equivalence class has at most one element
        flat = flatten(self.value)
        assert len(flat) == len(self.value)
        return list(itertools.combinations(flat, 2))

    def flat(self) -> tuple[Choice, ...]:
        flattened = tuple(flatten(self.value))
        assert len(flattened) == len(self.value)
        return flattened


# fmt: off
@dataclass(frozen=True)
class StrictlyPreferred: pass  # noqa: E701
@dataclass(frozen=True)
class Indifferent: pass  # noqa: E701
@dataclass(frozen=True)
class StrictlyDispreferred: pass  # noqa: E701
# fmt: on

Preference: TypeAlias = StrictlyPreferred | Indifferent | StrictlyDispreferred

Incomplete: TypeAlias = Literal["Incomplete"]
Complete: TypeAlias = Literal["Complete"]

PopType = TypeVar("PopType", Complete, Incomplete)


@dataclass(frozen=True)
class Population(Generic[PopType, Choice]):
    """Represents a population of voters with different preference orders.
    The `float` associated with each order represents the prevalence of that order in the population.
    """

    orders: Mapping[OrderedPartition[Choice], float]

    @staticmethod
    def mk_incomplete(orders: Mapping[OrderedPartition[Choice], float]) -> Population[Incomplete, Choice]:
        assert 1 - 1e-5 < sum(orders.values()) < 1 + 1e-5, sum(orders.values())
        # assert len({len(order.value) for order in orders}) == 1
        return Population(orders)

    @staticmethod
    def mk_complete(orders: Mapping[OrderedPartition[Choice], float]) -> Population[Complete, Choice]:
        assert 1 - 1e-5 < sum(orders.values()) < 1 + 1e-5, sum(orders.values())
        assert len({len(order.value) for order in orders}) == 1
        return Population(orders)

    @staticmethod
    def orders_to_incomplete(orders: Sequence[OrderedPartition[Choice]]) -> Population[Incomplete, Choice]:
        num_orders = len(orders)
        return Population.mk_incomplete({order: n / num_orders for order, n in Counter(orders).items()})

    @staticmethod
    def orders_to_complete(orders: Sequence[OrderedPartition[Choice]]) -> Population[Complete, Choice]:
        num_orders = len(orders)
        return Population.mk_complete({order: n / num_orders for order, n in Counter(orders).items()})

    def restrict(self, subset: Collection[Choice]) -> Population[PopType, Choice]:
        return Population({order.restrict(subset): prevalence for order, prevalence in self.orders.items()})

    def map_(
        self, f: Callable[[OrderedPartition[Choice]], OrderedPartition[Choice2]]
    ) -> Population[PopType, Choice2]:
        return Population({f(order): prevalence for order, prevalence in self.orders.items()})

    def choices(self) -> set[Choice]:
        return set(flatten([flatten(order.value) for order in self.orders]))


SocialWelfareFunction: TypeAlias = Callable[[Population[Complete, Choice]], OrderedPartition[Choice]]
SocialChoiceFunction: TypeAlias = Callable[[Population[Complete, Choice]], Collection[Choice]]
RnSocialWelfareFunction: TypeAlias = Callable[[Population[PopType, Choice], jax.Array], OrderedPartition[Choice]]


@dataclass(frozen=True)
class CardinalPreferences(Generic[Choice]):
    value: tuple[tuple[Choice, float], ...]

    @staticmethod
    def mk(preferences: Mapping[Choice, float]) -> CardinalPreferences[Choice]:
        return CardinalPreferences(tuple(sorted(preferences.items(), key=lambda x: x[1], reverse=True)))

    def to_ordinal(self) -> OrderedPartition[Choice]:
        return OrderedPartition.mk(
            tuple(
                tuple(c[0] for c in choices)
                for _, choices in itertools.groupby(
                    sorted(self.value, key=lambda x: x[1], reverse=True),
                    key=lambda x: x[1],
                )
            )
        )


@dataclass(frozen=True)
class CardinalPopulation(Generic[PopType, Choice]):
    orders: Mapping[CardinalPreferences[Choice], float]

    @staticmethod
    def mk_complete(orders: Mapping[CardinalPreferences[Choice], float]) -> CardinalPopulation[Complete, Choice]:
        assert 1 - 1e-5 < sum(orders.values()) < 1 + 1e-5, sum(orders.values())
        assert len({len(order.value) for order in orders}) == 1
        return CardinalPopulation(orders)

    def to_ordinal(
        self: CardinalPopulation[Complete, Choice],
    ) -> Population[Complete, Choice]:
        pop: dict[OrderedPartition[Choice], float] = defaultdict(float)
        for k, v in self.orders.items():
            pop[k.to_ordinal()] += v
        return Population.mk_complete(pop)
