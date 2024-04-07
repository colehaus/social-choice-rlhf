from __future__ import annotations

from social_choice_rlhf.lm.rlhf.social_choice.types import *  # noqa: F403


# Empty `NamedTuple`s all compare equal.
# fmt: off
@dataclass(frozen=True)
class Memphis: pass  # noqa: E701
@dataclass(frozen=True)
class Nashville: pass  # noqa: E701
@dataclass(frozen=True)
class Knoxville: pass  # noqa: E701
@dataclass(frozen=True)
class Chattanooga: pass  # noqa: E701
@dataclass(frozen=True)
class Morristown: pass  # noqa: E701
# fmt: on
TnCity: TypeAlias = Memphis | Nashville | Knoxville | Chattanooga | Morristown
tn_cities = {Memphis(), Nashville(), Knoxville(), Chattanooga(), Morristown()}


def to_short_str(city: TnCity) -> str:
    return city.__class__.__name__[:2]


memphis_order: OrderedPartition[TnCity] = OrderedPartition.mk_simple(
    (Memphis(), Nashville(), Chattanooga(), Knoxville())
)
nashville_order: OrderedPartition[TnCity] = OrderedPartition.mk_simple(
    (Nashville(), Chattanooga(), Knoxville(), Memphis())
)
chattanooga_order: OrderedPartition[TnCity] = OrderedPartition.mk_simple(
    (Chattanooga(), Knoxville(), Nashville(), Memphis())
)
knoxville_order: OrderedPartition[TnCity] = OrderedPartition.mk_simple(
    (Knoxville(), Chattanooga(), Nashville(), Memphis())
)
morristown_order: OrderedPartition[TnCity] = OrderedPartition.mk_simple(
    (Morristown(), Knoxville(), Chattanooga(), Nashville(), Memphis())
)
memphis_morristown_order: OrderedPartition[TnCity] = OrderedPartition.mk_simple(
    (Memphis(), Nashville(), Chattanooga(), Knoxville(), Morristown())
)
nashville_morristown_order: OrderedPartition[TnCity] = OrderedPartition.mk_simple(
    (Nashville(), Chattanooga(), Knoxville(), Morristown(), Memphis())
)
chattanooga_morristown_order: OrderedPartition[TnCity] = OrderedPartition.mk_simple(
    (Chattanooga(), Knoxville(), Morristown(), Nashville(), Memphis())
)
knoxville_morristown_order: OrderedPartition[TnCity] = OrderedPartition.mk_simple(
    (Knoxville(), Morristown(), Chattanooga(), Nashville(), Memphis())
)
