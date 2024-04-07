from __future__ import annotations

import itertools as it
import random
from collections import defaultdict
from collections.abc import ItemsView, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, NamedTuple, TypeAlias, TypeVar, TypeVarTuple

import jax
import numpy as np
from numpy import ndarray

from social_choice_rlhf.lm.rlhf.architecture import Ordered
from social_choice_rlhf.lm.rlhf.loss import IRCallInput, PairwiseInput
from social_choice_rlhf.lm.rlhf.social_choice.tn import (
    TnCity,
    chattanooga_morristown_order,
    chattanooga_order,
    knoxville_morristown_order,
    knoxville_order,
    memphis_morristown_order,
    memphis_order,
    nashville_morristown_order,
    nashville_order,
    tn_cities,
)
from social_choice_rlhf.lm.rlhf.social_choice.types import (
    Complete,
    Incomplete,
    OrderedPartition,
    Population,
    RnSocialWelfareFunction,
)
from social_choice_rlhf.util.misc import declare_axis, fin, flatten, unzip_list

if TYPE_CHECKING:
    from numpy import Fin

One: TypeAlias = Literal[1]
A = TypeVar("A")
B = TypeVar("B")
Completion = TypeVar("Completion")
VocabSize = TypeVar("VocabSize", bound=int)
PromptLen = TypeVar("PromptLen", bound=int)
NumPrompts = TypeVar("NumPrompts", bound=int)
CompletionLen = TypeVar("CompletionLen", bound=int)
PrefLen = TypeVar("PrefLen", bound=int)
InPrefLen = TypeVar("InPrefLen", bound=int)
OutPrefLen = TypeVar("OutPrefLen", bound=int)
BatchLen = TypeVar("BatchLen", bound=int)
Vocab = TypeVar("Vocab", bound=int)
Shape = TypeVarTuple("Shape")
Prompt = TypeVar("Prompt")
Dim1 = TypeVar("Dim1", bound=int)
DType = TypeVar("DType")
Choice = TypeVar("Choice")

# =============================================================================
# Types for putting a population in the context of a prompt
# =============================================================================


@dataclass(frozen=True)
class HashableMapping(Mapping[A, B]):
    value: tuple[tuple[A, B], ...]

    @staticmethod
    def mk(mapping: Mapping[A, B]) -> HashableMapping[A, B]:
        return HashableMapping(tuple(mapping.items()))

    def to_mapping(self) -> Mapping[A, B]:
        return dict(self.value)

    def __getitem__(self, key: A) -> B:
        return self.to_mapping()[key]

    def __iter__(self) -> Iterator[A]:
        return (x[0] for x in iter(self.value))

    def __len__(self) -> int:
        return len(self.value)

    def items(self) -> ItemsView[A, B]:
        return self.to_mapping().items()


@dataclass(frozen=True)
class Profile(Generic[Prompt, Choice], Mapping[Prompt, OrderedPartition[Choice]]):
    """An individual's preference profile is defined by their preference order over a variety of prompts"""

    value: HashableMapping[Prompt, OrderedPartition[Choice]]

    def shuffled_prompts(self, seed: int) -> Profile[Prompt, Choice]:
        random.seed(seed)
        return Profile(HashableMapping.mk(dict(random.sample(list(self.value.items()), k=len(self.value)))))

    def __getitem__(self, key: Prompt) -> OrderedPartition[Choice]:
        return self.value[key]

    def __iter__(self) -> Iterator[Prompt]:
        return iter(self.value)

    def __len__(self) -> int:
        return len(self.value)

    def items(self) -> ItemsView[Prompt, OrderedPartition[Choice]]:
        return self.value.items()


@dataclass(frozen=True)
class PromptPopulation(Generic[Prompt, Choice]):
    """A `PromptPopulation` is a collection of `Profile`s along with their prevalences"""

    profiles: Mapping[Profile[Prompt, Choice], float]

    def iter_profiles(self, *, seed: int) -> Iterator[Profile[Prompt, Choice]]:
        random.seed(seed)
        return (
            random.choices(list(self.profiles.keys()), weights=list(self.profiles.values()), k=1)[0]
            for _ in it.count()
        )

    def iter_orders(self, *, seed: int) -> Iterator[tuple[Prompt, OrderedPartition[Choice]]]:
        random.seed(seed)
        return (random.choice(list(profile.items())) for profile in self.iter_profiles(seed=seed + 1))

    @staticmethod
    def mk(
        contests: Mapping[HashableMapping[Prompt, OrderedPartition[Choice]], float]
    ) -> PromptPopulation[Prompt, Choice]:
        assert 1 - 1e-5 < sum(contests.values()) < 1 + 1e-5
        return PromptPopulation({Profile(k): v for k, v in contests.items()})

    def pop_for_prompt(self, prompt: Prompt) -> Population[Incomplete, Choice]:
        raw_pop = defaultdict(float)
        for profile, weight in self.profiles.items():
            if prompt in profile:
                raw_pop[profile[prompt]] += weight
        total = sum(raw_pop.values())
        return Population.mk_incomplete({k: v / total for k, v in raw_pop.items()})

    def prompts(self) -> set[Prompt]:
        return set(flatten([p.value.keys() for p in self.profiles]))

    def choices(self) -> set[Choice]:
        return set(flatten([order.flat() for profile in self.profiles for order in profile.value.values()]))

    def prompt_weights(self) -> Mapping[Prompt, float]:
        all_prompts = self.prompts()
        raw = {p: sum(v for k, v in self.profiles.items() if p in k.value) for p in all_prompts}
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    def order_weights(self) -> Sequence[tuple[tuple[Prompt, OrderedPartition[Choice]], float]]:
        raw = flatten([[(i, weight) for i in profile.items()] for profile, weight in self.profiles.items()])
        total = sum(v for _, v in raw)
        return [(k, v / total) for k, v in raw]


# Empty `NamedTuple`s all compare equal.
# fmt: off
@dataclass(frozen=True)
class Pos: pass  # noqa: E701
@dataclass(frozen=True)
class Neg: pass  # noqa: E701
PosNeg: TypeAlias = Pos | Neg
# fmt: on


def split_population(pop: PromptPopulation[PosNeg, Choice]) -> PromptPopulation[PosNeg, Choice]:
    """Split a `PromptPopulation` defined over composite profiles into
    a population where each profile consists of only a single order
    """
    return PromptPopulation[PosNeg, Choice].mk(
        {HashableMapping.mk({Pos(): k}): v / 2 for k, v in pop.pop_for_prompt(Pos()).orders.items()}
        | {HashableMapping.mk({Neg(): k}): v / 2 for k, v in pop.pop_for_prompt(Neg()).orders.items()}
    )


# =============================================================================
# Tokenization
# =============================================================================

TnTokenSize: TypeAlias = Literal[8]
tn_token_size: TnTokenSize = 8


tn_city_to_token_mapping: Mapping[TnCity, Fin[TnTokenSize]] = {
    k: fin(v, tn_token_size) for v, k in enumerate(tn_cities, 3)
}


def tn_city_to_token(city: TnCity) -> Fin[TnTokenSize]:
    return tn_city_to_token_mapping[city]


token_to_tn_city_mapping: Mapping[Fin[TnTokenSize], TnCity] = {v: k for k, v in tn_city_to_token_mapping.items()}


def token_to_tn_city(token: Fin[TnTokenSize]) -> TnCity:
    return token_to_tn_city_mapping[token]


pad_token_id: Fin[TnTokenSize] = fin(0, tn_token_size)
normal_prompt_token_id: Fin[TnTokenSize] = fin(1, tn_token_size)
bizarro_prompt_token_id: Fin[TnTokenSize] = fin(2, tn_token_size)


def prompt_to_token_id(prompt: PosNeg) -> Fin[TnTokenSize]:
    match prompt:
        case Pos():
            return normal_prompt_token_id
        case Neg():
            return bizarro_prompt_token_id


# =============================================================================
# Masking logic
# =============================================================================


@dataclass(frozen=True)
class Mask:
    pass


def mk_dummy_partition(order: OrderedPartition[A], *, repeat_prob: float):
    """Mask out all but one position in ordered partition.
    Keep masks in their original positions.
    This effectively asks the model to make inferences based on a single absolution position in the input.
    (i.e. Because we put padding tokens in the masked slots, they're invisible to attention.
    But the position embedder ensures our one visible token still has the right position info.)
    """
    flat_order = order.flat()
    retained = random.randrange(0, len(flat_order))
    should_repeat = random.random() < repeat_prob
    return Completions(
        [Mask() if i != retained else x for i, x in enumerate(flat_order)],
        [x for i, x in enumerate(flat_order) if i != retained or should_repeat]
        + ([] if should_repeat else [Mask()]),
    )


def mk_mask_partition(order: OrderedPartition[A], *, mask_prob: float, repeat_prob: float):
    """Mask out some positions in the ordered partition.
    Shift all masks to the end of the sequence.
    This effectively asks the model to make inferences based on relative preferences.
    """
    flat_order = order.flat()
    choices = random.choices([None, Mask()], k=len(flat_order), weights=(1 - mask_prob, mask_prob))
    input_ = [x for x, y in zip(flat_order, choices, strict=True) if y is None]
    output = [
        x for x, y in zip(flat_order, choices, strict=True) if isinstance(y, Mask) or random.random() < repeat_prob
    ]
    return Completions(
        input_ + [Mask()] * (len(flat_order) - len(input_)), output + [Mask()] * (len(flat_order) - len(output))
    )


@dataclass
class Ref(Generic[A]):
    """A mutable reference"""

    value: A


class ObscureProbs(NamedTuple):
    # `mask` controls the probability of masking each element when in masking "mode"
    mask: float
    # `dummy` controls the probability of using a dummy obscure strategy vs a masking obscure strategy
    dummy: float
    # `repeat` controls the probability of an unmasked input element being repeated in the output
    # It's a mutable reference because we increase the repeat rate over the course of training
    repeat: Ref[float]


class Completions(NamedTuple, Generic[A]):
    in_: Sequence[A | Mask]
    out: Sequence[A | Mask]


def obscure_ordered_partition(order: OrderedPartition[A], probs: ObscureProbs) -> Completions[A]:
    if random.random() < probs.dummy:
        return mk_dummy_partition(order, repeat_prob=probs.repeat.value)
    else:
        return mk_mask_partition(order, mask_prob=probs.mask, repeat_prob=probs.repeat.value)


def obscure_ordered_partitions(
    orders: Sequence[OrderedPartition[A]], probs: ObscureProbs
) -> Sequence[Completions[A]]:
    obscured = [obscure_ordered_partition(order, probs) for order in orders]
    visible_out_count = sum(1 for x in flatten([out_seq for _, out_seq in obscured]) if not isinstance(x, Mask))
    if visible_out_count < 2:  # noqa: PLR2004
        return obscure_ordered_partitions(orders, probs)
    else:
        return obscured


# =============================================================================
# Batch handling logic
# =============================================================================


class PromptEncoding(NamedTuple, Generic[InPrefLen, OutPrefLen, PromptLen, CompletionLen, Vocab]):
    prompt: ndarray[PromptLen, Vocab]
    in_completions: ndarray[Ordered[InPrefLen], CompletionLen, Vocab]
    out_completions: ndarray[Ordered[OutPrefLen], CompletionLen, Vocab]


def padded_block(
    block_size: tuple[Dim1, *Shape],
    fill: DType,
    elems: Sequence[ndarray[*Shape, DType]],
):
    block = np.full(block_size, fill)
    for i, elem in enumerate(elems):
        block[i, : elem.shape[0]] = elem
    return block


def pad_prompt(  # noqa: PLR0913
    prompt: ndarray[PromptLen, Vocab],
    in_order: Sequence[ndarray[Any, Vocab]],
    out_order: Sequence[ndarray[Any, Vocab]],
    *,
    pad_token_id: Vocab,
    max_pref_len: Ordered[PrefLen],
    max_completion_len: CompletionLen,
) -> PromptEncoding[PrefLen, PrefLen, PromptLen, CompletionLen, Vocab]:
    return PromptEncoding(
        prompt,
        padded_block((max_pref_len, max_completion_len), pad_token_id, in_order),
        padded_block((max_pref_len, max_completion_len), pad_token_id, out_order),
    )


def pad_prompts(  # noqa: PLR0913
    x: Sequence[PromptEncoding[PrefLen, PrefLen, PromptLen, CompletionLen, Vocab]],
    *,
    pad_token_id: Vocab,
    max_num_prompts: NumPrompts,
    max_prompt_len: PromptLen,
    max_pref_len: Ordered[PrefLen],
    max_completion_len: CompletionLen,
) -> IRCallInput[NumPrompts, PrefLen, PrefLen, PromptLen, CompletionLen, Vocab]:
    prompt = padded_block((max_num_prompts, max_prompt_len), pad_token_id, [y.prompt for y in x])
    in_completions = padded_block(
        (max_num_prompts, max_pref_len, max_completion_len), pad_token_id, [y.in_completions for y in x]
    )
    out_completions = padded_block(
        (max_num_prompts, max_pref_len, max_completion_len), pad_token_id, [y.out_completions for y in x]
    )
    return IRCallInput(prompt, in_completions, out_completions)


def encode_profile(  # noqa: PLR0913
    x: Sequence[tuple[Prompt, Completions[Completion]]],
    completion_tokenize: Callable[[Completion | Mask], ndarray[CompletionLen, Vocab]],
    prompt_tokenize: Callable[[Prompt], ndarray[PromptLen, Vocab]],
    *,
    pad_token_id: Vocab,
    max_num_prompts: NumPrompts,
    max_prompt_len: PromptLen,
    max_pref_len: Ordered[PrefLen],
    max_completion_len: CompletionLen,
) -> IRCallInput[NumPrompts, PrefLen, PrefLen, PromptLen, CompletionLen, Vocab]:
    tokenized_profile = [
        (
            prompt_tokenize(p),
            [completion_tokenize(c) for c in comp.in_],
            [completion_tokenize(c) for c in comp.out],
        )
        for p, comp in x
    ]
    padded = [
        pad_prompt(
            prompt,
            in_completions,
            out_completions,
            pad_token_id=pad_token_id,
            max_pref_len=max_pref_len,
            max_completion_len=max_completion_len,
        )
        for prompt, in_completions, out_completions in tokenized_profile
    ]
    return pad_prompts(
        padded,
        pad_token_id=pad_token_id,
        max_num_prompts=max_num_prompts,
        max_prompt_len=max_prompt_len,
        max_pref_len=max_pref_len,
        max_completion_len=max_completion_len,
    )


def to_batch(  # noqa: PLR0913
    x: Sequence[Sequence[tuple[Prompt, Completions[Completion]]]],
    completion_tokenize: Callable[[Completion | Mask], ndarray[CompletionLen, Vocab]],
    prompt_tokenize: Callable[[Prompt], ndarray[PromptLen, Vocab]],
    *,
    pad_token_id: Vocab,
    max_num_prompts: NumPrompts,
    max_prompt_len: PromptLen,
    max_pref_len: Ordered[PrefLen],
    max_completion_len: CompletionLen,
) -> IRCallInput[int, NumPrompts, PrefLen, PrefLen, PromptLen, CompletionLen, Vocab]:
    return stack_inputs(
        [
            encode_profile(
                y,
                completion_tokenize,
                prompt_tokenize,
                pad_token_id=pad_token_id,
                max_num_prompts=max_num_prompts,
                max_prompt_len=max_prompt_len,
                max_pref_len=max_pref_len,
                max_completion_len=max_completion_len,
            )
            for y in x
        ]
    )


def stack_inputs(
    x: Sequence[IRCallInput[NumPrompts, InPrefLen, OutPrefLen, PromptLen, CompletionLen, Vocab]]
) -> IRCallInput[Any, NumPrompts, InPrefLen, OutPrefLen, PromptLen, CompletionLen, Vocab]:
    return IRCallInput(
        np.stack([y.prompt for y in x]),
        np.stack([y.in_completions for y in x]),
        np.stack([y.out_completions for y in x]),
    )


def pair_batch_from_order(
    prompt: ndarray[PromptLen, VocabSize],
    order: OrderedPartition[ndarray[CompletionLen, VocabSize]],
) -> PairwiseInput[int, PromptLen, CompletionLen, VocabSize]:
    """We train the `SocialRewardModel` on pairs but
    we may want to use multiple pairs from any given social order
    """
    winners, losers = unzip_list(order.pairs())
    return PairwiseInput(np.tile(prompt, (len(winners), 1)), np.stack(winners), np.stack(losers))


def pair_batch_from_order_batch(
    prompts: ndarray[BatchLen, PromptLen, VocabSize],
    orders: Sequence[OrderedPartition[ndarray[CompletionLen, VocabSize]]],
) -> PairwiseInput[int, PromptLen, CompletionLen, VocabSize]:
    batches = [pair_batch_from_order(prompt, order) for prompt, order in zip(prompts, orders, strict=True)]
    return PairwiseInput(
        np.concatenate([batch.prompt for batch in batches], axis=0),
        np.concatenate([batch.winner for batch in batches], axis=0),
        np.concatenate([batch.loser for batch in batches], axis=0),
    )


max_int = 2**31 - 1


def mk_pair_batch(
    pop: PromptPopulation[PosNeg, TnCity],
    swf: RnSocialWelfareFunction[Incomplete, TnCity],
    *,
    batch_len: BatchLen,
    seed: int,
) -> PairwiseInput[BatchLen, One, One, Fin[TnTokenSize]]:
    prompt_weights = pop.prompt_weights()
    order_sub_batches: list[PairwiseInput[int, One, One, Fin[TnTokenSize]]] = []

    def fn(seed: int):
        prompt = random.choices(list(prompt_weights.keys()), weights=list(prompt_weights.values()))[0]
        subpop = pop.pop_for_prompt(prompt)
        order = swf(subpop, jax.random.PRNGKey(seed))
        return pair_batch_from_order(
            np.array((prompt_to_token_id(prompt),)), order.map_(lambda x: np.array((tn_city_to_token(x),)))
        )

    while sum(x.prompt.shape[0] for x in order_sub_batches) < batch_len:
        order_sub_batches.append(fn(seed))
        seed += 1

    return PairwiseInput(
        prompt=declare_axis[BatchLen](
            0, np.concatenate([x.prompt for x in order_sub_batches], axis=0)[:batch_len, ...]
        ),
        winner=declare_axis[BatchLen](
            0, np.concatenate([x.winner for x in order_sub_batches], axis=0)[:batch_len, ...]
        ),
        loser=declare_axis[BatchLen](
            0, np.concatenate([x.loser for x in order_sub_batches], axis=0)[:batch_len, ...]
        ),
    )


# =============================================================================
# Test populations
# =============================================================================


# https://en.wikipedia.org/wiki/Kemeny%E2%80%93Young_method#Example
base_example_pop = Population[Complete, TnCity].mk_complete(
    {memphis_order: 0.42, nashville_order: 0.26, chattanooga_order: 0.15, knoxville_order: 0.17}
)
example_pop = PromptPopulation[PosNeg, TnCity].mk(
    {HashableMapping.mk({Pos(): k, Neg(): k.reverse()}): v for k, v in base_example_pop.orders.items()}
)
# Morristown is always ranked just below Knoxville
morristown_full_info_pop = PromptPopulation[PosNeg, TnCity].mk(
    {
        HashableMapping.mk({Pos(): memphis_morristown_order}): 0.30,
        HashableMapping.mk({Pos(): nashville_morristown_order}): 0.25,
        HashableMapping.mk({Pos(): chattanooga_morristown_order}): 0.15,
        HashableMapping.mk({Pos(): knoxville_morristown_order}): 0.30,
    }
)

# Like `morristown_full_info_pop` but some voters don't include Morristown in their ballots
# Useful for testing:
# Does the model learn to represent profiles similarly when they're compatible but
# some preference orders are less complete?
morristown_partial_info_pop = PromptPopulation[PosNeg, TnCity].mk(
    {
        HashableMapping.mk({Pos(): memphis_order}): 0.05,
        HashableMapping.mk({Pos(): memphis_morristown_order}): 0.25,
        HashableMapping.mk({Pos(): nashville_order}): 0.20,
        HashableMapping.mk({Pos(): nashville_morristown_order}): 0.05,
        HashableMapping.mk({Pos(): chattanooga_order}): 0.10,
        HashableMapping.mk({Pos(): chattanooga_morristown_order}): 0.05,
        HashableMapping.mk({Pos(): knoxville_order}): 0.25,
        HashableMapping.mk({Pos(): knoxville_morristown_order}): 0.05,
    }
)

prompt_generalization_full_info_pop = PromptPopulation[PosNeg, TnCity].mk(
    {
        HashableMapping.mk({Pos(): memphis_order, Neg(): memphis_order.reverse()}): 0.4,
        HashableMapping.mk({Pos(): nashville_order, Neg(): nashville_order.reverse()}): 0.25,
        HashableMapping.mk({Pos(): chattanooga_order, Neg(): chattanooga_order.reverse()}): 0.2,
        HashableMapping.mk({Pos(): knoxville_order, Neg(): knoxville_order.reverse()}): 0.15,
    }
)
# When we have some profiles with only partial prompt info,
# does the model learn to represent those profiles similarly to the compatible full info profiles?
prompt_generalization_partial_info_pop = PromptPopulation[PosNeg, TnCity].mk(
    {
        HashableMapping.mk({Pos(): memphis_order, Neg(): memphis_order.reverse()}): 0.35,
        HashableMapping.mk({Pos(): memphis_order}): 0.05,
        HashableMapping.mk({Pos(): nashville_order, Neg(): nashville_order.reverse()}): 0.05,
        HashableMapping.mk({Pos(): nashville_order}): 0.20,
        HashableMapping.mk({Pos(): chattanooga_order, Neg(): chattanooga_order.reverse()}): 0.05,
        HashableMapping.mk({Pos(): chattanooga_order}): 0.15,
        HashableMapping.mk({Pos(): knoxville_order, Neg(): knoxville_order.reverse()}): 0.05,
        HashableMapping.mk({Pos(): knoxville_order}): 0.10,
    }
)
# Does the model do this even when it doesn't have access to the full prompt info?
prompt_generalization_missing_info_pop = PromptPopulation[PosNeg, TnCity].mk(
    {
        HashableMapping.mk({Pos(): memphis_order, Neg(): memphis_order.reverse()}): 0.2,
        HashableMapping.mk({Pos(): memphis_order}): 0.2,
        HashableMapping.mk({Pos(): nashville_order}): 0.25,
        HashableMapping.mk({Pos(): chattanooga_order, Neg(): chattanooga_order.reverse()}): 0.10,
        HashableMapping.mk({Pos(): chattanooga_order}): 0.1,
        HashableMapping.mk({Pos(): knoxville_order, Neg(): knoxville_order.reverse()}): 0.07,
        HashableMapping.mk({Pos(): knoxville_order}): 0.08,
    }
)


pairwise_pop1 = Population[Complete, TnCity].mk_complete(
    {
        OrderedPartition.mk_simple(("a", "b", "c", "d")): 0.5,
        OrderedPartition.mk_simple(("b", "a", "d", "c")): 0.5,
    }
)
pairwise_pop2 = Population[Complete, TnCity].mk_complete(
    {
        OrderedPartition.mk_simple(("b", "a", "c", "d")): 0.5,
        OrderedPartition.mk_simple(("a", "b", "d", "c")): 0.5,
    }
)

# Any pairwise SWF will be unable to distinguish between populations like the above
# assert pairwise_scores(pairwise_pop1) == pairwise_scores(pairwise_pop2)
