from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar, TypeVarTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy import ndarray

from social_choice_rlhf.lm.rlhf.architecture import (
    IndividualRewardModel,
    IndividualRewardOutput,
    Ordered,
    SocialRewardModel,
)

if TYPE_CHECKING:
    from numpy import Fin

Batch = TypeVar("Batch")
Input = TypeVar("Input")
Extra = TypeVar("Extra")
Float = TypeVar("Float", bound=float)
BatchLen = TypeVar("BatchLen", bound=int)
Vocab = TypeVar("Vocab", bound=int)
SeqLen = TypeVar("SeqLen", bound=int)
PromptLen = TypeVar("PromptLen", bound=int)
NumPrompts = TypeVar("NumPrompts", bound=int)
CompletionLen = TypeVar("CompletionLen", bound=int)
VocabSize = TypeVar("VocabSize", bound=int)
MaxSeqLen = TypeVar("MaxSeqLen", bound=int)
MaxPromptLen = TypeVar("MaxPromptLen", bound=int)
MaxPrefLen = TypeVar("MaxPrefLen", bound=int)
EmbedSize = TypeVar("EmbedSize", bound=int)
InPrefLen = TypeVar("InPrefLen", bound=int)
OutPrefLen = TypeVar("OutPrefLen", bound=int)
PrefSize = TypeVar("PrefSize", bound=int)
Dim1 = TypeVar("Dim1", bound=int)
Model = TypeVar("Model", bound=eqx.Module)
Args = TypeVarTuple("Args")
Shape = TypeVarTuple("Shape")


class PairwiseInput(NamedTuple, Generic[*Shape, PromptLen, CompletionLen, Vocab]):
    prompt: ndarray[*Shape, PromptLen, Vocab]
    winner: ndarray[*Shape, CompletionLen, Vocab]
    loser: ndarray[*Shape, CompletionLen, Vocab]


@eqx.filter_value_and_grad(has_aux=True)
def pairwise_loss_fn(
    model: SocialRewardModel[VocabSize, MaxSeqLen, EmbedSize, Float],
    input_: PairwiseInput[BatchLen, PromptLen, CompletionLen, Fin[VocabSize]],
    key: jax.Array,
) -> tuple[ndarray[Float], tuple[()]]:
    def per_pair(
        prompt: ndarray[PromptLen, Fin[VocabSize]],
        winner: ndarray[CompletionLen, Fin[VocabSize]],
        loser: ndarray[CompletionLen, Fin[VocabSize]],
        key: jax.Array,
    ):
        def per_completion(completion: ndarray[CompletionLen, Fin[VocabSize]], key: jax.Array) -> ndarray[Float]:
            return model.__call__(prompt, completion)

        rewards = jax.vmap(per_completion)(jnp.stack((winner, loser), axis=0), jax.random.split(key, num=2))
        return -jax.nn.log_sigmoid(cast(ndarray[Float], rewards[0] - rewards[1]))

    return jnp.mean(
        jax.vmap(per_pair)(
            input_.prompt, input_.winner, input_.loser, jax.random.split(key, num=input_.prompt.shape[0])
        )
    ), ()


def kl_div(
    mean: ndarray[EmbedSize, Float],
    std_dev: ndarray[Float],
) -> ndarray[Float]:
    """KL divergence between an isotropic Gaussian with a constant std dev and a standard isotropic Gaussian"""
    variance = jnp.square(std_dev)
    # Constants pulled out of mean for perf
    return 0.5 * jnp.mean(jnp.square(mean)) + variance - jnp.log(variance) - 1


def list_mle(ordered_logits: ndarray[SeqLen, Float], mask: ndarray[SeqLen, bool]):
    # Without clipping, we sometimes get negative values
    clipped = jnp.clip(ordered_logits, -30, 30)
    neg_inf = jnp.array(-jnp.inf, dtype=clipped.dtype)
    # `cumsum` doesn't support `where` so we manually mask
    masked_logits = jnp.where(mask, clipped, neg_inf)
    # For numerical stability
    max_logit = jax.lax.stop_gradient(jnp.max(masked_logits, where=mask, initial=neg_inf))
    log_cum_sum = (
        jnp.flip(
            jnp.log(jnp.cumsum(jnp.exp(jnp.flip(masked_logits - max_logit, axis=0)))),
            axis=0,
        )
        + max_logit
    )
    return -jnp.sum(masked_logits - log_cum_sum, where=mask)


class ListLossOutput(
    NamedTuple, Generic[*Shape, NumPrompts, OutPrefLen, PromptLen, CompletionLen, PrefSize, EmbedSize, Float]
):
    list_mle_loss: ndarray[Float]
    kl_div_loss: ndarray[Float]
    outputs: IndividualRewardOutput[
        *Shape, NumPrompts, OutPrefLen, PromptLen, CompletionLen, PrefSize, EmbedSize, Float
    ]


class IRCallInput(NamedTuple, Generic[*Shape, NumPrompts, InPrefLen, OutPrefLen, PromptLen, CompletionLen, Vocab]):
    prompt: ndarray[*Shape, NumPrompts, PromptLen, Vocab]
    in_completions: ndarray[*Shape, NumPrompts, Ordered[InPrefLen], CompletionLen, Vocab]
    out_completions: ndarray[*Shape, NumPrompts, Ordered[OutPrefLen], CompletionLen, Vocab]


class PartialInput(NamedTuple, Generic[*Shape, InPrefLen, PromptLen, CompletionLen, Vocab]):
    prompt: ndarray[*Shape, PromptLen, Vocab]
    completions: ndarray[*Shape, InPrefLen, CompletionLen, Vocab]


def stack_partial_input(
    x: Sequence[PartialInput[InPrefLen, PromptLen, CompletionLen, VocabSize]],
) -> PartialInput[Any, InPrefLen, PromptLen, CompletionLen, VocabSize]:
    return PartialInput(
        prompt=np.stack([y.prompt for y in x], axis=0),
        completions=np.stack([y.completions for y in x], axis=0),
    )


@eqx.filter_value_and_grad(has_aux=True)
def list_loss_fn(
    model: IndividualRewardModel[VocabSize, MaxSeqLen, MaxPromptLen, MaxPrefLen, PrefSize, EmbedSize, Float],
    input_: IRCallInput[BatchLen, NumPrompts, InPrefLen, OutPrefLen, PromptLen, CompletionLen, Fin[VocabSize]],
    beta: Float,
    pad_token_id: Fin[VocabSize],
    key: jax.Array,
) -> tuple[
    ndarray[Float],
    ListLossOutput[
        BatchLen, NumPrompts, Ordered[OutPrefLen], PromptLen, CompletionLen, PrefSize, EmbedSize, Float
    ],
]:
    def single(
        prompts: ndarray[NumPrompts, PromptLen, Fin[VocabSize]],
        in_completions: ndarray[NumPrompts, Ordered[InPrefLen], CompletionLen, Fin[VocabSize]],
        out_completions: ndarray[NumPrompts, Ordered[OutPrefLen], CompletionLen, Fin[VocabSize]],
        key: jax.Array,
    ):
        output = model.__call__(
            prompts=prompts,
            ordered_completions=in_completions,
            out_completions=out_completions,
            key=key,
        )
        all_pad = jnp.all(out_completions == pad_token_id, axis=-1)
        mle_losses = jax.vmap(list_mle)(output.rewards, ~all_pad)
        return (
            output,
            mle_losses,
            kl_div(
                output.preference_output.mean,
                output.preference_output.std_dev,
            ),
        )

    def de_aux(
        x: jax.AuxDim[
            Dim1,
            tuple[
                IndividualRewardOutput[
                    NumPrompts, Ordered[OutPrefLen], PromptLen, CompletionLen, PrefSize, EmbedSize, Float
                ],
                ndarray[NumPrompts, Float],
                ndarray[Float],
            ],
        ],
    ) -> tuple[
        IndividualRewardOutput[
            Dim1, NumPrompts, Ordered[OutPrefLen], PromptLen, CompletionLen, PrefSize, EmbedSize, Float
        ],
        ndarray[Dim1, NumPrompts, Float],
        ndarray[Dim1, Float],
    ]:
        """Push auxiliary dim from `vmap` down into the pieces.
        Kind of ugly but `AuxDim` allows us to annotate `vmap` properly without
        `jax` knowing about every pytree we use.
        """
        return cast(Any, x)

    all_pad = jnp.all(input_.prompt == pad_token_id, axis=-1)

    outputs, list_mles, kl_divs = de_aux(
        jax.vmap(single)(
            input_.prompt,
            input_.in_completions,
            input_.out_completions,
            jax.random.split(key, num=input_.prompt.shape[0]),
        )
    )
    list_mle_loss = jnp.mean(list_mles, where=~all_pad)
    kl_div_loss = jnp.mean(kl_divs)

    return list_mle_loss + beta * kl_div_loss, ListLossOutput(list_mle_loss, beta * kl_div_loss, outputs)
