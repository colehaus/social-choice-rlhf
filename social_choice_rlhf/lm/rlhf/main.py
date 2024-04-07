from __future__ import annotations

import itertools as it
from collections import deque
from collections.abc import Callable, Sequence
from statistics import mean
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar, cast

import equinox as eqx
import jax
import matplotlib as mpl
import numpy as np
import optax
import seaborn as sns
from numpy import float32, ndarray
from sklearn.mixture import GaussianMixture

from social_choice_rlhf.lm.architecture import ArchConfig, NormConfig, ScaledAttention, TransformerLayerConfig
from social_choice_rlhf.lm.rlhf.architecture import (
    Decoded,
    IndividualRewardModel,
    IRConfig,
    Ordered,
    PrefDecoderConfig,
    PreferenceModel,
    SocialRewardModel,
)
from social_choice_rlhf.lm.rlhf.data import (
    Completions,
    Mask,
    ObscureProbs,
    Pos,
    PosNeg,
    Profile,
    PromptPopulation,
    Ref,
    TnTokenSize,
    mk_pair_batch,
    obscure_ordered_partitions,
    pad_token_id,
    pair_batch_from_order_batch,
    prompt_to_token_id,
    tn_city_to_token,
    to_batch,
)
from social_choice_rlhf.lm.rlhf.diagnostics import check_learned_orders, nd_to_2d_plots, sorted_scores
from social_choice_rlhf.lm.rlhf.loss import (
    IRCallInput,
    ListLossOutput,
    PairwiseInput,
    PartialInput,
    list_loss_fn,
    pairwise_loss_fn,
    stack_partial_input,
)
from social_choice_rlhf.lm.rlhf.social_choice.core import borda_count, random_ballot_swf
from social_choice_rlhf.lm.rlhf.social_choice.tn import *  # noqa: F403
from social_choice_rlhf.lm.rlhf.social_choice.types import Complete, OrderedPartition, Population
from social_choice_rlhf.util.jax import arrays_of
from social_choice_rlhf.util.misc import batched, declare_axis, unzip_list
from social_choice_rlhf.util.train import StepFn, mk_step, mk_stop_fn, train

if TYPE_CHECKING:
    from numpy import Fin

One: TypeAlias = Literal[1]
A = TypeVar("A")
B = TypeVar("B")
Prompt = TypeVar("Prompt")
Completion = TypeVar("Completion")
Float = TypeVar("Float", bound=float)
VocabSize = TypeVar("VocabSize", bound=int)
PromptLen = TypeVar("PromptLen", bound=int)
NumPrompts = TypeVar("NumPrompts", bound=int)
CompletionLen = TypeVar("CompletionLen", bound=int)
EmbedDim = TypeVar("EmbedDim", bound=int)
PrefLen = TypeVar("PrefLen", bound=int)
InPrefLen = TypeVar("InPrefLen", bound=int)
OutPrefLen = TypeVar("OutPrefLen", bound=int)
PrefDim = TypeVar("PrefDim", bound=int)
MaxSeqLen = TypeVar("MaxSeqLen", bound=int)
MaxPromptLen = TypeVar("MaxPromptLen", bound=int)
MaxPrefLen = TypeVar("MaxPrefLen", bound=int)
BatchLen = TypeVar("BatchLen", bound=int)
NumSamples = TypeVar("NumSamples", bound=int)
NumComponents = TypeVar("NumComponents", bound=int)
NumPeople = TypeVar("NumPeople", bound=int)

sns.set_theme()
mpl.rcParams["figure.autolayout"] = True

# =============================================================================
# Model that directly learns an ordering from pairwise comparisons
# =============================================================================

EmbedDim32: TypeAlias = Literal[32]
embed_dim_32: EmbedDim32 = 32
BatchLenA: TypeAlias = int
TrivialPairBatch: TypeAlias = PairwiseInput[BatchLenA, One, One, "Fin[TnTokenSize]"]
TnSocialModel: TypeAlias = SocialRewardModel[TnTokenSize, One, EmbedDim32, float32]


def mk_social_model() -> TnSocialModel:
    """Overparameterized for this problem, but jax compilation is the slowest part anyway"""
    embed_dim = embed_dim_32
    arch_config: ArchConfig[EmbedDim32, TnTokenSize, One, float32] = ArchConfig(
        layer_config=TransformerLayerConfig(
            q_dim=embed_dim,
            kv_dim=embed_dim,
            hidden_dim=embed_dim * 4,
            num_heads=1,
            norm_config=NormConfig(pos="norm_former", type_="layer_norm"),
            dropout_rate=float32(0),
            attention_dropout_rate=float32(0),
            attention_type=ScaledAttention(use_softmax1=False),
        ),
        vocab_size=8,
        num_layers=8,
        max_seq_len=1,
        pad_token_id=pad_token_id,
        tie_embeddings=False,
        stochastic_encoder=False,
    )
    return SocialRewardModel(arch_config, key=jax.random.PRNGKey(0))


def vanilla_reward(learn_pop: PromptPopulation[PosNeg, TnCity], eval_pop: PromptPopulation[PosNeg, TnCity]):
    vanilla_model = mk_social_model()
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adafactor(learning_rate=1e-3))
    vanilla_opt_state = tx.init(arrays_of(vanilla_model))
    step: StepFn[TnSocialModel, TrivialPairBatch, float32, tuple[()]] = mk_step(tx.update, pairwise_loss_fn)

    train_key = jax.random.PRNGKey(1)

    # social_order = borda_count(pop)
    # print(social_order.value)
    batch_iter = (mk_pair_batch(learn_pop, random_ballot_swf, batch_len=128, seed=s) for s in it.count())
    vanilla_model, vanilla_opt_state = train(
        vanilla_model,
        vanilla_opt_state,
        step,
        lambda: next(batch_iter),
        lambda x, _: {"loss": x},
        mk_stop_fn(lambda x: "stop" if mean(x) < 0.7 else "continue", lambda x, _: x, lookback_len=1000),  # noqa: PLR2004
        key=train_key,
    )
    print(
        "Proper Borda order",
        {
            prompt: borda_count(cast(Population[Complete, TnCity], eval_pop.pop_for_prompt(prompt)))
            for prompt in eval_pop.prompts()
        },
        sep="\n",
    )
    print(
        "Expected erroneous order",
        {
            prompt: borda_count(cast(Population[Complete, TnCity], learn_pop.pop_for_prompt(prompt)))
            for prompt in learn_pop.prompts()
        },
        sep="\n",
    )
    print(
        "Learned order",
        {
            prompt: sorted_scores(
                vanilla_model, np.array([prompt_to_token_id(prompt)]), tn_city_to_token, eval_pop.choices()
            )
            for prompt in eval_pop.prompts()
        },
        sep="\n",
    )
    return vanilla_model, vanilla_opt_state

    # run_pops()


# =============================================================================
# Model that learns to represent an ensemble of individual preferences
# =============================================================================

PrefDimL: TypeAlias = Literal[8]
pref_dim_l: PrefDimL = 8
EmbedDimL: TypeAlias = Literal[16]
embed_dim_l: EmbedDimL = 16

# Batch of int size with preferences over up to 5 cities with 1 token prompt and each city encoded as 1 token
TrivialIRBatch: TypeAlias = IRCallInput[
    BatchLenA, Literal[2], Literal[5], Literal[5], One, One, "Fin[TnTokenSize]"
]
TrivialIROutput: TypeAlias = ListLossOutput[
    BatchLenA, Literal[2], Ordered[Literal[5]], One, One, PrefDimL, EmbedDimL, float32
]
TnIRModel: TypeAlias = IndividualRewardModel[
    TnTokenSize, One, Literal[2], Literal[5], PrefDimL, EmbedDimL, float32
]
TnIRConfig: TypeAlias = IRConfig[TnTokenSize, One, Literal[2], Literal[5], PrefDimL, EmbedDimL, float32]


def get_ir_config() -> TnIRConfig:
    pref_layer_config: TransformerLayerConfig[PrefDimL, EmbedDimL, float32] = TransformerLayerConfig(
        q_dim=pref_dim_l,
        kv_dim=embed_dim_l,
        hidden_dim=embed_dim_l * 4,
        num_heads=1,
        norm_config=NormConfig(pos="norm_former", type_="layer_norm"),
        dropout_rate=float32(0),
        attention_dropout_rate=float32(0),
        attention_type=ScaledAttention(use_softmax1=False),
    )
    lm_layer_config: TransformerLayerConfig[EmbedDimL, EmbedDimL, float32] = TransformerLayerConfig(
        q_dim=embed_dim_l,
        kv_dim=embed_dim_l,
        hidden_dim=embed_dim_l * 4,
        num_heads=1,
        norm_config=NormConfig(pos="norm_former", type_="layer_norm"),
        dropout_rate=float32(0),
        attention_dropout_rate=float32(0),
        attention_type=ScaledAttention(use_softmax1=False),
    )
    pref_decoder_config: PrefDecoderConfig[PrefDimL, float32] = PrefDecoderConfig(
        num_layers=4,
        pref_dim=pref_dim_l,
        num_heads=1,
        hidden_dim=pref_dim_l * 4,
        norm_config=NormConfig(pos="norm_former", type_="layer_norm"),
    )
    return IRConfig(
        lm_layer_config,
        pref_layer_config,
        pref_decoder_config,
        num_lm_layers=4,
        num_pref_encoder_layers=8,
        vocab_size=8,
        max_pref_len=5,
        max_prompt_len=2,
        max_seq_len=1,
        pad_token_id=pad_token_id,
    )


def ir_postfix_fn(
    loss: Float,
    extra: ListLossOutput[
        BatchLen, NumPrompts, Ordered[PrefLen], PromptLen, CompletionLen, PrefDim, EmbedDim, Float
    ],
):
    return {
        "loss": loss,
        "list_mle": extra.list_mle_loss,
        "kl_div": extra.kl_div_loss,
        "std_dev": np.mean(extra.outputs.preference_output.std_dev),
        "mean": np.mean(np.abs(extra.outputs.preference_output.mean)),
    }


# =============================================================================
# Make a simulated population from a learned individual reward model
# and use this simulated population to generate a social preference order
# for any given prompt and completions
# =============================================================================


@eqx.filter_jit
def _mk_samples(
    model: IndividualRewardModel[VocabSize, MaxSeqLen, MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float],
    batch: IRCallInput[BatchLen, NumPrompts, InPrefLen, OutPrefLen, PromptLen, CompletionLen, Fin[VocabSize]],
    *,
    key: jax.Array,
) -> ndarray[BatchLen, PrefDim, Float]:
    """Generate a sample of preference representations in the latent space (i.e. a set of "voters")"""

    def single(
        prompt: ndarray[NumPrompts, PromptLen, Fin[VocabSize]],
        completions: ndarray[NumPrompts, Ordered[PrefLen], CompletionLen, Fin[VocabSize]],
        key: jax.Array,
    ) -> ndarray[PrefDim, Float]:
        return model.mk_pref_rep(prompt, completions, key=key).preference_output.sample

    return jax.vmap(single)(
        batch.prompt,
        batch.in_completions,
        jax.random.split(key, num=batch.prompt.shape[0]),
    )


def _decoded_samples_via_gmm(
    model: PreferenceModel[MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float],
    preference_samples: ndarray[NumSamples, PrefDim, Float],
    *,
    num_components: NumComponents,
    covariance_type: Literal["full", "tied", "diag", "spherical"],
    num_people: NumPeople,
) -> tuple[GaussianMixture[NumComponents, PrefDim], ndarray[NumPeople, Decoded[PrefDim], Float]]:
    gmm: GaussianMixture[NumComponents, PrefDim] = GaussianMixture(num_components, covariance_type=covariance_type)
    gmm.fit(preference_samples)
    samples = gmm.sample(num_people)[0].astype(preference_samples.dtype)
    return gmm, jax.vmap(model.decode_sample)(samples)


def mk_simulated_pop(  # noqa: PLR0913
    model: IndividualRewardModel[VocabSize, MaxSeqLen, MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float],
    population_sample_source: Sequence[
        IRCallInput[BatchLen, NumPrompts, InPrefLen, OutPrefLen, PromptLen, CompletionLen, Fin[VocabSize]]
    ],
    *,
    num_components: NumComponents,
    covariance_type: Literal["full", "tied", "spherical", "diag"],
    num_people: NumPeople,
    key: jax.Array,
):
    """Make a simulated population of "voters" by:
    1. Generating a set of preference representations in the latent space from a set of prompts and completions
    2. Doing ex post density estimation on the generated preference representations
    3. Sampling from the estimated density to get a set of "voters"
    with a density approximating the original population
    4. Decoding these samples to avoid repeating this work later
    We could arguably jump straight from 1 to 4, but sampling from an estimated density doesn't
    add much complexity and gives us additional flexibility and control.
    """
    samples = np.concatenate(
        [
            _mk_samples(model, batch, key=k)
            for batch, k in zip(
                population_sample_source,
                jax.random.split(key, len(population_sample_source)),
                strict=True,
            )
        ],
        axis=0,
    )

    return _decoded_samples_via_gmm(
        model.preference_model,
        samples,
        num_components=num_components,
        covariance_type=covariance_type,
        num_people=num_people,
    )


@eqx.filter_jit
def _pop_rewards(
    model: IndividualRewardModel[VocabSize, MaxSeqLen, MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float],
    samples: ndarray[NumPeople, Decoded[PrefDim], Float],
    prompt: ndarray[PromptLen, Fin[VocabSize]],
    completions: ndarray[PrefLen, CompletionLen, Fin[VocabSize]],
) -> ndarray[NumPeople, PrefLen, Float]:
    prompt_embs = model._interp_prompt(prompt)  # pyright: ignore[reportPrivateUsage]
    completion_embs = model._interp_completions(prompt_embs, prompt, completions)  # pyright: ignore[reportPrivateUsage]

    def reward(pref: ndarray[Decoded[PrefDim], Float]):
        return model.rewards(pref, completions, completion_embs)

    return jax.vmap(reward)(samples)


def social_order_from_individuals(
    model: IndividualRewardModel[VocabSize, MaxSeqLen, MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float],
    population: ndarray[NumPeople, Decoded[PrefDim], Float],
    data: PartialInput[BatchLen, PrefLen, PromptLen, CompletionLen, Fin[VocabSize]],
    swf: RnSocialWelfareFunction[Complete, int],
    *,
    key: jax.Array,
) -> Sequence[OrderedPartition[ndarray[CompletionLen, Fin[VocabSize]]]]:
    """Generate a social order from the simulated `population`'s preferences via `swf`"""

    def inner(
        prompt: ndarray[PromptLen, Fin[VocabSize]],
        completions: ndarray[PrefLen, CompletionLen, Fin[VocabSize]],
    ):
        return _pop_rewards(model, population, prompt, completions)

    rewards = jax.vmap(inner)(data.prompt, data.completions)
    return [
        swf_via_surrogate(r, list(c), swf, key=k)
        for r, c, k in zip(
            np.array(rewards),
            np.array(data.completions),
            np.array(jax.random.split(key, num=rewards.shape[0])),
            strict=True,
        )
    ]


# =============================================================================
# Putting all the pieces together to learn a social reward in two stages:
# 1. Learn an ensemble of individual "personas" from pairwise comparisons
# 2. Learn a social reward from the ensemble and a SWF
# =============================================================================


def to_batch_(x: Sequence[Profile[PosNeg, TnCity]], *, probs: ObscureProbs | None, seed: int) -> TrivialIRBatch:
    return to_batch(
        [maybe_obscure(y.shuffled_prompts(seed=s), probs) for s, y in enumerate(x, seed)],
        completion_tokenize=lambda y: (
            np.array((pad_token_id,)) if isinstance(y, Mask) else np.array((tn_city_to_token(y),))
        ),
        prompt_tokenize=lambda y: np.array((prompt_to_token_id(y),)),
        max_num_prompts=2,
        max_prompt_len=1,
        max_pref_len=Ordered(5),
        max_completion_len=1,
        pad_token_id=pad_token_id,
    )


def maybe_obscure(x: Profile[B, A], probs: ObscureProbs | None):
    prompts, orders = unzip_list(list(x.items()))
    obscured = (
        [Completions(o.flat(), o.flat()) for o in orders]
        if probs is None
        else obscure_ordered_partitions(orders, probs)
    )
    return list(zip(prompts, obscured, strict=True))


def print_batch(x: Sequence[Profile[PosNeg, A]], probs: ObscureProbs | None):
    print(*[maybe_obscure(y.shuffled_prompts(seed=s), probs) for s, y in enumerate(x)], sep="\n")


def mk_repeat_cb(repeat_ref: Ref[float], *, lookback_len: int):
    losses = deque[float32](maxlen=lookback_len)
    count = 0

    def inner(loss: float32):
        nonlocal count
        count += 1
        losses.append(loss)
        if count % 100 == 0 and len(losses) == lookback_len and has_stabilized(losses):
            repeat_ref.value += 0.1
            print(f"Incremented repeat to {repeat_ref.value}")
            losses.clear()

    return inner


def train_ir_model(learn_pop: PromptPopulation[PosNeg, TnCity]):
    ir_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adafactor(learning_rate=5e-3))

    beta = float32(0.1)

    def list_loss_fn_(model: TnIRModel, input_: TrivialIRBatch, key: jax.Array):
        return list_loss_fn(model, input_, beta, pad_token_id, key)

    ir_config = get_ir_config()
    ir_model = IndividualRewardModel(ir_config, key=jax.random.PRNGKey(0))
    ir_opt_state = ir_tx.init(arrays_of(ir_model))
    ir_step: StepFn[TnIRModel, TrivialIRBatch, float32, TrivialIROutput] = mk_step(ir_tx.update, list_loss_fn_)

    obscure_probs = ObscureProbs(mask=0.5, dummy=0.2, repeat=Ref(0.0))
    ir_batch_iter = (
        to_batch_(x, probs=obscure_probs, seed=s)
        for x, s in zip(batched(learn_pop.iter_profiles(seed=0), 128), it.count(3))
    )
    print_batch(next(batched(learn_pop.iter_profiles(seed=0), 128)), obscure_probs)

    # example_threshold = 1.0
    # morristown_threshold = 2.0
    prompt_generalization_threshold = 2.0
    ir_model, ir_opt_state = train(
        ir_model,
        ir_opt_state,
        ir_step,
        lambda: next(ir_batch_iter),
        ir_postfix_fn,
        mk_stop_fn(
            lambda x: "stop"
            if obscure_probs.repeat.value > 1 and mean(x) < prompt_generalization_threshold
            else "continue",
            extractor=lambda _, extra: extra.list_mle_loss.item(),
            lookback_len=300,
        ),
        [mk_repeat_cb(obscure_probs.repeat, lookback_len=900)],
        key=jax.random.PRNGKey(0),
    )

    return ir_model, ir_opt_state


def train_social_model(
    ir_model: TnIRModel,
    learn_pop: PromptPopulation[PosNeg, TnCity],
    eval_pop: PromptPopulation[PosNeg, TnCity],
):
    gmm, pop = mk_simulated_pop(
        ir_model,
        [
            to_batch_(x, probs=None, seed=s)
            for s, x in enumerate(it.islice(batched(learn_pop.iter_profiles(seed=1), 128), 100), 5)
        ],
        num_components=10,
        covariance_type="diag",
        num_people=1000,
        key=jax.random.PRNGKey(0),
    )
    print(gmm.weights_)
    social_order_iter = (
        (
            stack_partial_input(input_),
            social_order_from_individuals(
                ir_model,
                np.array(pop),
                stack_partial_input(input_),
                swf=lambda pop, _: borda_count(pop),
                key=jax.random.PRNGKey(0),
            ),
        )
        for input_ in batched(
            it.cycle(
                [
                    PartialInput(
                        np.array((prompt_to_token_id(p),)),
                        np.array([(tn_city_to_token(c),) for c in learn_pop.choices()]),
                    )
                    for p in learn_pop.prompts()
                ],
            ),
            128,
        )
    )
    pair_batch_iter = (pair_batch_from_order_batch(input_.prompt, orders) for input_, orders in social_order_iter)

    social_model = mk_social_model()
    social_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adafactor(learning_rate=1e-3))
    social_opt_state = social_tx.init(arrays_of(social_model))
    social_step: StepFn[TnSocialModel, TrivialPairBatch, float32, tuple[()]] = mk_step(
        social_tx.update, pairwise_loss_fn
    )
    threshold = 0.7
    social_model, social_opt_state = train(
        social_model,
        social_opt_state,
        social_step,
        lambda: next(pair_batch_iter),
        lambda x, _: {"loss": x},
        mk_stop_fn(
            lambda x: "stop" if mean(x) < threshold else "continue",
            extractor=lambda loss, _: loss,
            lookback_len=100,
        ),
        key=jax.random.PRNGKey(0),
    )
    print(
        "Borda count",
        {
            prompt: borda_count(cast(Population[Complete, TnCity], eval_pop.pop_for_prompt(prompt)))
            for prompt in eval_pop.prompts()
        },
        sep="\n",
    )
    print(
        "Learned order",
        {
            prompt: sorted_scores(
                social_model,
                np.array([prompt_to_token_id(prompt)]),
                tn_city_to_token,
                eval_pop.choices(),
            )
            for prompt in eval_pop.prompts()
        },
        sep="\n",
    )


# =============================================================================
# Miscellaneous helpers
# =============================================================================


def call_model(  # noqa: PLR0913
    model: IndividualRewardModel[VocabSize, MaxSeqLen, MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float],
    prompt: Prompt,
    in_completions: Sequence[Completion | Mask],
    out_completions: Sequence[Completion | Mask],
    prompt_tokenize: Callable[[Prompt], ndarray[PromptLen, Fin[VocabSize]]],
    completion_tokenize: Callable[[Completion | Mask], ndarray[CompletionLen, Fin[VocabSize]]],
):
    in_toks = declare_axis[Ordered[int]](
        0, np.stack([completion_tokenize(completion) for completion in in_completions])
    )
    out_toks = np.stack([completion_tokenize(completion) for completion in out_completions])
    return model.__call__(
        np.expand_dims(prompt_tokenize(prompt), axis=0),
        np.expand_dims(in_toks, axis=0),
        np.expand_dims(out_toks, axis=0),
        key=jax.random.PRNGKey(0),
    )


def call_tn_model(
    model: IndividualRewardModel[TnTokenSize, MaxSeqLen, MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float],
    prompt: PosNeg,
    in_completions: Sequence[TnCity | Mask],
    out_completions: Sequence[TnCity | Mask],
):
    return call_model(
        model,
        prompt,
        in_completions,
        out_completions,
        prompt_tokenize=lambda x: np.array((prompt_to_token_id(x),)),
        completion_tokenize=lambda x: np.array((pad_token_id if isinstance(x, Mask) else tn_city_to_token(x),)),
    )


def pop_from_voter_samples(
    model: IndividualRewardModel[VocabSize, MaxSeqLen, MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float],
    pop: ndarray[NumPeople, Decoded[PrefDim], Float],
    prompt: ndarray[PromptLen, Fin[VocabSize]],
    completions: ndarray[PrefLen, CompletionLen, Fin[VocabSize]],
    completion_to_label: Callable[[ndarray[CompletionLen, Fin[VocabSize]]], A],
) -> Population[Complete, A]:
    """Summarize the simulated population's preferences on the provided `completions` in the context of `prompt`
    (`completion_to_label` is required because `ndarray`s aren't hashable)
    """
    rewards = _pop_rewards(model, pop, prompt, completions)
    return Population.orders_to_complete(
        rewards_to_orders(np.array(rewards), [completion_to_label(c) for c in completions])
    )


def swf_via_surrogate(
    rewards: ndarray[NumSamples, PrefLen, Float],
    flat_order: Sequence[A],
    swf: RnSocialWelfareFunction[Complete, int],
    *,
    key: jax.Array,
) -> OrderedPartition[A]:
    """For when our items aren't hashable"""
    order_mapping = dict(enumerate(flat_order))
    # Putting the rewards array back on the CPU provides a huge speedup
    pop = Population.orders_to_complete(rewards_to_orders(np.array(rewards), list(order_mapping.keys())))
    order = swf(pop, key)
    return OrderedPartition.mk_simple(tuple(order_mapping[i[0]] for i in order.value))


def rewards_to_orders(rewards: ndarray[NumSamples, PrefLen, Float], flat_order: Sequence[A]):
    assert rewards.shape[1] == len(flat_order)
    indices = np.argsort(rewards, axis=1)[..., ::-1]
    return [OrderedPartition.mk_simple(tuple(flat_order[i] for i in x)) for x in indices]


def has_stabilized(x: Sequence[Float]) -> ndarray[bool]:
    first_third, _, third_third = np.split(np.array(x), 3)
    return np.percentile(first_third, 50) <= np.percentile(third_third, 55)


def check_ir_model(ir_model: TnIRModel, eval_pop: PromptPopulation[PosNeg, TnCity]):
    check_learned_orders(
        ir_model,
        eval_pop,
        tokenize_completion=lambda x: np.array([tn_city_to_token(x)]),
        tokenize_prompt=lambda x: np.array([prompt_to_token_id(x)]),
    )
    nested_labels, nested_samples = unzip_list(
        [
            (
                x,
                _mk_samples(ir_model, to_batch_(x, probs=None, seed=s), key=k),
            )
            for x, s, k in zip(
                batched(eval_pop.iter_profiles(seed=3), 128),
                it.count(8),
                jax.random.split(jax.random.PRNGKey(0), 8),
            )
        ]
    )
    labels, samples = flatten(nested_labels), np.concatenate(nested_samples, axis=0)

    def to_str(x: Profile[PosNeg, TnCity]):
        def fn(prompt: PosNeg, order: OrderedPartition[TnCity]):
            prompt_str = "Nm" if isinstance(prompt, Pos) else "Bz"
            return f"{prompt_str} {'≻'.join([to_short_str(o) for o in order.flat()])}"

        return "\n".join(fn(*y) for y in x.items())

    nd_to_2d_plots(samples, labels, lambda x: to_str(x))
