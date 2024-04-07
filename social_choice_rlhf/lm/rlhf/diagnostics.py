from __future__ import annotations

import itertools as it
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar, cast

import jax
import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import float32, ndarray
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding

from social_choice_rlhf.lm.rlhf.architecture import IndividualRewardModel, Ordered, SocialRewardModel
from social_choice_rlhf.lm.rlhf.data import Mask, PromptPopulation
from social_choice_rlhf.lm.rlhf.social_choice.types import OrderedPartition
from social_choice_rlhf.util.misc import declare_axes, declare_axis

if TYPE_CHECKING:
    from numpy import Fin

Two: TypeAlias = Literal[2]
Label = TypeVar("Label")
Float = TypeVar("Float", bound=float)
VocabSize = TypeVar("VocabSize", bound=int)
PromptLen = TypeVar("PromptLen", bound=int)
CompletionLen = TypeVar("CompletionLen", bound=int)
EmbedDim = TypeVar("EmbedDim", bound=int)
PrefDim = TypeVar("PrefDim", bound=int)
SeqLen = TypeVar("SeqLen", bound=int)
MaxSeqLen = TypeVar("MaxSeqLen", bound=int)
MaxPromptLen = TypeVar("MaxPromptLen", bound=int)
MaxPrefLen = TypeVar("MaxPrefLen", bound=int)
NumSamples = TypeVar("NumSamples", bound=int)
NumFeatures = TypeVar("NumFeatures", bound=int)
Prompt = TypeVar("Prompt")
Completion = TypeVar("Completion")
Choice = TypeVar("Choice")


def nd_to_2d_plots(
    preference_samples: ndarray[NumSamples, PrefDim, Float],
    preference_labels: Sequence[Label],
    label_to_str: Callable[[Label], str],
    reducers: Sequence[Literal["iso", "spectral", "tsne", "pca"]] | None = None,
):
    """For visualizing the latent space of learned preference representations"""
    assert preference_samples.shape[0] == len(preference_labels)
    # One of them (`Isomap`, IIRC) doesn't work with bfloat16
    preference_samples_f32 = preference_samples.astype(float32)

    def iso(x: ndarray[NumSamples, NumFeatures, float32]) -> ndarray[NumSamples, Two, float32]:
        with warnings.catch_warnings():
            # SparseEfficiencyWarning
            warnings.simplefilter("ignore")
            return Isomap(n_components=2).fit_transform(cast(Any, x))

    reducers_ = {
        k: v
        for k, v in {
            "iso": iso,
            "spectral": SpectralEmbedding(n_components=2).fit_transform,
            "tsne": TSNE(n_components=2).fit_transform,
            "pca": PCA(n_components=2).fit_transform,
        }.items()
        if reducers is None or k in reducers
    }

    return _plot_reductions(preference_samples_f32, preference_labels, cast(Any, reducers_), label_to_str)


def _plot_reductions(
    x: ndarray[NumSamples, NumFeatures, Float],
    y: Sequence[Label],
    reduce_dims: Mapping[
        str, Callable[[ndarray[NumSamples, NumFeatures, Float]], ndarray[NumSamples, Two, Float]]
    ],
    y_to_str: Callable[[Label], str],
):
    assert x.shape[0] == len(y)
    indices_by_label = {
        y_to_str(label): declare_axes[NumSamples](np.array([i == label for i in y])) for label in set(y)
    }
    num_reducers = len(reduce_dims)
    fig, ax = plt.subplots(figsize=(7, 7 * num_reducers), nrows=num_reducers)
    ax = ax if num_reducers > 1 else declare_axes[int](np.expand_dims(ax, axis=0))
    for i, (name, reduce) in enumerate(reduce_dims.items()):
        x_2 = reduce(x)
        _plot_2d(x_2, indices_by_label, ax[i])
        ax[i].set_title(name)
    return fig, ax


def _plot_2d(
    preference_samples: ndarray[NumSamples, Two, Float],
    indices_by_label: Mapping[str, ndarray[NumSamples, bool]],
    ax: Axes,
):
    for color, (label, indices), marker in zip(
        it.cycle(mpl.cm.tab10.colors),
        indices_by_label.items(),
        it.cycle(["v", "^", "<", ">", "1", "2", "3", "4", "P", "*", "D"]),
    ):
        ax.scatter(
            preference_samples[indices, 0],
            preference_samples[indices, 1],
            label=label,
            color=color,
            marker=cast(Any, marker),
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.legend()
        ax.legend()


def single_unmasked(order: OrderedPartition[Choice]) -> Sequence[Sequence[Choice | Mask]]:
    flat_order = order.flat()

    def mk_dummy(i: int):
        return tuple(Mask() if j != i else flat_order[i] for j in range(len(flat_order)))

    return [mk_dummy(i) for i in range(len(flat_order))]


def all_masking_patterns(order: OrderedPartition[Choice]) -> Sequence[Sequence[Choice | Mask]]:
    flat_order = order.flat()

    masks = it.product([True, False], repeat=len(flat_order))

    return [tuple(Mask() if mask else choice for mask, choice in zip(m, flat_order)) for m in masks]


def embed_and_plot(
    model: IndividualRewardModel[VocabSize, MaxSeqLen, MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float],
    prompt: ndarray[PromptLen, Fin[VocabSize]],
    orders: Sequence[Sequence[Choice | Mask]],
    tokenize: Callable[[Choice | Mask], ndarray[CompletionLen, Fin[VocabSize]]],
    to_str: Callable[[Choice], str],
):
    """Find the mean represenation of each (partial) order and plot the set of them in 2D via PCA"""

    def pref_rep(order: Sequence[Choice | Mask]):
        in_toks = declare_axis[Ordered[int]](0, np.stack([tokenize(completion) for completion in order]))
        return model.mk_pref_rep(
            np.expand_dims(prompt, axis=0),
            np.expand_dims(in_toks, axis=0),
            key=jax.random.PRNGKey(0),
        ).preference_output.mean

    samples = [pref_rep(order) for order in orders]

    def to_str_(x: Sequence[Choice | Mask]):
        return "≻".join([to_str(c) for c in x if not isinstance(c, Mask)])

    return nd_to_2d_plots(np.stack(samples), orders, to_str_, ["pca"])


def latent_traversal(  # noqa: PLR0913
    model: IndividualRewardModel[VocabSize, MaxSeqLen, MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float],
    prompt: ndarray[PromptLen, Fin[VocabSize]],
    tokenize: Callable[[Choice], ndarray[CompletionLen, Fin[VocabSize]]],
    *,
    start: OrderedPartition[Choice],
    end: OrderedPartition[Choice],
    num_steps: NumSamples,
) -> tuple[ndarray[NumSamples, PrefDim, Float], Sequence[Mapping[Choice, Float]]]:
    """Traverse the latent space between two preference orders.
    Returns the preference representations and their rewards on `Choice`s for each step.
    """
    flat_start = start.flat()
    flat_end = end.flat()
    start_toks = declare_axis[Ordered[int]](0, np.stack([tokenize(c) for c in flat_start]))
    end_toks = declare_axis[Ordered[int]](0, np.stack([tokenize(c) for c in flat_end]))
    assert set(flat_start) == set(flat_end)
    start_rep = model.mk_pref_rep(
        np.expand_dims(prompt, axis=0),
        np.expand_dims(start_toks, axis=0),
        key=jax.random.PRNGKey(0),
    ).preference_output.mean
    end_rep = model.mk_pref_rep(
        np.expand_dims(prompt, axis=0),
        np.expand_dims(end_toks, axis=0),
        key=jax.random.PRNGKey(0),
    ).preference_output.mean
    blends = declare_axis[NumSamples](
        0, np.stack([start_rep * (1 - step) + end_rep * step for step in np.linspace(0, 1, num_steps)])
    )

    def fn(pref_rep: ndarray[PrefDim, Float]):
        return model.from_sample(prompt, start_toks, pref_rep).rewards

    rewards = [dict(zip(flat_start, [np.array(r).item() for r in rs], strict=True)) for rs in jax.vmap(fn)(blends)]
    return blends, rewards


def plot_traversal(
    x: Sequence[Mapping[Choice, Float]],
    *,
    start: OrderedPartition[Choice],
    end: OrderedPartition[Choice],
    to_str: Callable[[Choice], str] = str,
):
    diffs = {(l, r): [m[l] - m[r] for m in x] for l, r in it.combinations(list(x[0].keys()), 2)}
    fig, ax = plt.subplots(figsize=(7, 7))
    for (l, r), y in diffs.items():
        sns.lineplot(x=[i / len(y) for i in range(len(y))], y=y, label=f"{to_str(l)} - {to_str(r)}", ax=ax)
    start_str = "≻".join([to_str(c) for c in start.flat()])
    end_str = "≻".join([to_str(c) for c in end.flat()])
    ax.set_xlabel(f"From {start_str} to {end_str}")
    return fig, ax


def sorted_scores(
    model: SocialRewardModel[VocabSize, MaxSeqLen, EmbedDim, Float],
    prompt: ndarray[SeqLen, Fin[VocabSize]],
    choice_to_token: Callable[[Choice], Fin[VocabSize]],
    choices: set[Choice],
):
    return sorted(
        {
            choice: model.__call__(prompt=prompt, completion=np.array([choice_to_token(choice)])).item()
            for choice in choices
        }.items(),
        key=lambda x: x[1],
        reverse=True,
    )


def check_learned_orders(
    model: IndividualRewardModel[VocabSize, MaxSeqLen, MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float],
    pop: PromptPopulation[Prompt, Completion],
    tokenize_prompt: Callable[[Prompt], ndarray[PromptLen, Fin[VocabSize]]],
    tokenize_completion: Callable[[Completion], ndarray[CompletionLen, Fin[VocabSize]]],
):
    """Check that our model is able to faithfully learn and represent distinct individual preference orders"""
    for profile in pop.profiles:
        for prompt, order in profile.items():
            flat_order = order.flat()
            completion_toks = declare_axis[Ordered[int]](
                0, np.stack([tokenize_completion(completion) for completion in flat_order])
            )
            prompt_toks = tokenize_prompt(prompt)
            learned_order = [
                (l, np.array(x).item())
                for l, x in zip(
                    flat_order,
                    np.squeeze(
                        model.__call__(
                            np.expand_dims(prompt_toks, axis=0),
                            np.expand_dims(completion_toks, axis=0),
                            np.expand_dims(completion_toks, axis=0),
                            key=jax.random.PRNGKey(0),
                        ).rewards,
                        axis=0,
                    ),
                    strict=True,
                )
            ]
            print(learned_order)
            print(flat_order == tuple(x[0] for x in sorted(learned_order, key=lambda x: x[1], reverse=True)))
