from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, NamedTuple, TypeAlias, TypeVar, TypeVarTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from numpy import ndarray

from social_choice_rlhf.lm.architecture import (
    ArchConfig,
    Decoder,
    EmbeddingDecoder,
    EmbeddingEncoder,
    FeedForward,
    MaxSeqLen,
    NoMask,
    NormConfig,
    Sampler,
    TransformerLayerConfig,
)
from social_choice_rlhf.util.jax import scan_layers
from social_choice_rlhf.util.misc import InstanceSingleton, declare_axis, flatten_product, product_, sum_

if TYPE_CHECKING:
    from jax import AuxDim
    from numpy import Fin, Product, Sum

One: TypeAlias = Literal[1]
A = TypeVar("A")
Left = TypeVar("Left", bound=int)
Right = TypeVar("Right", bound=int)
VocabSize = TypeVar("VocabSize", bound=int)
EmbedDim = TypeVar("EmbedDim", bound=int)
PrefDim = TypeVar("PrefDim", bound=int)
InPrefLen = TypeVar("InPrefLen", bound=int)
OutPrefLen = TypeVar("OutPrefLen", bound=int)
MaxPrefLen = TypeVar("MaxPrefLen", bound=int)
MaxPromptLen = TypeVar("MaxPromptLen", bound=int)
PromptLen = TypeVar("PromptLen", bound=int)
CompletionLen = TypeVar("CompletionLen", bound=int)
Float = TypeVar("Float", bound=float)
DType = TypeVar("DType")
Shape = TypeVarTuple("Shape")
SeqLen = TypeVar("SeqLen", bound=int)
QDim = TypeVar("QDim", bound=int)
KVDim = TypeVar("KVDim", bound=int)
QSeqLen = TypeVar("QSeqLen", bound=int)
KVSeqLen = TypeVar("KVSeqLen", bound=int)
DType = TypeVar("DType")
Dim1 = TypeVar("Dim1", bound=int)
NumPrompts = TypeVar("NumPrompts", bound=int)


class Decoded(int, Generic[A]):
    def __new__(cls, value: A) -> Decoded[A]:
        return cast(Any, value)


class Ordered(int, Generic[A]):
    def __new__(cls, value: A) -> Ordered[A]:
        return cast(Any, value)


class SocialRewardModel(eqx.Module, Generic[VocabSize, MaxSeqLen, EmbedDim, Float]):
    encoder: EmbeddingEncoder[VocabSize, MaxSeqLen, EmbedDim, Float]
    decoder: EmbeddingDecoder[VocabSize, MaxSeqLen, EmbedDim, Float]
    reward: eqx.nn.Linear[EmbedDim, One, Float]
    pad_token_id: Fin[VocabSize] = eqx.field(static=True)

    def __init__(
        self,
        config: ArchConfig[EmbedDim, VocabSize, MaxSeqLen, Float],
        *,
        key: jax.Array,
    ):
        self.pad_token_id = config.pad_token_id
        encoder_key, decoder_key, rm_key = jax.random.split(key, 3)
        self.encoder = EmbeddingEncoder(
            layer_config=config.layer_config,
            num_layers=config.num_layers,
            pad_token_id=config.pad_token_id,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            key=encoder_key,
        )
        self.decoder = EmbeddingDecoder(
            layer_config=config.layer_config,
            num_layers=config.num_layers,
            pad_token_id=config.pad_token_id,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            mask_type=NoMask(),
            key=decoder_key,
        )
        self.reward = eqx.nn.Linear(in_features=config.layer_config.q_dim, out_features=1, key=rm_key)

    def __call__(
        self,
        prompt: ndarray[PromptLen, Fin[VocabSize]],
        completion: ndarray[CompletionLen, Fin[VocabSize]],
    ) -> ndarray[Float]:
        enc_out = self.encoder.__call__(prompt)
        dec_out = self.decoder.__call__(completion, enc_out, (prompt != self.pad_token_id))
        return jnp.mean(jax.vmap(self.reward)(dec_out)[..., 0], where=completion != self.pad_token_id)


class PreferenceOutput(NamedTuple, Generic[*Shape, PrefDim, Float]):
    mean: ndarray[*Shape, PrefDim, Float]
    std_dev: ndarray[*Shape, Float]
    sample: ndarray[*Shape, PrefDim, Float]


class FullPreferenceOutput(NamedTuple, Generic[*Shape, PrefDim, Float]):
    mean: ndarray[*Shape, PrefDim, Float]
    std_dev: ndarray[*Shape, Float]
    sample: ndarray[*Shape, PrefDim, Float]
    dec_out: ndarray[*Shape, Decoded[PrefDim], Float]


@dataclass(frozen=True)
class PrefDecoderConfig(Generic[PrefDim, Float]):
    num_layers: int
    pref_dim: PrefDim
    hidden_dim: int
    num_heads: int
    norm_config: NormConfig


class PreferenceModel(eqx.Module, Generic[MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float]):
    NumDecoderLayers: TypeAlias = InstanceSingleton[Literal["NumDecoderLayers"]]
    completion_position_embedder: eqx.nn.Embedding[MaxPrefLen, EmbedDim, Float]
    completion_position_embed_norm: eqx.nn.LayerNorm[EmbedDim, Float]
    # Sort of silly at the moment but useful in the future if/when
    # we accept multiple prompt-completion sets simultaneously.
    prompt_position_embedder: eqx.nn.Embedding[MaxPromptLen, EmbedDim, Float]
    prompt_position_embed_norm: eqx.nn.LayerNorm[EmbedDim, Float]
    # Sort of weird that our encoder is a decoder but, in this case,
    # we have the preference ordering as the queries and the prompt as the KVs that the decoder
    # cross-attends to.
    encoder: Decoder[PrefDim, EmbedDim, Float]
    sampler: Sampler[Float]
    decoder: AuxDim[PreferenceModel.NumDecoderLayers, FeedForward[PrefDim, Float]]
    decoder_norm: eqx.nn.LayerNorm[PrefDim, Float]
    dtype: type[Float] = eqx.field(static=True)
    pref_dim: PrefDim = eqx.field(static=True)

    def __init__(  # noqa: PLR0913
        self,
        encoder_config: TransformerLayerConfig[PrefDim, EmbedDim, Float],
        decoder_config: PrefDecoderConfig[PrefDim, Float],
        *,
        key: jax.Array,
        max_pref_len: MaxPrefLen,
        max_prompt_len: MaxPromptLen,
        num_encoder_layers: int,
    ):
        encoder_key, completion_embed_key, pos_embed_key, decoder_key = jax.random.split(key, num=4)
        self.encoder = Decoder(encoder_config, num_layers=num_encoder_layers, key=encoder_key, mask_type=NoMask())
        self.completion_position_embedder = eqx.nn.Embedding(
            num_embeddings=max_pref_len,
            embedding_size=encoder_config.kv_dim,
            key=completion_embed_key,
        )
        self.sampler = Sampler(dtype=type(encoder_config.dropout_rate))
        self.decoder_norm = eqx.nn.LayerNorm(shape=decoder_config.pref_dim)
        self.completion_position_embed_norm = eqx.nn.LayerNorm(shape=encoder_config.kv_dim)
        self.prompt_position_embed_norm = eqx.nn.LayerNorm(shape=encoder_config.kv_dim)
        self.prompt_position_embedder = eqx.nn.Embedding(
            num_embeddings=max_prompt_len, embedding_size=encoder_config.kv_dim, key=pos_embed_key
        )

        def mk_decoder_layer(layer_key: jax.Array) -> FeedForward[PrefDim, Float]:
            return FeedForward(
                embed_dim=decoder_config.pref_dim,
                hidden_dim=decoder_config.hidden_dim,
                norm_config=decoder_config.norm_config,
                key=layer_key,
            )

        self.decoder = eqx.filter_vmap(mk_decoder_layer)(
            jax.random.split(
                decoder_key, num=InstanceSingleton(self, "NumDecoderLayers", decoder_config.num_layers)
            )
        )
        self.pref_dim = decoder_config.pref_dim
        self.dtype = type(encoder_config.dropout_rate)

    def __call__(  # noqa: PLR0913
        self,
        prompt: ndarray[NumPrompts, PromptLen, EmbedDim, Float],
        prompt_padding_mask: ndarray[NumPrompts, PromptLen, bool],
        completion_embeddings: ndarray[NumPrompts, Ordered[InPrefLen], CompletionLen, EmbedDim, Float],
        completion_mask: ndarray[NumPrompts, Ordered[InPrefLen], CompletionLen, bool],
        *,
        key: jax.Array,
    ) -> FullPreferenceOutput[PrefDim, Float]:
        mean, std_dev, sample = self.mk_pref_rep(
            prompt,
            prompt_padding_mask,
            completion_embeddings,
            completion_mask,
            key=key,
        )
        decoded_sample = self.decode_sample(sample)

        return FullPreferenceOutput(mean, std_dev, sample, decoded_sample)

    def _flatten_completions_for_prompt(
        self, completions: ndarray[Ordered[InPrefLen], CompletionLen, EmbedDim, Float]
    ) -> ndarray[Product[Ordered[InPrefLen], CompletionLen], EmbedDim, Float]:
        def add_pos(
            completion: ndarray[Dim1, EmbedDim, Float], pos: ndarray[Fin[Any]]
        ) -> ndarray[Dim1, EmbedDim, Float]:
            return completion + self.completion_position_embed_norm(self.completion_position_embedder(pos))

        completions_with_pos = jax.vmap(add_pos)(completions, jnp.arange(completions.shape[0]))
        return jnp.reshape(
            completions_with_pos,
            (
                product_((completions_with_pos.shape[0], completions_with_pos.shape[1])),
                completions_with_pos.shape[2],
            ),
        )

    def _flatten_completions(
        self,
        completions: ndarray[NumPrompts, Ordered[InPrefLen], CompletionLen, EmbedDim, Float],
    ) -> ndarray[Product[NumPrompts, Ordered[InPrefLen], CompletionLen], EmbedDim, Float]:
        flat_completions = jax.vmap(self._flatten_completions_for_prompt)(completions)

        def add_pos(
            flat_completion: ndarray[Dim1, EmbedDim, Float], pos: ndarray[Fin[Any]]
        ) -> ndarray[Dim1, EmbedDim, Float]:
            return flat_completion + self.prompt_position_embed_norm(self.prompt_position_embedder(pos))

        flat_completions_with_pos = jax.vmap(add_pos)(flat_completions, jnp.arange(flat_completions.shape[0]))

        return jnp.reshape(
            flat_completions_with_pos,
            (flatten_product(product_(flat_completions_with_pos.shape[:2])), flat_completions_with_pos.shape[2]),
        )

    @eqx.filter_jit
    def mk_pref_rep(  # noqa: PLR0913
        self,
        prompts: ndarray[NumPrompts, PromptLen, EmbedDim, Float],
        prompt_padding_mask: ndarray[NumPrompts, PromptLen, bool],
        completions: ndarray[NumPrompts, Ordered[InPrefLen], CompletionLen, EmbedDim, Float],
        completion_mask: ndarray[NumPrompts, Ordered[InPrefLen], CompletionLen, bool],
        *,
        key: jax.Array,
    ):
        flattened_completions = self._flatten_completions(completions)

        def add_pos(
            prompt: ndarray[Dim1, EmbedDim, Float], pos: ndarray[Fin[Any]]
        ) -> ndarray[Dim1, EmbedDim, Float]:
            return prompt + self.prompt_position_embed_norm(self.prompt_position_embedder(pos))

        prompts_with_pos = jax.vmap(add_pos)(prompts, jnp.arange(prompts.shape[0]))
        flat_prompts = jnp.reshape(
            prompts_with_pos,
            (product_((prompts_with_pos.shape[0], prompts_with_pos.shape[1])), prompts_with_pos.shape[2]),
        )
        preference_rep_mean = jnp.squeeze(
            self.encoder.__call__(
                jnp.zeros((1, self.pref_dim), dtype=prompts.dtype),
                jnp.concatenate((flat_prompts, flattened_completions), axis=0),
                key_value_padding_mask=jnp.concatenate(
                    (jnp.ravel(prompt_padding_mask), jnp.ravel(completion_mask)), axis=0
                ),
                query_padding_mask=jnp.ones(1, dtype=bool),
            ),
            axis=0,
        )

        pref_rep_sample = self.sampler.__call__(preference_rep_mean, key)

        return PreferenceOutput(preference_rep_mean, jnp.square(self.sampler.sqrt_std_dev), pref_rep_sample)

    @eqx.filter_jit
    def sample_and_decode(self, *, sampler_key: jax.Array) -> ndarray[Decoded[PrefDim], Float]:
        preference_rep_sample = self.sampler.__call__(jnp.zeros(self.pref_dim, dtype=self.dtype), sampler_key)
        return self.decode_sample(preference_rep_sample)

    @eqx.filter_jit
    def decode_sample(self, preference_rep_sample: ndarray[PrefDim, Float]) -> ndarray[Decoded[PrefDim], Float]:
        return declare_axis[Decoded[PrefDim]](
            0, self.decoder_norm(scan_layers(preference_rep_sample, initial_key=None, layers=self.decoder))
        )


class RewardLayer(eqx.Module, Generic[PrefDim, EmbedDim, Float]):
    _HiddenDim: TypeAlias = InstanceSingleton[Literal["HiddenDim"]]
    hidden: eqx.nn.Linear[Sum[PrefDim, EmbedDim], RewardLayer._HiddenDim, Float]
    reward: eqx.nn.Linear[RewardLayer._HiddenDim, One, Float]
    norm: eqx.nn.LayerNorm[RewardLayer._HiddenDim, Float]

    def __init__(
        self,
        *,
        pref_dim: PrefDim,
        embed_dim: EmbedDim,
        hidden_dim: int,
        key: jax.Array,
    ):
        hidden_dim_ = InstanceSingleton[Literal["HiddenDim"]](self, "HiddenDim", hidden_dim)
        self.hidden = eqx.nn.Linear(in_features=sum_((pref_dim, embed_dim)), out_features=hidden_dim_, key=key)
        self.norm = eqx.nn.LayerNorm(shape=hidden_dim_)
        self.reward = eqx.nn.Linear(in_features=hidden_dim_, out_features=1, key=key)

    def __call__(
        self,
        completion: ndarray[CompletionLen, EmbedDim, Float],
        mask: ndarray[CompletionLen, bool],
        preference_rep: ndarray[PrefDim, Float],
    ) -> ndarray[Float]:
        def inner(x: ndarray[EmbedDim, Float]) -> ndarray[One, Float]:
            return self.reward(self.norm(jax.nn.gelu(self.hidden(jnp.concatenate((preference_rep, x), axis=0)))))

        return jnp.mean(jax.vmap(inner)(completion)[..., 0], where=mask)


class IndividualRewardOutput(
    NamedTuple, Generic[*Shape, NumPrompts, OutPrefLen, PromptLen, CompletionLen, PrefDim, EmbedDim, Float]
):
    rewards: ndarray[*Shape, NumPrompts, OutPrefLen, Float]
    preference_output: FullPreferenceOutput[*Shape, PrefDim, Float]
    prompt_embs: ndarray[*Shape, NumPrompts, PromptLen, EmbedDim, Float]
    completion_embs: ndarray[*Shape, NumPrompts, OutPrefLen, CompletionLen, EmbedDim, Float]


class IndividualRewardSampleOutput(
    NamedTuple, Generic[InPrefLen, PromptLen, CompletionLen, VocabSize, PrefDim, EmbedDim, Float]
):
    rewards: ndarray[InPrefLen, Float]
    preference_rep: ndarray[Decoded[PrefDim], Float]
    prompt_embs: ndarray[PromptLen, EmbedDim, Float]
    completion_embs: ndarray[InPrefLen, CompletionLen, EmbedDim, Float]


class PrefRepOutput(
    NamedTuple, Generic[NumPrompts, InPrefLen, PromptLen, CompletionLen, PrefDim, EmbedDim, Float]
):
    prompt_embs: ndarray[NumPrompts, PromptLen, EmbedDim, Float]
    completion_embs: ndarray[NumPrompts, InPrefLen, CompletionLen, EmbedDim, Float]
    preference_output: PreferenceOutput[PrefDim, Float]


@dataclass(frozen=True)
class IRConfig(Generic[VocabSize, MaxSeqLen, MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float]):
    lm_layer_config: TransformerLayerConfig[EmbedDim, EmbedDim, Float]
    pref_encoder_config: TransformerLayerConfig[PrefDim, EmbedDim, Float]
    pref_decoder_config: PrefDecoderConfig[PrefDim, Float]
    num_lm_layers: int
    num_pref_encoder_layers: int
    vocab_size: VocabSize
    max_seq_len: MaxSeqLen
    max_prompt_len: MaxPromptLen
    max_pref_len: MaxPrefLen
    pad_token_id: Fin[VocabSize]


class IndividualRewardModel(
    eqx.Module, Generic[VocabSize, MaxSeqLen, MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float]
):
    preference_model: PreferenceModel[MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float]
    encoder: EmbeddingEncoder[VocabSize, MaxSeqLen, EmbedDim, Float]
    decoder: EmbeddingDecoder[VocabSize, MaxSeqLen, EmbedDim, Float]
    reward: RewardLayer[Decoded[PrefDim], EmbedDim, Float]
    pad_token_id: Fin[VocabSize] = eqx.field(static=True)
    dtype: type[Float] = eqx.field(static=True)

    def __init__(
        self,
        config: IRConfig[VocabSize, MaxSeqLen, MaxPromptLen, MaxPrefLen, PrefDim, EmbedDim, Float],
        *,
        key: jax.Array,
    ):
        pref_key, enc_key, dec_key, reward_key = jax.random.split(key, num=4)
        self.preference_model = PreferenceModel(
            encoder_config=config.pref_encoder_config,
            num_encoder_layers=config.num_pref_encoder_layers,
            max_pref_len=config.max_pref_len,
            max_prompt_len=config.max_prompt_len,
            decoder_config=config.pref_decoder_config,
            key=pref_key,
        )
        self.encoder = EmbeddingEncoder(
            config.lm_layer_config,
            num_layers=config.num_lm_layers,
            pad_token_id=config.pad_token_id,
            max_seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            key=enc_key,
        )
        self.decoder = EmbeddingDecoder(
            config.lm_layer_config,
            num_layers=config.num_lm_layers,
            pad_token_id=config.pad_token_id,
            max_seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            mask_type=NoMask(),
            key=dec_key,
        )
        self.reward = RewardLayer(
            embed_dim=config.lm_layer_config.q_dim,
            pref_dim=Decoded(config.pref_decoder_config.pref_dim),
            hidden_dim=config.lm_layer_config.hidden_dim,
            key=reward_key,
        )
        self.pad_token_id = config.pad_token_id
        self.dtype = type(config.lm_layer_config.dropout_rate)

    @eqx.filter_jit
    def __call__(
        self,
        prompts: ndarray[NumPrompts, PromptLen, Fin[VocabSize]],
        ordered_completions: ndarray[NumPrompts, Ordered[InPrefLen], CompletionLen, Fin[VocabSize]],
        out_completions: ndarray[NumPrompts, OutPrefLen, CompletionLen, Fin[VocabSize]],
        *,
        key: jax.Array,
    ) -> IndividualRewardOutput[NumPrompts, OutPrefLen, PromptLen, CompletionLen, PrefDim, EmbedDim, Float]:
        prompt_embs: ndarray[*tuple[NumPrompts, PromptLen, EmbedDim], Float] = jax.vmap(self._interp_prompt)(
            prompts
        )
        completion_embs = jax.vmap(self._interp_completions)(prompt_embs, prompts, ordered_completions)
        preference_output = self.preference_model.__call__(
            prompt_embs,
            prompts != self.pad_token_id,
            completion_embs,
            ordered_completions != self.pad_token_id,
            key=key,
        )

        out_completion_embs = jax.vmap(self._interp_completions)(prompt_embs, prompts, out_completions)

        def inner_reward(
            completion_tokens: ndarray[CompletionLen, Fin[VocabSize]],
            completion_embeds: ndarray[CompletionLen, EmbedDim, Float],
        ) -> ndarray[Float]:
            return self.reward.__call__(
                completion_embeds, completion_tokens != self.pad_token_id, preference_output.dec_out
            )

        def outer_reward(
            x: ndarray[OutPrefLen, CompletionLen, Fin[VocabSize]],
            y: ndarray[OutPrefLen, CompletionLen, EmbedDim, Float],
        ) -> ndarray[OutPrefLen, Float]:
            return jax.vmap(inner_reward)(x, y)

        rewards: ndarray[*tuple[NumPrompts, OutPrefLen], Float] = jax.vmap(outer_reward)(
            out_completions, out_completion_embs
        )
        return IndividualRewardOutput(rewards, preference_output, prompt_embs, out_completion_embs)

    def rewards(
        self,
        preference_dec_out: ndarray[Decoded[PrefDim], Float],
        completion_tokens: ndarray[InPrefLen, CompletionLen, Fin[VocabSize]],
        completion_embeds: ndarray[InPrefLen, CompletionLen, EmbedDim, Float],
    ) -> ndarray[InPrefLen, Float]:
        def inner(
            x: ndarray[CompletionLen, Fin[VocabSize]], y: ndarray[CompletionLen, EmbedDim, Float]
        ) -> ndarray[Float]:
            return self.reward.__call__(y, x != self.pad_token_id, preference_dec_out)

        return jax.vmap(inner)(completion_tokens, completion_embeds)

    @eqx.filter_jit
    def mk_pref_rep(
        self,
        prompts: ndarray[NumPrompts, PromptLen, Fin[VocabSize]],
        ordered_completions: ndarray[NumPrompts, Ordered[InPrefLen], CompletionLen, Fin[VocabSize]],
        *,
        key: jax.Array,
    ) -> PrefRepOutput[NumPrompts, Ordered[InPrefLen], PromptLen, CompletionLen, PrefDim, EmbedDim, Float]:
        prompt_embs = jax.vmap(self._interp_prompt)(prompts)
        completion_embs = jax.vmap(self._interp_completions)(prompt_embs, prompts, ordered_completions)
        return PrefRepOutput(
            prompt_embs,
            completion_embs,
            self.preference_model.mk_pref_rep(
                prompt_embs,
                prompts != self.pad_token_id,
                completion_embs,
                ordered_completions != self.pad_token_id,
                key=key,
            ),
        )

    def _interp_prompt(
        self,
        prompt: ndarray[PromptLen, Fin[VocabSize]],
    ) -> ndarray[PromptLen, EmbedDim, Float]:
        return self.encoder.__call__(prompt)

    @eqx.filter_jit
    def _interp_completions(
        self,
        prompt_embs: ndarray[PromptLen, EmbedDim, Float],
        prompt_tokens: ndarray[PromptLen, Fin[VocabSize]],
        completions: ndarray[InPrefLen, CompletionLen, Fin[VocabSize]],
    ):
        def decode(x: ndarray[CompletionLen, Fin[VocabSize]]) -> ndarray[CompletionLen, EmbedDim, Float]:
            return self.decoder.__call__(x, prompt_embs, prompt_tokens != self.pad_token_id)

        return jax.vmap(decode)(completions)

    @eqx.filter_jit
    def from_sample(
        self,
        prompt: ndarray[PromptLen, Fin[VocabSize]],
        completions: ndarray[InPrefLen, CompletionLen, Fin[VocabSize]],
        sample: ndarray[PrefDim, Float],
    ) -> IndividualRewardSampleOutput[InPrefLen, PromptLen, CompletionLen, VocabSize, PrefDim, EmbedDim, Float]:
        prompt_embs = self._interp_prompt(prompt)
        completion_embs = self._interp_completions(prompt_embs, prompt, completions)
        preference_rep = self.preference_model.decode_sample(sample)
        rewards = self.rewards(preference_rep, completions, completion_embs)
        return IndividualRewardSampleOutput(rewards, preference_rep, prompt_embs, completion_embs)

    @eqx.filter_jit
    def sample(
        self,
        prompt: ndarray[PromptLen, Fin[VocabSize]],
        completions: ndarray[InPrefLen, CompletionLen, Fin[VocabSize]],
        *,
        sampler_key: jax.Array,
    ) -> IndividualRewardSampleOutput[InPrefLen, PromptLen, CompletionLen, VocabSize, PrefDim, EmbedDim, Float]:
        prompt_embs = self._interp_prompt(prompt)
        completion_embs = self._interp_completions(prompt_embs, prompt, completions)
        preference_rep: ndarray[Decoded[PrefDim], Float] = self.preference_model.sample_and_decode(
            sampler_key=sampler_key
        )
        rewards = self.rewards(preference_rep, completions, completion_embs)
        return IndividualRewardSampleOutput(rewards, preference_rep, prompt_embs, completion_embs)
