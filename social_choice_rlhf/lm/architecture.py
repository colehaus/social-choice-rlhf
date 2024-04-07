from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NamedTuple,
    Protocol,
    TypeAlias,
    TypeVar,
    TypeVarTuple,
    cast,
)

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.numpy import ndarray

from social_choice_rlhf.lm.attention import ScaledMHAttention
from social_choice_rlhf.util.jax import scan_layers_dropout_key, split_optional
from social_choice_rlhf.util.misc import InstanceSingleton

if TYPE_CHECKING:
    from numpy import Fin

SeqLen = TypeVar("SeqLen", bound=int)
InSeqLen = TypeVar("InSeqLen", bound=int)
OutSeqLen = TypeVar("OutSeqLen", bound=int)
EmbedDim = TypeVar("EmbedDim", bound=int)
QDim = TypeVar("QDim", bound=int)
KDim = TypeVar("KDim", bound=int)
VDim = TypeVar("VDim", bound=int)
OutputDim = TypeVar("OutputDim", bound=int)
KVDim = TypeVar("KVDim", bound=int)
VocabSize = TypeVar("VocabSize", bound=int)
NumLayers = TypeVar("NumLayers", bound=int)
BatchLen = TypeVar("BatchLen", bound=int)
QSeqLen = TypeVar("QSeqLen", bound=int)
KVSeqLen = TypeVar("KVSeqLen", bound=int)
NumClasses = TypeVar("NumClasses", bound=int)
Shape = TypeVarTuple("Shape")
Float = TypeVar("Float", bound=float)
MaxSeqLen = TypeVar("MaxSeqLen", bound=int)


# fmt: off
@dataclass(frozen=True)
class NoMask: pass  # noqa: E701
@dataclass(frozen=True)
class CausalMask: pass  # noqa: E701
# fmt: on

MaskType: TypeAlias = NoMask | CausalMask


def mk_mask(padding_mask: ndarray[SeqLen, bool], *, mask_type: MaskType) -> ndarray[SeqLen, SeqLen, bool]:
    full_padding_mask: ndarray[SeqLen, SeqLen, bool] = jnp.expand_dims(padding_mask, axis=-1) * jnp.expand_dims(
        padding_mask, axis=-2
    )
    match mask_type:
        case NoMask():
            return full_padding_mask
        case CausalMask():
            causal_mask = jnp.tril(jnp.ones((padding_mask.shape[0], padding_mask.shape[0]), bool), k=0)
            return full_padding_mask * causal_mask


class RMSNorm(eqx.Module, Generic[*Shape, Float]):
    weight: ndarray[*Shape, Float] | ndarray[Float]
    bias: ndarray[*Shape, Float] | ndarray[Float]
    # It seems like we'd be able to omit the `static=True` here and rely on equinox filtering
    # since it's not an array. However, when we do our `to_bfloat16` conversion pass,
    # it becomes an `np.generic` which satisfies `eqx.is_array`.
    # https://github.com/patrick-kidger/equinox/issues/507
    eps: Float = eqx.field(static=True)
    shape: tuple[*Shape] = eqx.field(static=True)

    def __init__(self, shape: tuple[*Shape], *, shared: bool, eps: Float = cast(Float, 1e-5)):
        """`shared=True` uses one scalar for weight and one for bias after normalization.
        This makes `RMSNorm` equivalent to `ScaleNorm` except for a `sqrt(N)` factor.
        https://arxiv.org/pdf/1910.05895.pdf
        """
        self.eps = eps
        self.shape = shape
        if shared:
            self.weight = jnp.array(1, dtype=type(eps))
            self.bias = jnp.array(0, dtype=type(eps))
        else:
            self.weight = jnp.ones(shape, dtype=type(eps))
            self.bias = jnp.zeros(shape, dtype=type(eps))

    def __call__(self, x: ndarray[*Shape, Float]) -> ndarray[*Shape, Float]:
        if x.shape != self.shape:
            raise ValueError(f"Argument shape {x.shape} must match constructor shape {self.shape}.")

        inv_rms = jax.lax.rsqrt(jnp.maximum(jnp.mean(x * x), 0) + jnp.array(self.eps, dtype=x.dtype))
        return self.weight * (x * inv_rms) + self.bias


# `dataclass` because getting `NamedTuple`s to serialize in a custom way is a pain
@dataclass(frozen=True)
class NormConfig:
    pos: Literal["post", "norm_former"]
    type_: Literal["layer_norm", "rms_norm", "scale_norm"]


@dataclass(frozen=True)
class TransformerLayerConfig(Generic[QDim, KVDim, Float]):
    q_dim: QDim
    kv_dim: KVDim
    hidden_dim: int
    num_heads: int
    norm_config: NormConfig
    attention_type: TradAttention | ScaledAttention
    dropout_rate: Float
    attention_dropout_rate: Float


@dataclass(frozen=True)
class ArchConfig(Generic[EmbedDim, VocabSize, MaxSeqLen, Float]):
    layer_config: TransformerLayerConfig[EmbedDim, EmbedDim, Float]
    num_layers: int
    vocab_size: VocabSize
    max_seq_len: MaxSeqLen
    pad_token_id: Fin[VocabSize]
    tie_embeddings: bool
    stochastic_encoder: bool


class Embedder(eqx.Module, Generic[VocabSize, MaxSeqLen, EmbedDim, Float]):
    token_embedder: eqx.nn.Embedding[VocabSize, EmbedDim, Float]
    position_embedder: eqx.nn.Embedding[MaxSeqLen, EmbedDim, Float]
    norm: eqx.nn.LayerNorm[EmbedDim, Float] | RMSNorm[EmbedDim, Float]
    dropout: eqx.nn.Dropout[Float]

    def __init__(  # noqa: PLR0913
        self,
        *,
        vocab_size: VocabSize,
        max_seq_len: MaxSeqLen,
        embed_size: EmbedDim,
        norm_config: NormConfig,
        dropout_rate: Float,
        key: jax.Array,
    ):
        token_key, position_key = jax.random.split(key, 2)

        self.token_embedder = eqx.nn.Embedding(num_embeddings=vocab_size, embedding_size=embed_size, key=token_key)
        self.position_embedder = eqx.nn.Embedding(
            num_embeddings=max_seq_len, embedding_size=embed_size, key=position_key
        )
        self.dropout = eqx.nn.Dropout(dropout_rate)
        match norm_config.type_:
            case "layer_norm":
                self.norm = eqx.nn.LayerNorm(shape=embed_size)
            case "rms_norm":
                self.norm = RMSNorm(shape=(embed_size,), shared=False)
            case "scale_norm":
                self.norm = RMSNorm(shape=(embed_size,), shared=True)

    def __call__(
        self,
        token_ids: ndarray[SeqLen, Fin[VocabSize]],
        *,
        dropout_key: jax.Array | None,
    ) -> ndarray[SeqLen, EmbedDim, Float]:
        tokens: ndarray[SeqLen, EmbedDim, Float] = jax.vmap(self.token_embedder)(token_ids)
        assert token_ids.shape[0] <= self.position_embedder.num_embeddings, (
            token_ids.shape,
            self.position_embedder.num_embeddings,
        )
        positions: ndarray[SeqLen, EmbedDim, Float] = jax.vmap(self.position_embedder)(
            jnp.arange(token_ids.shape[-1])
        )
        embedded_inputs = jax.vmap(self.norm)(tokens + positions)
        return self.dropout(embedded_inputs, inference=dropout_key is None, key=dropout_key)


class FeedForward(eqx.Module, Generic[EmbedDim, Float]):
    IntermediateSize: TypeAlias = InstanceSingleton[Literal["IntermediateSize"]]

    hidden: eqx.nn.Linear[EmbedDim, IntermediateSize, Float]
    output: eqx.nn.Linear[IntermediateSize, EmbedDim, Float]
    dropout: eqx.nn.Dropout[Float]
    norms: tuple[NormProt[EmbedDim, Float], NormProt[IntermediateSize, Float]] | tuple[NormProt[EmbedDim, Float]]

    def __init__(  # noqa: PLR0913
        self,
        *,
        embed_dim: EmbedDim,
        hidden_dim: int,
        norm_config: NormConfig,
        dropout_rate: Float = cast(Any, 0.0),
        key: jax.Array,
    ):
        mlp_key, output_key = jax.random.split(key)
        intermediate_size_ = InstanceSingleton[Literal["IntermediateSize"]](self, "IntermediateSize", hidden_dim)
        self.hidden = eqx.nn.Linear(in_features=embed_dim, out_features=intermediate_size_, key=mlp_key)
        self.output = eqx.nn.Linear(in_features=intermediate_size_, out_features=embed_dim, key=output_key)

        self.dropout = eqx.nn.Dropout(dropout_rate)
        match norm_config:
            case NormConfig("post", "layer_norm"):
                self.norms = (eqx.nn.LayerNorm(shape=embed_dim),)
            case NormConfig("post", "rms_norm"):
                self.norms = (RMSNorm(shape=(embed_dim,), shared=False),)
            case NormConfig("post", "scale_norm"):
                self.norms = (RMSNorm(shape=(embed_dim,), shared=True),)
            case NormConfig("norm_former", "layer_norm"):
                self.norms = (eqx.nn.LayerNorm(shape=embed_dim), eqx.nn.LayerNorm(shape=intermediate_size_))
            case NormConfig("norm_former", "rms_norm"):
                self.norms = (
                    RMSNorm(shape=(embed_dim,), shared=False),
                    RMSNorm(shape=(intermediate_size_,), shared=False),
                )
            case NormConfig("norm_former", "scale_norm"):
                self.norms = (
                    RMSNorm(shape=(embed_dim,), shared=True),
                    RMSNorm(shape=(intermediate_size_,), shared=True),
                )
            case NormConfig(_, _):
                raise ValueError(f"Unexpected norm config: {norm_config}")

    def _norm_former_call(
        self,
        input_: ndarray[EmbedDim, Float],
        dropout_key: jax.Array | None,
    ) -> ndarray[EmbedDim, Float]:
        assert len(self.norms) == 2  # noqa: PLR2004
        output = self.output(self.norms[1](jax.nn.gelu(self.hidden(self.norms[0](input_)))))
        return self.dropout(output, inference=dropout_key is None, key=dropout_key) + input_

    def _post_ln_call(
        self,
        input_: ndarray[EmbedDim, Float],
        dropout_key: jax.Array | None,
    ) -> ndarray[EmbedDim, Float]:
        assert len(self.norms) == 1
        output = self.output(jax.nn.gelu(self.hidden(input_)))
        return self.norms[0](self.dropout(output, inference=dropout_key is None, key=dropout_key) + input_)

    def __call__(
        self,
        input_: ndarray[EmbedDim, Float],
        dropout_key: jax.Array | None,
    ) -> ndarray[EmbedDim, Float]:
        match len(self.norms):
            case 1:
                return self._post_ln_call(input_, dropout_key)
            case 2:
                return self._norm_former_call(input_, dropout_key)
            case _:
                raise ValueError(f"Unexpected number of norms: {len(self.norms)}")


class CrossAttention(eqx.Module, Generic[QDim, KVDim, Float]):
    _NumHeads: TypeAlias = InstanceSingleton[Literal["NumHeads"]]
    attention: AttentionProt[QDim, KVDim, KVDim, QDim, Float]
    norms: tuple[NormProt[QDim, Float], NormProt[QDim, Float]] | tuple[NormProt[QDim, Float]]
    dropout: eqx.nn.Dropout[Float]

    def __init__(  # noqa: PLR0913
        self,
        *,
        q_dim: QDim,
        kv_dim: KVDim,
        num_heads: int,
        norm_config: NormConfig,
        attention_type: TradAttention | ScaledAttention,
        dropout_rate: Float,
        attention_dropout_rate: Float,
        key: jax.Array,
    ):
        num_heads = InstanceSingleton[Literal["NumHeads"]](self, "NumHeads", num_heads)
        match attention_type:
            case TradAttention():
                self.attention = eqx.nn.MultiheadAttention(
                    num_heads=num_heads,
                    query_size=q_dim,
                    output_size=q_dim,
                    key_size=kv_dim,
                    value_size=kv_dim,
                    dropout_p=attention_dropout_rate,
                    key=key,
                )
            case ScaledAttention(use_softmax1):
                self.attention = ScaledMHAttention(
                    num_heads=num_heads,
                    query_size=q_dim,
                    output_size=q_dim,
                    key_size=kv_dim,
                    value_size=kv_dim,
                    dropout_p=attention_dropout_rate,
                    use_softmax1=use_softmax1,
                    key=key,
                )
        self.dropout = eqx.nn.Dropout(dropout_rate)

        match norm_config:
            case NormConfig("post", "layer_norm"):
                self.norms = (eqx.nn.LayerNorm(shape=q_dim),)
            case NormConfig("post", "rms_norm"):
                self.norms = (RMSNorm(shape=(q_dim,), shared=False),)
            case NormConfig("post", "scale_norm"):
                self.norms = (RMSNorm(shape=(q_dim,), shared=True),)
            case NormConfig("norm_former", "layer_norm"):
                self.norms = (eqx.nn.LayerNorm(shape=q_dim), eqx.nn.LayerNorm(shape=q_dim))
            case NormConfig("norm_former", "rms_norm"):
                self.norms = (
                    RMSNorm(shape=(q_dim,), shared=False),
                    RMSNorm(shape=(q_dim,), shared=False),
                )
            case NormConfig("norm_former", "scale_norm"):
                self.norms = (
                    RMSNorm(shape=(q_dim,), shared=True),
                    RMSNorm(shape=(q_dim,), shared=True),
                )
            case NormConfig(_, _):
                raise ValueError(f"Unexpected norm config: {norm_config}")

    def __call__(  # noqa: PLR0913
        self,
        query: ndarray[QSeqLen, QDim, Float],
        key_value: ndarray[KVSeqLen, KVDim, Float],
        query_padding_mask: ndarray[QSeqLen, bool],
        key_value_padding_mask: ndarray[KVSeqLen, bool],
        dropout_key: jax.Array | None,
    ) -> ndarray[QSeqLen, QDim, Float]:
        match len(self.norms):
            case 1:
                return self._post_ln_call(
                    query, key_value, query_padding_mask, key_value_padding_mask, dropout_key
                )
            case 2:
                return self._norm_former_call(
                    query, key_value, query_padding_mask, key_value_padding_mask, dropout_key
                )
            case _:
                raise ValueError(f"Unexpected number of norms: {len(self.norms)}")

    def _norm_former_call(  # noqa: PLR0913
        self,
        query: ndarray[QSeqLen, QDim, Float],
        key_value: ndarray[KVSeqLen, KVDim, Float],
        query_padding_mask: ndarray[QSeqLen, bool],
        key_value_padding_mask: ndarray[KVSeqLen, bool],
        dropout_key: jax.Array | None,
    ) -> ndarray[QSeqLen, QDim, Float]:
        assert len(self.norms) == 2  # noqa: PLR2004
        attn_key, post_attn_dropout_key = split_optional(dropout_key, 2)
        attn_out = jax.vmap(self.norms[1])(
            self.attention(
                query=jax.vmap(self.norms[0])(query),
                key_=key_value,
                value=key_value,
                mask=(
                    jnp.expand_dims(query_padding_mask, axis=-1) * jnp.expand_dims(key_value_padding_mask, axis=-2)
                ),
                inference=dropout_key is None,
                key=attn_key,
            )
        )
        return self.dropout(attn_out, inference=dropout_key is None, key=post_attn_dropout_key) + query

    def _post_ln_call(  # noqa: PLR0913
        self,
        query: ndarray[QSeqLen, QDim, Float],
        key_value: ndarray[KVSeqLen, KVDim, Float],
        query_padding_mask: ndarray[QSeqLen, bool],
        key_value_padding_mask: ndarray[KVSeqLen, bool],
        dropout_key: jax.Array | None,
    ) -> ndarray[QSeqLen, QDim, Float]:
        assert len(self.norms) == 1
        attn_key, post_attn_dropout_key = split_optional(dropout_key, 2)

        attn_out = self.attention(
            query=query,
            key_=key_value,
            value=key_value,
            mask=jnp.expand_dims(query_padding_mask, axis=-1) * jnp.expand_dims(key_value_padding_mask, axis=-2),
            inference=dropout_key is None,
            key=attn_key,
        )

        result = self.dropout(attn_out, inference=dropout_key is None, key=post_attn_dropout_key)
        return jax.vmap(self.norms[0])(result + query)


@dataclass(frozen=True)
class TradAttention:
    pass


@dataclass(frozen=True)
class ScaledAttention:
    use_softmax1: bool


class AttentionProt(Protocol[QDim, KDim, VDim, OutputDim, Float]):
    def __call__(  # noqa: PLR0913
        self,
        query: ndarray[QSeqLen, QDim, Float],
        key_: ndarray[KVSeqLen, KDim, Float],
        value: ndarray[KVSeqLen, VDim, Float],
        mask: ndarray[QSeqLen, KVSeqLen, bool] | None = None,
        *,
        key: jax.Array | None = None,
        inference: bool | None = None,
    ) -> ndarray[QSeqLen, OutputDim, Float]:
        ...


class NormProt(Protocol[EmbedDim, Float]):
    def __call__(self, x: ndarray[EmbedDim, Float]) -> ndarray[EmbedDim, Float]:
        ...


class SelfAttention(eqx.Module, Generic[EmbedDim, Float]):
    _NumHeads: TypeAlias = InstanceSingleton[Literal["NumHeads"]]

    attention: AttentionProt[EmbedDim, EmbedDim, EmbedDim, EmbedDim, Float]
    norms: tuple[NormProt[EmbedDim, Float], NormProt[EmbedDim, Float]] | tuple[NormProt[EmbedDim, Float]]
    dropout: eqx.nn.Dropout[Float]
    norm_config: NormConfig = eqx.field(static=True)
    mask_type: MaskType = eqx.field(static=True)

    def __init__(  # noqa: PLR0913
        self,
        *,
        embed_dim: EmbedDim,
        num_heads: int,
        norm_config: NormConfig,
        attention_type: TradAttention | ScaledAttention,
        dropout_rate: Float,
        attention_dropout_rate: Float,
        mask_type: MaskType,
        key: jax.Array,
    ):
        num_heads = InstanceSingleton[Literal["NumHeads"]](self, "NumHeads", num_heads)
        match attention_type:
            case TradAttention():
                self.attention = eqx.nn.MultiheadAttention(
                    num_heads=num_heads,
                    query_size=embed_dim,
                    dropout_p=attention_dropout_rate,
                    key=key,
                )
            case ScaledAttention(use_softmax1):
                self.attention = ScaledMHAttention(
                    num_heads=num_heads,
                    query_size=embed_dim,
                    dropout_p=attention_dropout_rate,
                    use_softmax1=use_softmax1,
                    key=key,
                )
        self.norm_config = norm_config
        match norm_config:
            case NormConfig("post", "layer_norm"):
                self.norms = (eqx.nn.LayerNorm(shape=embed_dim),)
            case NormConfig("post", "rms_norm"):
                self.norms = (RMSNorm(shape=(embed_dim,), shared=False),)
            case NormConfig("post", "scale_norm"):
                self.norms = (RMSNorm(shape=(embed_dim,), shared=True),)
            case NormConfig("norm_former", "layer_norm"):
                self.norms = (eqx.nn.LayerNorm(shape=embed_dim), eqx.nn.LayerNorm(shape=embed_dim))
            case NormConfig("norm_former", "rms_norm"):
                self.norms = (
                    RMSNorm(shape=(embed_dim,), shared=False),
                    RMSNorm(shape=(embed_dim,), shared=False),
                )
            case NormConfig("norm_former", "scale_norm"):
                self.norms = (RMSNorm(shape=(embed_dim,), shared=True), RMSNorm(shape=(embed_dim,), shared=True))
            case NormConfig(_, _):
                raise ValueError(f"Unexpected norm config: {norm_config}")
        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.mask_type = mask_type

    def __call__(
        self,
        input_: ndarray[SeqLen, EmbedDim, Float],
        padding_mask: ndarray[SeqLen, bool],
        dropout_key: jax.Array | None,
    ) -> ndarray[SeqLen, EmbedDim, Float]:
        match self.norm_config.pos:
            case "post":
                return self._post_ln_call(input_, padding_mask, dropout_key)
            case "norm_former":
                return self._norm_former_call(input_, padding_mask, dropout_key)

    def _norm_former_call(
        self,
        input_: ndarray[SeqLen, EmbedDim, Float],
        padding_mask: ndarray[SeqLen, bool],
        dropout_key: jax.Array | None,
    ) -> ndarray[SeqLen, EmbedDim, Float]:
        assert len(self.norms) == 2  # noqa: PLR2004
        attn_dropout_key, post_attn_dropout_key = split_optional(dropout_key, 2)
        mask = mk_mask(padding_mask, mask_type=self.mask_type)
        normed = jax.vmap(self.norms[0])(input_)
        attn_out = jax.vmap(self.norms[1])(
            self.attention(
                query=normed,
                key_=normed,
                value=normed,
                mask=mask,
                inference=dropout_key is None,
                key=attn_dropout_key,
            )
        )
        return self.dropout(attn_out, inference=dropout_key is None, key=post_attn_dropout_key) + input_

    def _post_ln_call(
        self,
        input_: ndarray[SeqLen, EmbedDim, Float],
        padding_mask: ndarray[SeqLen, bool],
        dropout_key: jax.Array | None,
    ) -> ndarray[SeqLen, EmbedDim, Float]:
        assert len(self.norms) == 1
        attn_dropout_key, post_attn_dropout_key = split_optional(dropout_key, 2)
        mask = mk_mask(padding_mask, mask_type=self.mask_type)

        attn_out = self.attention(
            query=input_,
            key_=input_,
            value=input_,
            mask=mask,
            inference=dropout_key is None,
            key=attn_dropout_key,
        )

        result = self.dropout(attn_out, inference=dropout_key is None, key=post_attn_dropout_key)
        return jax.vmap(self.norms[0])(result + input_)


class DecoderLayer(eqx.Module, Generic[QDim, KVDim, Float]):
    self_attention: SelfAttention[QDim, Float]
    cross_attention: CrossAttention[QDim, KVDim, Float]
    feed_forward: FeedForward[QDim, Float]
    layer_num: ndarray[int] = eqx.field(static=True)

    def __init__(
        self,
        config: TransformerLayerConfig[QDim, KVDim, Float],
        *,
        layer_num: ndarray[int],
        mask_type: MaskType,
        key: jax.Array,
    ):
        self_attention_key, cross_attention_key, ff_key = jax.random.split(key, num=3)

        self.layer_num = layer_num
        self.self_attention = SelfAttention(
            embed_dim=config.q_dim,
            num_heads=config.num_heads,
            norm_config=config.norm_config,
            attention_type=config.attention_type,
            dropout_rate=config.dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            key=self_attention_key,
            mask_type=mask_type,
        )
        self.cross_attention = CrossAttention(
            q_dim=config.q_dim,
            kv_dim=config.kv_dim,
            num_heads=config.num_heads,
            norm_config=config.norm_config,
            attention_type=config.attention_type,
            dropout_rate=config.dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            key=cross_attention_key,
        )
        self.feed_forward = FeedForward(
            embed_dim=config.q_dim,
            hidden_dim=config.hidden_dim,
            norm_config=config.norm_config,
            dropout_rate=config.dropout_rate,
            key=ff_key,
        )

    def __call__(  # noqa: PLR0913
        self,
        query: ndarray[QSeqLen, QDim, Float],
        key_value: ndarray[KVSeqLen, KVDim, Float],
        query_padding_mask: ndarray[QSeqLen, bool],
        key_value_padding_mask: ndarray[KVSeqLen, bool],
        dropout_key: jax.Array | None,
    ) -> ndarray[QSeqLen, QDim, Float]:
        self_attn_key, cross_attn_key, ff_key = split_optional(dropout_key, num=3)

        self_attn_out = self.self_attention.__call__(query, query_padding_mask, dropout_key=self_attn_key)
        cross_attn_out = self.cross_attention.__call__(
            query=self_attn_out,
            key_value=key_value,
            query_padding_mask=query_padding_mask,
            key_value_padding_mask=key_value_padding_mask,
            dropout_key=cross_attn_key,
        )
        return jax.vmap(self.feed_forward.__call__)(
            cross_attn_out, None if ff_key is None else jax.random.split(ff_key, num=query.shape[0])
        )


class EncoderLayer(eqx.Module, Generic[EmbedDim, Float]):
    self_attention: SelfAttention[EmbedDim, Float]
    feed_forward: FeedForward[EmbedDim, Float]
    layer_num: ndarray[int] = eqx.field(static=True)

    def __init__(
        self,
        config: TransformerLayerConfig[EmbedDim, EmbedDim, Float],
        *,
        layer_num: ndarray[int],
        key: jax.Array,
    ):
        attention_key, ff_key = jax.random.split(key)
        self.layer_num = layer_num

        self.self_attention = SelfAttention(
            embed_dim=config.q_dim,
            num_heads=config.num_heads,
            norm_config=config.norm_config,
            attention_type=config.attention_type,
            dropout_rate=config.dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            mask_type=NoMask(),
            key=attention_key,
        )
        self.feed_forward = FeedForward(
            embed_dim=config.q_dim,
            hidden_dim=config.hidden_dim,
            norm_config=config.norm_config,
            dropout_rate=config.dropout_rate,
            key=ff_key,
        )

    def __call__(
        self,
        input_: ndarray[SeqLen, EmbedDim, Float],
        mask: ndarray[SeqLen, bool],
        dropout_key: jax.Array | None,
    ) -> ndarray[SeqLen, EmbedDim, Float]:
        attn_key, ff_key = split_optional(dropout_key, num=2)
        attn_out = self.self_attention.__call__(input_, mask, dropout_key=attn_key)
        return jax.vmap(self.feed_forward)(
            attn_out, None if ff_key is None else jax.random.split(ff_key, num=input_.shape[0])
        )


class Sampler(eqx.Module, Generic[Float]):
    # Squared to ensure positivity
    sqrt_std_dev: ndarray[Float]

    def __init__(self, dtype: type[Float]):
        self.sqrt_std_dev = jnp.array(0.1, dtype=dtype)

    def __call__(self, mean: ndarray[EmbedDim, Float], key: jax.Array) -> ndarray[EmbedDim, Float]:
        # Reparameterization trick
        sample = jax.random.normal(
            key=key,
            shape=mean.shape,
            dtype=mean.dtype,
        )
        return mean + jnp.square(self.sqrt_std_dev) * sample


class Decoder(eqx.Module, Generic[QDim, KVDim, Float]):
    _NumLayers: TypeAlias = InstanceSingleton[Literal["NumLayers"]]

    layers: jax.AuxDim[_NumLayers, DecoderLayer[QDim, KVDim, Float]]
    norm: eqx.nn.LayerNorm[QDim, Float] | RMSNorm[QDim, Float] | None

    def __init__(
        self,
        layer_config: TransformerLayerConfig[QDim, KVDim, Float],
        *,
        num_layers: int,
        mask_type: MaskType,
        key: jax.Array,
    ):
        def mk_decoder_layer(
            layer_key: jax.Array, layer_num: ndarray[Fin[Decoder._NumLayers]]
        ) -> DecoderLayer[QDim, KVDim, Float]:
            return DecoderLayer(
                layer_config,
                layer_num=cast("ndarray[int]", layer_num),
                mask_type=mask_type,
                key=layer_key,
            )

        num_layers = InstanceSingleton[Literal["NumLayers"]](self, "NumLayers", num_layers)

        self.layers = eqx.filter_vmap(mk_decoder_layer)(
            jax.random.split(key, num=num_layers), jnp.arange(num_layers)
        )
        match layer_config.norm_config:
            case NormConfig("post", _):
                # Post norm has a norm on the residual stream in each layer so doesn't need a final one here
                self.norm = None
            case NormConfig("norm_former", "layer_norm"):
                self.norm = eqx.nn.LayerNorm(shape=layer_config.q_dim)
            case NormConfig("norm_former", "rms_norm"):
                self.norm = RMSNorm(shape=(layer_config.q_dim,), shared=False)
            case NormConfig("norm_former", "scale_norm"):
                self.norm = RMSNorm(shape=(layer_config.q_dim,), shared=True)
            case NormConfig(_, _):
                raise ValueError(f"Unexpected norm config: {layer_config.norm_config}")

    def __call__(  # noqa: PLR0913
        self,
        query: ndarray[OutSeqLen, QDim, Float],
        key_value: ndarray[InSeqLen, KVDim, Float],
        query_padding_mask: ndarray[OutSeqLen, bool],
        key_value_padding_mask: ndarray[InSeqLen, bool],
        dropout_key: jax.Array | None = None,
    ) -> ndarray[OutSeqLen, QDim, Float]:
        raw = scan_layers_dropout_key(
            query,
            self.layers,
            key_value,
            query_padding_mask,
            key_value_padding_mask,
            dropout_key=dropout_key,
        )
        match self.norm:
            case None:
                return raw
            case _:
                return jax.vmap(self.norm)(raw)


class EmbeddingDecoder(eqx.Module, Generic[VocabSize, MaxSeqLen, EmbedDim, Float]):
    embedder: Embedder[VocabSize, MaxSeqLen, EmbedDim, Float]
    decoder: Decoder[EmbedDim, EmbedDim, Float]
    pad_token_id: Fin[VocabSize] = eqx.field(static=True)

    def __init__(  # noqa: PLR0913
        self,
        layer_config: TransformerLayerConfig[EmbedDim, EmbedDim, Float],
        *,
        pad_token_id: Fin[VocabSize],
        vocab_size: VocabSize,
        max_seq_len: MaxSeqLen,
        num_layers: int,
        mask_type: MaskType,
        key: jax.Array,
    ):
        self.pad_token_id = pad_token_id
        emb_key, dec_key = jax.random.split(key)
        self.embedder = Embedder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_size=layer_config.q_dim,
            norm_config=layer_config.norm_config,
            dropout_rate=layer_config.dropout_rate,
            key=emb_key,
        )
        self.decoder = Decoder(
            layer_config,
            num_layers=num_layers,
            mask_type=mask_type,
            key=dec_key,
        )

    def __call__(
        self,
        query: ndarray[OutSeqLen, Fin[VocabSize]],
        key_value: ndarray[InSeqLen, EmbedDim, Float],
        key_value_padding_mask: ndarray[InSeqLen, bool],
        dropout_key: jax.Array | None = None,
    ) -> ndarray[OutSeqLen, EmbedDim, Float]:
        emb_key, dec_key = split_optional(dropout_key, num=2)
        return self.decoder.__call__(
            self.embedder.__call__(query, dropout_key=emb_key),
            key_value,
            (query != self.pad_token_id),
            key_value_padding_mask,
            dropout_key=dec_key,
        )


class StochasticOutput(NamedTuple, Generic[*Shape, SeqLen, EmbedDim, Float]):
    mean: ndarray[*Shape, SeqLen, EmbedDim, Float]
    std_dev: ndarray[*Shape, Float]
    sample: ndarray[*Shape, SeqLen, EmbedDim, Float]


class DeterministicOutput(NamedTuple, Generic[*Shape, SeqLen, EmbedDim, Float]):
    value: ndarray[*Shape, SeqLen, EmbedDim, Float]


class Encoder(eqx.Module, Generic[EmbedDim, Float]):
    _NumLayers: TypeAlias = InstanceSingleton[Literal["NumLayers"]]

    layers: jax.AuxDim[_NumLayers, EncoderLayer[EmbedDim, Float]]
    norm: eqx.nn.LayerNorm[EmbedDim, Float] | RMSNorm[EmbedDim, Float] | None

    def __init__(
        self,
        layer_config: TransformerLayerConfig[EmbedDim, EmbedDim, Float],
        *,
        num_layers: int,
        key: jax.Array,
    ):
        # https://docs.kidger.site/equinox/tricks/#improve-compilation-speed-with-scan-over-layers
        def mk_encoder_layer(
            layer_key: jax.Array, layer_num: ndarray[Fin[Encoder._NumLayers]]
        ) -> EncoderLayer[EmbedDim, Float]:
            return EncoderLayer(
                layer_config,
                layer_num=cast("ndarray[int]", layer_num),
                key=layer_key,
            )

        num_layers = InstanceSingleton[Literal["NumLayers"]](self, "NumLayers", num_layers)
        self.layers = eqx.filter_vmap(mk_encoder_layer)(
            jax.random.split(key, num=num_layers), jnp.arange(num_layers)
        )

        match layer_config.norm_config:
            case NormConfig("post", _):
                # Post norm has a norm on the residual stream in each layer so doesn't need a final one here
                self.norm = None
            case NormConfig("norm_former", "layer_norm"):
                self.norm = eqx.nn.LayerNorm(shape=layer_config.q_dim)
            case NormConfig("norm_former", "rms_norm"):
                self.norm = RMSNorm(shape=(layer_config.q_dim,), shared=False)
            case NormConfig("norm_former", "scale_norm"):
                self.norm = RMSNorm(shape=(layer_config.q_dim,), shared=True)
            case NormConfig(_, _):
                raise ValueError(f"Unexpected norm config: {layer_config.norm_config}")

    def __call__(
        self,
        embeds: ndarray[SeqLen, EmbedDim, Float],
        padding_mask: ndarray[SeqLen, bool],
        dropout_key: jax.Array | None,
    ) -> ndarray[SeqLen, EmbedDim, Float]:
        raw = scan_layers_dropout_key(embeds, self.layers, padding_mask, dropout_key=dropout_key)
        # https://arxiv.org/pdf/2304.14802.pdf
        # x: ndarray[Encoder.NumLayers, SeqLen, EmbedSize, Float] = from_aux_dim(layer_outs)
        # diff = jnp.mean(jnp.abs(x[1:, ...] - x[:-1, ...]), axis=(1, 2))
        # jax.debug.print("diff per layer {d}", d=diff)
        match self.norm:
            case None:
                return raw
            case _:
                return jax.vmap(self.norm)(raw)


class EmbeddingEncoder(eqx.Module, Generic[VocabSize, MaxSeqLen, EmbedDim, Float]):
    embedder: Embedder[VocabSize, MaxSeqLen, EmbedDim, Float]
    encoder: Encoder[EmbedDim, Float]
    pad_token_id: Fin[VocabSize] = eqx.field(static=True)

    def __init__(  # noqa: PLR0913
        self,
        layer_config: TransformerLayerConfig[EmbedDim, EmbedDim, Float],
        *,
        pad_token_id: Fin[VocabSize],
        vocab_size: VocabSize,
        max_seq_len: MaxSeqLen,
        num_layers: int,
        key: jax.Array,
    ):
        self.pad_token_id = pad_token_id
        emb_key, enc_key = jax.random.split(key)
        self.embedder = Embedder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_size=layer_config.q_dim,
            norm_config=layer_config.norm_config,
            dropout_rate=layer_config.dropout_rate,
            key=emb_key,
        )
        self.encoder = Encoder(
            layer_config,
            num_layers=num_layers,
            key=enc_key,
        )

    def __call__(
        self,
        token_ids: ndarray[InSeqLen, Fin[VocabSize]],
        dropout_key: jax.Array | None = None,
    ) -> ndarray[InSeqLen, EmbedDim, Float]:
        emb_key, enc_key = split_optional(dropout_key, num=2)
        return self.encoder.__call__(
            self.embedder.__call__(token_ids, dropout_key=emb_key), token_ids != self.pad_token_id, enc_key
        )


class LMOutput(NamedTuple, Generic[*Shape, InSeqLen, OutSeqLen, EmbedDim, VocabSize, Float]):
    encoder_output: DeterministicOutput[*Shape, InSeqLen, EmbedDim, Float] | StochasticOutput[
        *Shape, InSeqLen, EmbedDim, Float
    ]
    decoder_output: ndarray[*Shape, OutSeqLen, EmbedDim, Float]
    logit_output: ndarray[*Shape, OutSeqLen, VocabSize, Float]


class LM(eqx.Module, Generic[EmbedDim, VocabSize, MaxSeqLen, Float]):
    encoder: EmbeddingEncoder[VocabSize, MaxSeqLen, EmbedDim, Float]
    sampler: Sampler[Float] | None
    decoder: EmbeddingDecoder[VocabSize, MaxSeqLen, EmbedDim, Float]
    logit: eqx.nn.Linear[EmbedDim, VocabSize, Float]
    pad_token_id: Fin[VocabSize] = eqx.field(static=True)

    def __init__(
        self,
        config: ArchConfig[EmbedDim, VocabSize, MaxSeqLen, Float],
        *,
        key: jax.Array,
    ):
        self.pad_token_id = config.pad_token_id
        encoder_key, decoder_key, logit_key = jax.random.split(key, num=3)

        self.encoder = EmbeddingEncoder(
            layer_config=config.layer_config,
            pad_token_id=config.pad_token_id,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            num_layers=config.num_layers,
            key=encoder_key,
        )
        if config.stochastic_encoder:
            self.sampler = Sampler(dtype=type(config.layer_config.dropout_rate))
        else:
            self.sampler = None
        self.decoder = EmbeddingDecoder(
            layer_config=config.layer_config,
            pad_token_id=config.pad_token_id,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            num_layers=config.num_layers,
            mask_type=CausalMask(),
            key=decoder_key,
        )
        self.logit = eqx.nn.Linear(
            in_features=config.layer_config.q_dim,
            out_features=config.vocab_size,
            key=logit_key,
        )

    def __call__(
        self,
        encoder_ids: ndarray[InSeqLen, Fin[VocabSize]],
        decoder_ids: ndarray[OutSeqLen, Fin[VocabSize]],
        dropout_key: jax.Array | None,
        sampler_key: jax.Array | None,
    ) -> LMOutput[InSeqLen, OutSeqLen, EmbedDim, VocabSize, Float]:
        enc_key, dec_key = split_optional(dropout_key, num=2)

        raw_enc_out = self.encoder.__call__(encoder_ids, enc_key)

        match (self.sampler, sampler_key):
            case (None, None):
                enc_out = DeterministicOutput(raw_enc_out)
            case (None, _):
                raise ValueError("Cannot sample without a sampler")
            case (_, None):
                enc_out = DeterministicOutput(raw_enc_out)
            case (sampler, key):
                keys = jax.random.split(key, num=raw_enc_out.shape[0])
                enc_out = StochasticOutput(
                    mean=raw_enc_out,
                    std_dev=jnp.square(sampler.sqrt_std_dev),
                    sample=jax.vmap(sampler.__call__)(raw_enc_out, keys),
                )
        match enc_out:
            case DeterministicOutput():
                dec_input = enc_out.value
            case StochasticOutput():
                dec_input = enc_out.sample

        decoder_out = self.decoder.__call__(
            decoder_ids, dec_input, (encoder_ids != self.pad_token_id), dropout_key=dec_key
        )

        logits = jax.vmap(self.logit.__call__)(decoder_out)
        return LMOutput(enc_out, decoder_out, logits)
