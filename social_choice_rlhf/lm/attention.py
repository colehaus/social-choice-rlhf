from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar, TypeVarTuple, cast, overload

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import Module
from equinox.nn import Dropout, Linear
from jax import Array
from jax.numpy import ndarray

from social_choice_rlhf.util.misc import InstanceSingleton, product_

if TYPE_CHECKING:
    from numpy import Product

QSeqLen = TypeVar("QSeqLen", bound=int)
KVSeqLen = TypeVar("KVSeqLen", bound=int)
QKSize = TypeVar("QKSize", bound=int)
QSize = TypeVar("QSize", bound=int)
KSize = TypeVar("KSize", bound=int)
VSize = TypeVar("VSize", bound=int)
Float = TypeVar("Float", bound=float)
NumHeads = TypeVar("NumHeads", bound=int)
OutputSize = TypeVar("OutputSize", bound=int)
InDim = TypeVar("InDim", bound=int)
OutDim = TypeVar("OutDim", bound=int)
SeqLen = TypeVar("SeqLen", bound=int)


def dot_product_attention_weights(
    query: ndarray[QSeqLen, QKSize, Float],
    key: ndarray[KVSeqLen, QKSize, Float],
    mask: ndarray[QSeqLen, KVSeqLen, bool] | None = None,
    *,
    use_softmax1: bool,
) -> ndarray[QSeqLen, KVSeqLen, Float]:
    query = query / jnp.sqrt(query.shape[-1]).astype(query.dtype)
    logits: ndarray[QSeqLen, KVSeqLen, Float] = query @ key.T
    if mask is not None:
        logits = jnp.where(mask, logits, jnp.array(jnp.finfo(logits.dtype).min))
    if use_softmax1:
        return softmax1(logits)
    else:
        return jax.nn.softmax(logits)


def dot_product_attention(  # noqa: PLR0913
    query: ndarray[QSeqLen, QKSize, Float],
    key_: ndarray[KVSeqLen, QKSize, Float],
    value: ndarray[KVSeqLen, VSize, Float],
    mask: ndarray[QSeqLen, KVSeqLen, bool] | None = None,
    dropout: Dropout[Float] | None = None,
    *,
    use_softmax1: bool,
    key: Array | None = None,
    inference: bool | None = None,
) -> ndarray[QSeqLen, VSize, Float]:
    weights = dot_product_attention_weights(query, key_, mask, use_softmax1=use_softmax1)
    if dropout is not None:
        weights = dropout(weights, key=key, inference=inference)
    return weights @ value


class ScaledMHAttention(Module, Generic[QSize, KSize, VSize, OutputSize, Float]):
    # Purely internal types that shouldn't be visible to callers
    _NumHeads: TypeAlias = InstanceSingleton[Literal["NumHeads"]]
    _QKSize: TypeAlias = InstanceSingleton[Literal["QKSize"]]
    _VOSize: TypeAlias = InstanceSingleton[Literal["VOSize"]]

    query_proj: Linear[QSize, Product[_NumHeads, _QKSize], Float]
    key_proj: Linear[KSize, Product[_NumHeads, _QKSize], Float]
    value_proj: Linear[VSize, Product[_NumHeads, _VOSize], Float]
    output_proj: Linear[Product[_NumHeads, _VOSize], OutputSize, Float]
    num_heads: _NumHeads = eqx.field(static=True)
    dropout: Dropout[Float]
    scalars: ndarray[_NumHeads, Float]
    use_softmax1: bool = eqx.field(static=True)

    @overload
    def __init__(  # noqa: PLR0913
        self: ScaledMHAttention[QSize, QSize, QSize, QSize, Float],
        *,
        num_heads: int,
        use_softmax1: bool = False,
        query_size: QSize,
        key_size: None = None,
        value_size: None = None,
        output_size: None = None,
        qk_size: int | None = None,
        vo_size: int | None = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: Float = 0.0,
        inference: bool = False,
        key: Array,
        **kwargs: Any,
    ) -> None:
        ...

    @overload
    def __init__(  # noqa: PLR0913
        self,
        *,
        num_heads: int,
        use_softmax1: bool = False,
        query_size: QSize,
        key_size: KSize,
        value_size: VSize,
        output_size: OutputSize,
        qk_size: int | None = None,
        vo_size: int | None = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: Float = 0.0,
        inference: bool = False,
        key: Array,
        **kwargs: Any,
    ) -> None:
        ...

    def __init__(  # noqa: PLR0913
        self,
        *,
        num_heads: int,
        use_softmax1: bool = False,
        query_size: QSize,
        key_size: KSize | None = None,
        value_size: VSize | None = None,
        output_size: OutputSize | None = None,
        qk_size: int | None = None,
        vo_size: int | None = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: Float = 0.0,
        inference: bool = False,
        key: Array,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        qkey, kkey, vkey, okey = jax.random.split(key, 4)
        if key_size is None:
            key_size = cast(KSize, query_size)
        if value_size is None:
            value_size = cast(VSize, query_size)
        qk_size = InstanceSingleton[Literal["QKSize"]](
            self, "QKSize", query_size // num_heads if qk_size is None else qk_size
        )
        vo_size = InstanceSingleton[Literal["VOSize"]](
            self, "VOSize", query_size // num_heads if vo_size is None else vo_size
        )
        if output_size is None:
            output_size = cast(OutputSize, query_size)
        self.num_heads = InstanceSingleton(self, "NumHeads", num_heads)
        self.query_proj = Linear(
            query_size, product_((self.num_heads, qk_size)), use_bias=use_query_bias, key=qkey
        )
        self.key_proj = Linear(key_size, product_((self.num_heads, qk_size)), use_bias=use_key_bias, key=kkey)
        self.value_proj = Linear(
            value_size, product_((self.num_heads, vo_size)), use_bias=use_value_bias, key=vkey
        )
        self.output_proj = Linear(
            product_((self.num_heads, vo_size)), output_size, use_bias=use_output_bias, key=okey
        )
        self.dropout = Dropout(dropout_p, inference=inference)
        # When/if the ultimate user does a `tree_map`, the `Float` will become accurate
        self.scalars = cast(
            "ndarray[ScaledMHAttention._NumHeads, Float]",
            jnp.ones(self.num_heads, dtype=jnp.float32),
        )
        self.use_softmax1 = use_softmax1

    def __call__(  # noqa: PLR0913
        self,
        query: ndarray[QSeqLen, QSize, Float],
        key_: ndarray[KVSeqLen, KSize, Float],
        value: ndarray[KVSeqLen, VSize, Float],
        mask: ndarray[QSeqLen, KVSeqLen, bool] | None = None,
        *,
        key: Array | None = None,
        inference: bool | None = None,
    ) -> ndarray[QSeqLen, OutputSize, Float]:
        query_heads: ndarray[
            QSeqLen, ScaledMHAttention._NumHeads, ScaledMHAttention._QKSize, Float
        ] = self._project(self.query_proj, query)
        key_heads: ndarray[
            KVSeqLen, ScaledMHAttention._NumHeads, ScaledMHAttention._QKSize, Float
        ] = self._project(self.key_proj, key_)
        value_heads: ndarray[
            KVSeqLen, ScaledMHAttention._NumHeads, ScaledMHAttention._VOSize, Float
        ] = self._project(self.value_proj, value)

        def attn_fn(
            q: ndarray[QSeqLen, ScaledMHAttention._QKSize, Float],
            k: ndarray[KVSeqLen, ScaledMHAttention._QKSize, Float],
            v: ndarray[KVSeqLen, VSize, Float],
            key: Array | None,
        ) -> ndarray[QSeqLen, VSize, Float]:
            return dot_product_attention(
                q,
                k,
                v,
                dropout=self.dropout,
                mask=mask,
                key=key,
                inference=inference,
                use_softmax1=self.use_softmax1,
            )

        keys = jax.random.split(key, self.num_heads) if key is not None else None
        attn: ndarray[QSeqLen, ScaledMHAttention._NumHeads, ScaledMHAttention._VOSize, Float] = jax.vmap(
            attn_fn, in_axes=(1, 1, 1, 0), out_axes=1
        )(query_heads, key_heads, value_heads, keys)
        attn = jnp.expand_dims(self.scalars, (0, 2)) * attn

        concatenated_attention: ndarray[
            QSeqLen, Product[ScaledMHAttention._NumHeads, ScaledMHAttention._VOSize], Float
        ] = jnp.reshape(attn, (query.shape[0], product_(attn.shape[1:])))
        return jax.vmap(self.output_proj)(concatenated_attention)

    def _project(
        self,
        proj: Linear[InDim, Product[ScaledMHAttention._NumHeads, OutDim], Float],
        x: ndarray[SeqLen, InDim, Float],
    ) -> ndarray[SeqLen, ScaledMHAttention._NumHeads, OutDim, Float]:
        projection: ndarray[SeqLen, Product[ScaledMHAttention._NumHeads, OutDim], Float] = jax.vmap(proj)(x)
        return jnp.reshape(projection, (x.shape[0], self.num_heads, cast(OutDim, -1)))


Shape = TypeVarTuple("Shape")
Dim1 = TypeVar("Dim1", bound=int)


# JAX implementation copied and simplified for reference
# Max for numerical stability
def softmax(x: ndarray[*Shape, Dim1, Float]):
    x_max = jnp.max(x, axis=-1, keepdims=True)
    unnormalized = jnp.exp(x - x_max)
    return unnormalized / jnp.sum(unnormalized, axis=-1, keepdims=True)


def softmax1(x: ndarray[*Shape, Dim1, Float]):
    """Add a one to the denominator to allow an attention head to upweight or downweight itself.
    See https://www.evanmiller.org/attention-is-off-by-one.html
    https://arxiv.org/pdf/1803.07294.pdf also describes the problem of attention heads all being given full weight,
    even when their subspace is not relevant ATM.

    We think of the 1 in the denominator as $e^0$ so it becomes $e^{-max}$
    after the max shift for numerical stability.

    https://github.com/google/flaxformer/blob/ea17eb012a1d340ddff017b7a534c2162aaec34c/flaxformer/components/attention/dense_attention.py#L50-L50
    """
    # `stop_gradient` for numerical stability
    m = jnp.maximum(jax.lax.stop_gradient(x.max(-1, keepdims=True)), 0)
    unnormalized = jnp.exp(x - m)
    return unnormalized / (jnp.exp(-m) + jnp.sum(unnormalized, axis=-1, keepdims=True))
