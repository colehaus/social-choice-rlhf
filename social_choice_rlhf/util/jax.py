from __future__ import annotations

import functools
import os
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Concatenate,
    Literal,
    NamedTuple,
    ParamSpec,
    TypeVar,
    TypeVarTuple,
    cast,
    overload,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.numpy import ndarray
from jax.random import KeyArray

from social_choice_rlhf.util.misc import human_bytes

if TYPE_CHECKING:
    from optax import ArraysOf

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")


def strip_part(x: eqx.PartOf[A]) -> A:
    return cast(A, x)


def set_host_device_count(n: int) -> None:
    xla_flags = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(r"--xla_force_host_platform_device_count=\S+", "", xla_flags).split()
    os.environ["XLA_FLAGS"] = " ".join([f"--xla_force_host_platform_device_count={n}", *xla_flags])


P = ParamSpec("P")


def key_as_str(x: jtu.KeyEntry):
    match x:
        case jtu.GetAttrKey(name):
            return name
        case jtu.SequenceKey(idx):
            return str(idx)
        case jtu.DictKey(key):
            return str(key)


def key_path_as_str(x: jtu.KeyPath):
    return "/".join([key_as_str(key) for key in x])


def tree_size(tree: Any) -> int:
    return sum(x.nbytes for x in jtu.tree_leaves(tree) if eqx.is_array(x))


def print_tree(tree: Any, level: int | None):
    outs: dict[jtu.KeyPath, list[ndarray[*tuple[Any, ...], float]]] = defaultdict(list)
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if eqx.is_array(leaf):
            outs[path[:level]].append(leaf)
    for path, leaves in outs.items():
        type_str = " ".join({str(leaf.dtype) for leaf in leaves})
        total_num = sum([leaf.size for leaf in leaves])
        total_size = sum([leaf.nbytes for leaf in leaves])
        formatted_num = f"{total_num:,}"
        if len(leaves) == 1:
            print(
                f"{key_path_as_str(path):<100} "
                f"{formatted_num:>12} {type_str:<16} {human_bytes(total_size):<12} {leaves[0].shape}"
            )
        else:
            print(
                f"{key_path_as_str(path):<100} "
                f"{formatted_num:>12} {type_str:<16} {human_bytes(total_size):<12}"
            )


Num = TypeVar("Num", bound=int)


def split_optional(key: KeyArray | None, num: Num) -> ndarray[Num, Literal[2], int] | Sequence[None]:
    return [None] * num if key is None else jax.random.split(key, num)


NumLayers = TypeVar("NumLayers", bound=int)
Rest = TypeVarTuple("Rest")


def scan_layers(
    initial: A,
    initial_key: KeyArray | None,
    layers: jax.AuxDim[NumLayers, Callable[[A, *Rest, KeyArray | None], A]],
    # Constant across layers
    *args: *Rest,
) -> A:
    return scan_layers_with_intermediates(initial, initial_key, layers, *args)[0]


def _aux_part_of(x: eqx.PartOf[jax.AuxDim[NumLayers, B]]) -> jax.AuxDim[NumLayers, eqx.PartOf[B]]:
    return cast(Any, x)


def scan_layers_with_intermediates(
    initial: A,
    initial_key: KeyArray | None,
    layers: jax.AuxDim[NumLayers, Callable[[A, *Rest, KeyArray | None], A]],
    # Constant across layers
    *args: *Rest,
) -> tuple[A, jax.AuxDim[NumLayers, A]]:
    dynamic_layers_, static_layers_ = eqx.partition(layers, lambda x: eqx.is_array(x) and x.ndim > 0)
    dynamic_layers, static_layers = _aux_part_of(dynamic_layers_), _aux_part_of(static_layers_)

    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_remat.html#practical-notes
    @functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    def f(
        acc: tuple[A, KeyArray | None],
        dynamic_layer: eqx.PartOf[Callable[[A, *Rest, KeyArray | None], A]],
    ):
        old_x, acc_key = acc
        layer_key, acc_key = split_optional(acc_key, num=2)
        layer = eqx.combine(dynamic_layer, drop_aux_dim(static_layers))
        new_x = layer(old_x, *args, layer_key)
        return (new_x, layer_key), new_x

    (x, _), layer_outs = jax.lax.scan(f, (initial, initial_key), dynamic_layers)
    return x, layer_outs


def scan_layers_dropout_key(
    initial: A,
    layers: jax.AuxDim[NumLayers, Callable[Concatenate[A, P], A]],
    *args: P.args,
    # We expect there to be a jax PRNGKey in `dropout_key`
    **kwargs: P.kwargs,
) -> A:
    return scan_layers_dropout_key_with_intermediates(initial, layers, *args, **kwargs)[0]


def scan_layers_dropout_key_with_intermediates(
    initial: A,
    layers: jax.AuxDim[NumLayers, Callable[Concatenate[A, P], A]],
    *args: P.args,
    # We expect there to be a jax PRNGKey in `dropout_key`
    **kwargs: P.kwargs,
) -> tuple[A, jax.AuxDim[NumLayers, A]]:
    dynamic_layers_, static_layers_ = eqx.partition(layers, lambda x: eqx.is_array(x) and x.ndim > 0)
    dynamic_layers, static_layers = _aux_part_of(dynamic_layers_), _aux_part_of(static_layers_)

    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_remat.html#practical-notes
    @functools.partial(eqx.filter_checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    def f(
        acc: tuple[A, KeyArray | None],
        dynamic_layer: eqx.PartOf[Callable[Concatenate[A, P], A]],
    ):
        old_x, acc_key = acc
        layer_key, acc_key = split_optional(acc_key, num=2)
        layer = eqx.combine(dynamic_layer, drop_aux_dim(static_layers))
        new_x = layer(old_x, *args, **(kwargs | {"dropout_key": layer_key}))  # pyright: ignore
        return (new_x, layer_key), new_x

    (final, _), layer_outs = jax.lax.scan(f, (initial, cast(Any, kwargs["dropout_key"])), dynamic_layers)
    return final, layer_outs


Dim1 = TypeVar("Dim1", bound=int)
Dim2 = TypeVar("Dim2", bound=int)
Shape = TypeVarTuple("Shape")
Shape2 = TypeVarTuple("Shape2")
DType = TypeVar("DType")
DType2 = TypeVar("DType2")


def from_aux_dim(x: jax.AuxDim[Dim1, ndarray[*Shape, DType]]) -> ndarray[Dim1, *Shape, DType]:
    return x  # pyright: ignore


def to_aux_dim(x: ndarray[Dim1, *Shape, DType]) -> jax.AuxDim[Dim1, ndarray[*Shape, DType]]:
    return x  # pyright: ignore


def drop_aux_dim(x: jax.AuxDim[Dim1, A]) -> A:
    return x  # pyright: ignore


def init_linear_weight(model: A, key: KeyArray) -> A:
    def is_linear(x: Any):
        return isinstance(x, eqx.nn.Linear)

    def get_weights(m: A) -> list[ndarray[Any, Any, float]]:
        return [x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]

    weights = get_weights(model)
    new_weights = [
        jax.nn.initializers.he_normal(batch_axis=0, dtype=jax.dtypes.bfloat16)(
            subkey, weight.shape, jax.dtypes.bfloat16
        )
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    return eqx.tree_at(get_weights, model, replace=new_weights)


def per_tensor(
    writer: Callable[[str, A], None],
    num_layers: int,
    layer_keys: list[str],
    pytree: Any,
    metrics: Mapping[str, Callable[[ndarray[*tuple[Any, ...], float]], A]],
):
    for p, x in ((p, x) for p, x in jtu.tree_leaves_with_path(pytree) if eqx.is_array(x) and x.ndim > 0):
        # Case where we have a repeated layer that we've vmapped but we want to split again here for diagnostics
        if x.shape[0] == num_layers and any(l in key_path_as_str(p) for l in layer_keys):
            if x.size > num_layers:
                for i in range(num_layers):
                    for l, fn in metrics.items():
                        writer(f"{l}/{key_path_as_str(p)}/{i}", fn(x[i, ...]))
        # Case where we have a single layer
        elif x.size > 1:
            for l, fn in metrics.items():
                writer(f"{l}/{key_path_as_str(p)}", fn(x))


def deserialise_filter_spec(f: BinaryIO, x: A) -> A:
    if isinstance(x, jax.dtypes.bfloat16):
        return jax.dtypes.bfloat16(jnp.load(f).item())  # pyright: ignore
    else:
        return eqx.default_deserialise_filter_spec(f, x)


def to_bfloat16(x: Any) -> Any:
    if eqx.is_array(x):
        return x.astype(jax.dtypes.bfloat16)
    elif isinstance(x, float):
        return jax.dtypes.bfloat16(x)
    else:
        return x


Tuple = TypeVar("Tuple", bound=NamedTuple)


@overload
def double_vmap(
    f: Callable[[ndarray[*Shape, DType]], Tuple]
) -> Callable[[ndarray[Dim1, Dim2, *Shape, DType]], jax.AuxDim[Dim1, Dim2, Tuple]]:
    ...


@overload
def double_vmap(
    f: Callable[[ndarray[*Shape, DType]], ndarray[*Shape2, DType2]]
) -> Callable[[ndarray[Dim1, Dim2, *Shape, DType]], ndarray[Dim1, Dim2, *Shape2, DType2]]:
    ...


def double_vmap(f: Callable[[ndarray[*Shape, DType]], Any]) -> Callable[[ndarray[Dim1, Dim2, *Shape, DType]], Any]:
    # Pyright has trouble with the type signature unless we explicitly declare it
    return jax.vmap(jax.vmap(f))


def arrays_of(x: A) -> ArraysOf[A]:
    return cast(Any, eqx.filter(x, eqx.is_array))
