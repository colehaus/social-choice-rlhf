# pylint: skip-file

from typing import Callable, Sequence

from jax import DType, Shape
from jax.numpy import ndarray
from jax.random import KeyArray

def he_normal(
    batch_axis: Sequence[int] | int, dtype: type[DType]
) -> Callable[[KeyArray, tuple[*Shape], type[DType]], ndarray[*Shape, DType]]: ...
