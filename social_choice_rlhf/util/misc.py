from __future__ import annotations

import functools
import subprocess
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import chain, islice, tee
from math import prod
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Generic,
    Literal,
    NewType,
    ParamSpec,
    TypeVar,
    TypeVarTuple,
    cast,
    overload,
)

from jax.numpy import ndarray

if TYPE_CHECKING:
    from numpy import Fin, Product, Sum

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
Int = TypeVar("Int", bound=int)
Dim1 = TypeVar("Dim1", bound=int)
Dim2 = TypeVar("Dim2", bound=int)
DType = TypeVar("DType")
Shape = TypeVarTuple("Shape")
Shape2 = TypeVarTuple("Shape2")
Float = TypeVar("Float", bound=float)
Label = TypeVar("Label", bound=str)
Left = TypeVar("Left", bound=int)
Middle = TypeVar("Middle", bound=int)
Right = TypeVar("Right", bound=int)


def has_gpu():
    try:
        _ = subprocess.check_output("nvidia-smi", stderr=subprocess.DEVNULL, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        return e.returncode != 127  # noqa: PLR2004


class declare(Generic[A]):  # noqa: N801
    """Helper for when pyright fails type inference. Weaker than `cast` in that it only picks out subtypes."""

    # You might think we can just do `def declare(_: type[A], x: A) -> A:`
    # but then we can't do forward references (i.e. stringified types) in the type hint

    def __new__(cls, x: A) -> A:
        return x


def flatten(l: Iterable[Iterable[A]]) -> list[A]:
    return list(chain.from_iterable(l))


def unzip(l: Iterable[tuple[A, B]]) -> tuple[Iterable[A], Iterable[B]]:
    a, b = tee(l)
    return (x[0] for x in a), (x[1] for x in b)


def unzip_list(l: Sequence[tuple[A, B]]) -> tuple[list[A], list[B]]:
    """Significantly faster than the naive `zip(*l)` implementation and somewhat faster than `unzip`"""
    a = cast(list[A], [None] * len(l))
    b = cast(list[B], [None] * len(l))
    for i, (x, y) in enumerate(l):
        a[i] = x
        b[i] = y
    return a, b


P = ParamSpec("P")


def partial1(f: Callable[Concatenate[B, P], A], b: B) -> Callable[P, A]:
    return cast(Any, functools.partial(f, b))


@overload
def partial(f: Callable[Concatenate[B, P], A], b: B) -> Callable[P, A]:
    ...


@overload
def partial(f: Callable[Concatenate[B, C, P], A], b: B, c: C) -> Callable[P, A]:
    ...


def partial(f: Callable[..., A], *args: Any, **kwargs: Any) -> Callable[..., A]:
    return functools.partial(f, *args, **kwargs)


gpu = has_gpu()


def human_bytes(size: float, decimal_places: int = 2) -> str:
    unit = "B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:  # noqa: B007
        if size < 1024.0:  # noqa: PLR2004
            break
        size /= 1024.0

    formatted_num = f"{size:.{decimal_places}f}".rstrip("0").rstrip(".")
    return f"{formatted_num:>4} {unit}"


def partition(pred: Callable[[A], bool], iterable: list[A]) -> tuple[list[A], list[A]]:
    true_part: list[A] = []
    false_part: list[A] = []
    for x in iterable:
        (true_part if pred(x) else false_part).append(x)
    return true_part, false_part


def batched(iterable: Iterable[A], n: int) -> Iterator[list[A]]:
    assert n > 0
    it = iter(iterable)
    while batch := list(islice(it, n)):
        if len(batch) < n:
            break
        else:
            yield batch


def batched_list(data: list[A], n: int) -> list[list[A]]:
    assert n > 0
    return [data[i : i + n] for i in range(0, len(data) // n * n, n)]


def format_time(x: float):
    if x < 1e-6:  # noqa: PLR2004
        return f"{x*1e9:.2f} ns"
    elif x < 1e-3:  # noqa: PLR2004
        return f"{x*1e6:.2f} us"
    elif x < 1:
        return f"{x*1e3:.2f} ms"
    else:
        return f"{x:.2f} s"


def product_(operands: tuple[*Shape]) -> Product[*Shape]:
    return cast(Any, prod(cast(tuple[int, ...], operands)))


def flatten_product(p: Product[Dim1, Product[*Shape]]) -> Product[Dim1, *Shape]:
    return cast(Any, p)


def sum_(operands: tuple[*Shape]) -> Sum[*Shape]:
    return cast(Any, sum(cast(tuple[int, ...], operands)))


class fin(int, Generic[Int]):  # noqa: N801
    def __new__(cls, x: int | ndarray[int], max_: Int) -> Fin[Int]:
        assert 0 <= x < max_
        return cast(Any, x)


@contextmanager
def time_segment(name: str):
    ts = time.perf_counter()
    yield
    print(f"Exiting {name} after: {format_time(time.perf_counter() - ts)}")


def drop_fin(x: ndarray[*Shape, Fin[Int]]) -> ndarray[*Shape, Int]:
    return cast(Any, x)


def fin_to_int(x: ndarray[*Shape, Fin[Int]]) -> ndarray[*Shape, int]:
    return cast(Any, x)


class Singleton(int, Generic[Label]):
    """Throughout the life of the program, there will only ever be one value corresponding to a label.
    i.e. You can be sure that any two `Singleton[Literal["Foo"]]` have the same value.
    """

    history = {}  # noqa: RUF012

    def __new__(cls, label: Label, value: int) -> Singleton[Label]:
        match cls.history.get(label):
            case None:
                cls.history[label] = value
                return cast(Any, value)
            case v:
                assert v == value, (label, v, value)
                return v


class InstanceSingleton(int, Generic[Label]):
    """As with `Singleton`, but we ensure that there's only one value with the label for a given instance.
    This is primarily used for defining "internal type variables".
    i.e. It's sometimes the case that the implementation of a class requires an annotating type
    (for e.g. an array dimension).
    But the class's user is uninterested in this type so we don't want to "pollute" the class with a type variable.
    (If we simply declare the type variable inside the class, pyright basically treats it as `Any`.)
    So instead we create an `InstanceSingleton` as the annotating type—it still gives us the guarantee
    a type variable would that any two arrays with this type for a dimension have the same size at runtime.
    """

    history: dict[tuple[int, str], int] = {}  # noqa: RUF012

    def __new__(cls, instance: Any, label: Label, value: int) -> InstanceSingleton[Label]:
        match cls.history.get((id(instance), label)):
            case None:
                cls.history[(id(instance), label)] = value
                return cast(Any, value)
            case v:
                assert v == value, (instance, label, v, value)
                return cast(Any, v)


class declare_dtype(Generic[DType]):  # noqa: N801
    def __new__(cls, array: ndarray[*Shape, Any]) -> ndarray[*Shape, DType]:
        return cast(Any, array)


class declare_axes(Generic[*Shape]):  # noqa: N801
    def __new__(cls, array: ndarray[*tuple[Any, ...], DType]) -> ndarray[*Shape, DType]:
        return cast(Any, array)


class declare_axis(Generic[A]):  # noqa: N801
    @overload
    def __new__(cls, axis: Literal[0], array: ndarray[Any, *Shape, DType]) -> ndarray[A, *Shape, DType]:
        ...

    @overload
    def __new__(
        cls, axis: Literal[1], array: ndarray[Dim1, Any, *Shape, DType]
    ) -> ndarray[Dim1, A, *Shape, DType]:
        ...

    @overload
    def __new__(
        cls, axis: Literal[2], array: ndarray[Dim1, Dim2, Any, *Shape, DType]
    ) -> ndarray[Dim1, Dim2, A, *Shape, DType]:
        ...

    def __new__(cls, axis: int, array: ndarray[*Shape, DType]) -> ndarray[*Shape, DType]:
        return array


def list_dicts_to_dict_lists(
    dicts: Sequence[Mapping[A, Any]],
) -> dict[A, list[Any]]:
    return {k: [d[k] for d in dicts] for k in dicts[0]}


def dict_lists_to_list_dicts(
    dict_: Mapping[str, Sequence[Any]],
) -> list[dict[str, Any]]:
    return [{k: dict_[k][i] for k in dict_} for i in range(len(next(iter(dict_.values()))))]


def from_maybe(x: A | None, default: B) -> A | B:
    return default if x is None else x


def from_maybe_ex(x: A | None) -> A:
    if x is None:
        raise Exception("Unexpected None")
    return x


# Yes, these look wrong, but, for some reason, things get flipped at call sites
def second_of_union(x: A | B) -> A:
    return cast(Any, x)


def first_of_union(x: A | B) -> B:
    return cast(Any, x)


Seconds = NewType("Seconds", float)


def our_lru_cache(maxsize: int | None = 128) -> Callable[[Callable[P, A]], Callable[P, A]]:
    return functools.lru_cache(maxsize)  # type: ignore
