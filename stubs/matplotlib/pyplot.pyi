# pylint: skip-file

from collections.abc import Sequence
from io import BytesIO
from typing import Any, Literal, TypeVar, overload

from matplotlib.colors import Colormap
from numpy import ndarray

from .axes import Axes
from .figure import Figure
from .text import Annotation

Numeric = TypeVar("Numeric", float, int)
Numeric2 = TypeVar("Numeric2", float, int)
IndexT = TypeVar("IndexT")
ColumnT = TypeVar("ColumnT")
Dim1 = TypeVar("Dim1", bound=int)
Dim2 = TypeVar("Dim2", bound=int)

def get_cmap() -> Colormap: ...
def title(title: str) -> None: ...
def suptitle(title: str) -> None: ...
@overload
def subplots(
    sharex: bool = False,
    sharey: bool = False,
    figsize: tuple[float, float] = (6.4, 4.8),
    subplot_kw: dict[str, Any] = ...,
) -> tuple[Figure, Axes]: ...
@overload
def subplots(
    ncols: Dim1,
    figsize: tuple[float, float] = (6.4, 4.8),
) -> tuple[Figure, ndarray[Dim1, Axes]]: ...
@overload
def subplots(
    nrows: Dim1,
    figsize: tuple[float, float] = (6.4, 4.8),
    subplot_kw: dict[str, Any] = ...,
) -> tuple[Figure, ndarray[Dim1, Axes]]: ...
@overload
def subplots(
    nrows: Dim1 = 1,
    ncols: Dim2 = 1,
    sharex: bool = False,
    sharey: bool = False,
    figsize: tuple[float, float] = (6.4, 4.8),
) -> tuple[Figure, ndarray[Dim1, Dim2, Axes]]: ...
def figure(figsize: tuple[float, float], tight_layout: bool = False) -> Figure: ...
def savefig(fname: str | BytesIO, format: str | None = None) -> None: ...
@overload
def close() -> None: ...
@overload
def close(fig: Literal["all"] | Figure) -> None: ...
def legend(loc: str) -> None: ...
@overload
def plot(
    data: ndarray[Any, Any, Numeric],
    alpha: float = 1,
    label: str | Sequence[str] | Sequence[float] | None = ...,
    lw: float = ...,
) -> Figure: ...
@overload
def plot(
    x: ndarray[Any, Numeric],
    y: ndarray[Any, Numeric2],
    alpha: float = 1,
    label: str | Sequence[str] | Sequence[float] | None = ...,
    lw: float = ...,
) -> Figure: ...
def annotate(label: str, pos: tuple[float, float]) -> Annotation: ...
def gca() -> Axes: ...
def gcf() -> Figure: ...
def axes(projection: Literal["3d"]) -> Axes: ...
def contourf(
    X: ndarray[Any, float],
    Y: ndarray[Any, float],
    Z: ndarray[Any, float],
    levels: int | None = None,
    cmap: Literal["viridis"] | None = None,
) -> None: ...
def subplots_adjust(
    left: float, bottom: float, right: float, top: float, wspace: float, hspace: float
) -> None: ...
def show() -> None: ...
