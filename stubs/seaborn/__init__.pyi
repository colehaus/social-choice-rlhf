# pylint: skip-file

from typing import Any, Literal, Sequence, TypeVar, overload

from matplotlib.axes import Axes
from numpy import ndarray

class FacetGrid:
    def set(self, yscale: Literal["log"]) -> None: ...
    ax: Axes

IndexT = TypeVar("IndexT")
ColumnT = TypeVar("ColumnT")
DType = TypeVar("DType")
Numeric = TypeVar("Numeric", float, int)
A = TypeVar("A")

Marker = Literal["_"]

def color_palette() -> Sequence[tuple[float, float, float]]: ...
def relplot(
    x: ndarray[Any, Numeric],
    y: ndarray[Any, Numeric],
    kind: Literal["line", "scatter"] = "scatter",
    hue: ndarray[Numeric] | None = None,
    alpha: float = 1,
    height: float = 5,
    aspect: float = 1,
) -> FacetGrid: ...

Scalar = TypeVar("Scalar", float, int)
Scalar2 = TypeVar("Scalar2", float, int)

def scatterplot(
    x: Sequence[Scalar] | ndarray[Any, Scalar],
    y: Sequence[Scalar2] | ndarray[Any, Scalar2],
    alpha: float = 1,
    hue: str | ndarray[Any, float] | list[A] | None = None,
    ax: Axes | None = None,
    label: str | None = None,
    s: float = ...,
) -> Axes: ...
def lineplot(
    x: ndarray[Any, Scalar] | Sequence[Scalar],
    y: ndarray[Any, Scalar2] | Sequence[Scalar2],
    alpha: float = 1,
    ax: Axes | None = None,
    label: str | None = None,
    color: tuple[float, float, float] | None = None,
    legend: Literal["auto", "brief", "full", False] = "auto",
    zorder: float | None = None,
    ci: None = None,
) -> Axes: ...
@overload
def kdeplot(
    data: ndarray[Any, Numeric],
    x: ColumnT | None = None,
    hue: ColumnT | None = None,
    alpha: float | None = 1,
    ax: Axes | None = None,
    fill: bool = False,
    thresh: float = 0.05,
    levels: int = 10,
    cmap: Literal["viridis"] | None = None,
    cut: float | None = 3,
) -> Axes: ...
@overload
def kdeplot(
    x: ndarray[Any, float],
    y: ndarray[Any, float],
    hue: ColumnT | None = None,
    alpha: float | None = 1,
    ax: Axes | None = None,
    fill: bool = False,
    thresh: float = 0.05,
    levels: int = 10,
    cmap: Literal["viridis"] | None = None,
    cut: float | None = 3,
    clip: tuple[float, float] | tuple[tuple[float, float] | tuple[float, float]] | None = None,
) -> Axes: ...
def set_theme() -> None: ...
