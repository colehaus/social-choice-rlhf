from typing import Any, Literal, Sequence, TypedDict, TypeVar, overload

from matplotlib.transforms import Transform
from numpy import ndarray

from .axis import Axis
from .figure import Figure
from .legend import Legend
from .lines import Line2D
from .patches import Patch
from .text import Annotation, Text

IndexT = TypeVar("IndexT")
ColumnT = TypeVar("ColumnT")
Float = TypeVar("Float", bound=float)
Float2 = TypeVar("Float2", bound=float)

class ArrowProps(TypedDict):
    arrowstyle: Literal["->"]
    color: Literal["black"]

class Axes:
    xaxis: Axis
    figure: Figure
    flat: ndarray[Any, Axes]
    patch: Patch
    transAxes: Transform
    def grid(self, grid: bool) -> None: ...
    def set_zorder(self, zorder: float) -> None: ...
    def annotate(
        self,
        text: str,
        xy: tuple[float, float],
        xytext: tuple[float, float],
        fontsize: float,
        arrowprops: ArrowProps,
    ) -> Annotation: ...
    def get_legend_handles_labels(self) -> tuple[Sequence[Line2D], Sequence[str]]: ...
    @overload
    def legend(self, handles: Sequence[Line2D], labels: Sequence[str]) -> Legend: ...
    @overload
    def legend(self, labels: Sequence[str] | None = ..., title: str | None = None) -> Legend: ...
    @overload
    def legend(
        self,
        loc: Literal[
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "center left",
            "center right",
            "lower center",
            "upper center",
        ],
    ) -> Legend: ...
    def get_lines(self) -> Sequence[Line2D]: ...
    def get_xticklabels(self) -> Sequence[Text]: ...
    def text(
        self,
        x: float,
        y: float,
        s: str,
        transform: Transform | None = None,
        rotation: float | None = None,
        alpha: float | None = None,
    ) -> Text: ...
    def set_xlabel(self, xlabel: str | None) -> None: ...
    def set_ylabel(self, ylabel: str | None) -> None: ...
    def set_yscale(self, value: Literal["log"]) -> None: ...
    def set_title(self, t: str) -> None: ...
    def twiny(self) -> Axes: ...
    def twinx(self) -> Axes: ...
    def plot(
        self,
        data: ndarray[Any, Float],
        alpha: float = 1,
    ) -> None: ...
    def fill_between(
        self,
        x: ndarray[Any, Float],
        y1: ndarray[Any, Float2],
        y2: ndarray[Any, Float2],
        alpha: float = 1,
        color: tuple[float, float, float] | None = None,
    ) -> object: ...
    def plot_surface(
        self,
        X: ndarray[Any, float],
        Y: ndarray[Any, float],
        Z: ndarray[Any, float],
        cmap: Literal["viridis", "magma"],
        edgecolor: Literal["none"],
        alpha: float = 1,
    ) -> None: ...
    def contourf(
        self,
        X: ndarray[Any, float],
        Y: ndarray[Any, float],
        Z: ndarray[Any, float],
        levels: int | None = None,
        cmap: Literal["viridis"] | None = None,
        zdir: Literal["z"] | None = None,
        offset: float | None = None,
    ) -> None: ...
    def set_xticklabels(self, labels: Sequence[str]) -> None: ...
    def set_yticklabels(self, labels: Sequence[str]) -> None: ...
    def set_zlim(self, bottom: float, top: float) -> None: ...
    def set_xlim(self, left: float, right: float) -> None: ...
    def set_ylim(self, bottom: float, top: float) -> None: ...
    def get_ylim(self) -> tuple[float, float]: ...
    def view_init(self, elev: float | None = None, azim: float | None = None) -> None: ...
    def axvline(self, x: float = 0, color: tuple[float, float, float] | None = None) -> Line2D: ...
    def scatter(
        self,
        x: ndarray[Any, Float],
        y: ndarray[Any, Float],
        z: ndarray[Any, Float] | None = None,
        c: list[int] | None = None,
        color: tuple[float, float, float] | None = None,
        label: str | None = None,
        marker: Literal["o", "x", "<", ">", "^", "v", "1", "2", "3", "4"] | None = None,
        s: float | None = None,
    ) -> None: ...
