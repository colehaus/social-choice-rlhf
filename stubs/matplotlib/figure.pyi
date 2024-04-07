# pylint: skip-file

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Literal, overload

from .axes import Axes
from .text import Text

class Figure:
    axes: Sequence[Axes]
    def gca(self) -> Axes: ...
    def legend(self, labels: Sequence[str], loc: str = "best") -> None: ...
    def tight_layout(self) -> None: ...
    def set_tight_layout(self, after_every_draw: bool = False) -> None: ...
    def suptitle(self, t: str) -> Text: ...
    @overload
    def savefig(self, fname: BytesIO, format: Literal["png"]) -> None: ...
    @overload
    def savefig(
        self, fname: str | Path, bbox_inches: str | None = None, pad_inches: float = 0.1, transparent: bool = False
    ) -> None: ...
    def add_subplot(self, nrows: int, ncols: int, index: int, projection: Literal["3d"] | None = None) -> Axes: ...
    def subplots_adjust(
        self,
        left: float | None = None,
        bottom: float | None = None,
        right: float | None = None,
        top: float | None = None,
        wspace: float | None = None,
        hspace: float | None = None,
    ) -> None: ...
    def show(self) -> None: ...
