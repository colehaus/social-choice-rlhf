# pylint: skip-file

from typing import Sequence

class Line2D:
    def __init__(
        self,
        xdata: Sequence[float],
        ydata: Sequence[float],
        color: tuple[float, float, float],
        linestyle: str,
        lw: float,
    ) -> None: ...
    def get_color(self) -> tuple[float, float, float]: ...
    def get_label(self) -> str: ...
    def get_linestyle(self) -> str: ...
