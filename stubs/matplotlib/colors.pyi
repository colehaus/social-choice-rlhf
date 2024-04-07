from collections.abc import Sequence

class Colormap: ...

class ListedColorMap(Colormap):
    colors: Sequence[tuple[float, float, float]]
