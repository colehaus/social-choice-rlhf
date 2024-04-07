# pylint: skip-file

from typing import TypeVar, TypeVarTuple

from numpy import ndarray

Float = TypeVar("Float", bound=float)
Shape = TypeVarTuple("Shape")

def norm(x: ndarray[*Shape, Float], ord: str | int | None = 2) -> ndarray[Float]: ...
