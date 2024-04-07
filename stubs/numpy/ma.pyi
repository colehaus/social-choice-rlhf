# pylint: skip-file

from typing import Any, Generic, Literal, TypeVar, TypeVarTuple

import numpy as np

Shape = TypeVarTuple("Shape")
DType = TypeVar("DType")
Dim1 = TypeVar("Dim1", bound=int)
Dim2 = TypeVar("Dim2", bound=int)
Float = TypeVar("Float", bound=float)

def array(
    data: np.ndarray[*Shape, DType], mask: np.ndarray[*Shape, bool], fill_value: DType | None = None
) -> ndarray[*Shape, DType]: ...
def average(
    a: ndarray[Dim1, Dim2, *Shape, DType], axis: Literal[1], weights: ndarray[Dim1, Dim2, *Shape, DType]
) -> ndarray[Dim1, *Shape, DType]: ...
def power(a: ndarray[*Shape, Float], b: float) -> ndarray[*Shape, Float]: ...
def hstack(tup: tuple[ndarray[Dim1, Any, DType], np.ndarray[Dim1, Any, DType]]) -> ndarray[Dim1, Any, DType]: ...

class ndarray(Generic[*Shape, DType]):
    data: np.ndarray[*Shape, DType]
    shape: tuple[*Shape]
    def __rtruediv__(self, other: float) -> ndarray[*Shape, DType]: ...
    def __add__(self, other: float) -> ndarray[*Shape, DType]: ...
    def set_fill_value(self, value: DType) -> None: ...
    def filled(self, fill_value: DType) -> np.ndarray[*Shape, DType]: ...
