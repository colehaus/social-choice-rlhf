# pylint: skip-file

from typing import TypeVar, TypeVarTuple

from numpy import ndarray

Shape = TypeVarTuple("Shape")
DType = TypeVar("DType")

def assert_array_equal(x: ndarray[*Shape, DType], y: ndarray[*Shape, DType], strict: bool = False) -> None: ...
