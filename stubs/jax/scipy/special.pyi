# pylint: skip-file

from typing import TypeVar, TypeVarTuple, overload

from numpy import ndarray

Shape = TypeVarTuple("Shape")
Float = TypeVar("Float", bound=float)
Dim1 = TypeVar("Dim1", bound=int)

@overload
def gammaln(x: float) -> float: ...
@overload
def gammaln(x: ndarray[*Shape, Float]) -> ndarray[*Shape, Float]: ...
def expit(x: ndarray[*Shape, Float]) -> ndarray[*Shape, Float]: ...
def entr(x: ndarray[Dim1, Float]) -> ndarray[Dim1, Float]: ...
def logsumexp(x: ndarray[*Shape, Float]) -> ndarray[Float]: ...
