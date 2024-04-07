from __future__ import annotations

from typing import Generic, Literal, Self, TypeVar

from numpy import ndarray

NumFeatures = TypeVar("NumFeatures", bound=int)
NumSamples = TypeVar("NumSamples", bound=int)
NumComponents = TypeVar("NumComponents", bound=int)
Float = TypeVar("Float", bound=float)

class GaussianMixture(Generic[NumComponents, NumFeatures]):
    def __init__(
        self,
        n_components: NumComponents,
        *,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
    ) -> None: ...
    def fit(self, X: ndarray[NumSamples, NumFeatures, Float]) -> Self: ...
    def sample(
        self, n_samples: NumSamples
    ) -> tuple[ndarray[NumSamples, NumFeatures, float], ndarray[NumSamples, int]]: ...
    means_: ndarray[NumComponents, NumFeatures, float]
    weights_: ndarray[NumComponents, float]
