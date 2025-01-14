from enum import Enum
from typing import Self

from casadi import MX, cos, sin, asin

from .muscle_hill_model_abstract import ComputePennationAngle


"""
Implementations of the ComputePennationAngle protocol
"""


class ComputePennationAngleConstant:
    def __init__(self, pennation_angle: float = 0.0):
        if pennation_angle < 0.0:
            raise ValueError("The pennation angle must be positive")
        self.pennation_angle = pennation_angle

    @property
    def copy(self) -> Self:
        return ComputePennationAngleConstant(pennation_angle=self.pennation_angle)

    def __call__(self, muscle_fiber_length: MX) -> MX:
        return self.pennation_angle

    def apply(self, muscle_fiber_length: MX, element: MX) -> MX:
        return cos(self(muscle_fiber_length)) * element

    def remove(self, muscle_fiber_length: MX, element: MX) -> MX:
        return element / cos(self(muscle_fiber_length))


class ComputePennationAngleWrtMuscleFiberLength:
    def __init__(self, optimal_pennation_angle: float = 0.0, optimal_muscle_fiber_length: float = 0.0):
        if optimal_pennation_angle < 0.0:
            raise ValueError("The optimal pennation angle must be positive")

        self.optimal_pennation_angle = optimal_pennation_angle
        self.optimal_muscle_fiber_length = optimal_muscle_fiber_length

    @property
    def copy(self) -> Self:
        return ComputePennationAngleWrtMuscleFiberLength(
            optimal_pennation_angle=self.optimal_pennation_angle,
            optimal_muscle_fiber_length=self.optimal_muscle_fiber_length,
        )

    def __call__(self, muscle_fiber_length: MX) -> MX:
        return asin(self.optimal_muscle_fiber_length * sin(self.optimal_pennation_angle) / muscle_fiber_length)

    def apply(self, muscle_fiber_length: MX, element: MX) -> MX:
        return cos(self(muscle_fiber_length)) * element

    def remove(self, muscle_fiber_length: MX, element: MX) -> MX:
        return element / cos(self(muscle_fiber_length))


class ComputePennationAngleMethods(Enum):
    Constant = ComputePennationAngleConstant
    WrtMuscleFiberLength = ComputePennationAngleWrtMuscleFiberLength

    def __call__(self, *args, **kwargs) -> ComputePennationAngle:
        return self.value(*args, **kwargs)
