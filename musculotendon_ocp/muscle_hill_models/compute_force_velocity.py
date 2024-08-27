from enum import Enum

from casadi import MX, log as logn, sqrt, sinh, log10

from .muscle_hill_model_abstract import ComputeForceVelocity


"""
Implementations of the ComputeForceVelocity protocol
"""


class ComputeForceVelocityHillType:
    def __init__(
        self,
        d1: float = -0.318,
        d2: float = -8.149,
        d3: float = -0.374,
        d4: float = 0.886,
    ):
        # TODO The default values may need to be more precise
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4

    def __call__(self, normalized_muscle_fiber_velocity: MX) -> MX:
        # alias so the next line is not too long
        velocity = normalized_muscle_fiber_velocity

        return (
            self.d1 * logn((self.d2 * velocity + self.d3) + sqrt(((self.d2 * velocity + self.d3) ** 2) + 1)) + self.d4
        )

    def inverse(self, force_velocity_inverse: MX) -> MX:
        return (1 / self.d2) * (sinh((1 / self.d1) * (force_velocity_inverse - self.d4)) - self.d3)

    def derivative(self, normalized_muscle_fiber_velocity) -> tuple[MX, MX]:
        p = normalized_muscle_fiber_velocity
        return (self.d1 * self.d2) / sqrt(1 + (self.d3 + self.d2 * p) ** (2))


class ComputeForceVelocityMethods(Enum):
    HillType = ComputeForceVelocityHillType

    def __call__(self, *args, **kwargs) -> ComputeForceVelocity:
        return self.value(*args, **kwargs)
