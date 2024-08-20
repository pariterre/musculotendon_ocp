from typing import override

from casadi import MX, exp, Function, rootfinder

from .muscle_model_hill_rigid_tendon import MuscleModelHillRigidTendon
from ..muscle_model_abstract import ComputeMuscleFiberLengthCallable, ComputeMuscleFiberVelocityCallable
from ..compute_muscle_fiber_length import (
    ComputeMuscleFiberLengthRigidTendon,
    ComputeMuscleFiberLengthInstantaneousEquilibrium,
)
from ..compute_muscle_fiber_velocity import (
    ComputeMuscleFiberVelocityRigidTendon,
    ComputeMuscleFiberVelocityFlexibleTendon,
)


class MuscleModelHillFlexibleTendon(MuscleModelHillRigidTendon):
    def __init__(
        self,
        c1: float = 0.2,
        c2: float = 0.995,
        c3: float = 0.250,
        kt: float = 35.0,
        compute_muscle_fiber_length: ComputeMuscleFiberLengthCallable = ComputeMuscleFiberLengthInstantaneousEquilibrium(),
        compute_muscle_fiber_velocity: ComputeMuscleFiberVelocityCallable = ComputeMuscleFiberVelocityFlexibleTendon(),
        **kwargs,
    ):
        """
        Parameters
        ----------
        tendon_slack_length: MX
            The tendon slack length
        """
        if isinstance(compute_muscle_fiber_length, ComputeMuscleFiberLengthRigidTendon):
            raise ValueError("The compute_muscle_fiber_length must be a flexible tendon")

        if isinstance(compute_muscle_fiber_velocity, ComputeMuscleFiberVelocityRigidTendon):
            raise ValueError("The compute_muscle_fiber_velocity must be a flexible tendon")

        super().__init__(
            compute_muscle_fiber_length=compute_muscle_fiber_length,
            compute_muscle_fiber_velocity=compute_muscle_fiber_velocity,
            **kwargs,
        )

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.kt = kt

    @override
    def normalize_tendon_length(self, tendon_length: MX) -> MX:
        return tendon_length / self.tendon_slack_length

    @override
    def compute_tendon_force(self, tendon_length: MX) -> MX:
        normalized_tendon_length = self.normalize_tendon_length(tendon_length)
        offset = 0.01175075667752834

        return self.c1 * exp(self.kt * (normalized_tendon_length - self.c2)) - self.c3 + offset
