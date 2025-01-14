from typing import override, Self
from casadi import MX, exp

from .compute_muscle_fiber_length import (
    ComputeMuscleFiberLengthRigidTendon,
    ComputeMuscleFiberLengthInstantaneousEquilibrium,
)
from .compute_muscle_fiber_velocity import (
    ComputeMuscleFiberVelocityRigidTendon,
    ComputeMuscleFiberVelocityFlexibleTendonFromForceDefects,
)
from .muscle_hill_model_rigid_tendon import MuscleHillModelRigidTendon
from .muscle_hill_model_abstract import ComputeMuscleFiberLength, ComputeMuscleFiberVelocity


class MuscleHillModelFlexibleTendon(MuscleHillModelRigidTendon):
    def __init__(
        self,
        name: str,
        c1: float = 0.2,
        c2: float = 0.995,
        c3: float = 0.250,
        kt: float = 35.0,
        compute_muscle_fiber_length: ComputeMuscleFiberLength | None = None,
        compute_muscle_fiber_velocity: ComputeMuscleFiberVelocity | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        tendon_slack_length: MX
            The tendon slack length
        """
        if compute_muscle_fiber_length is None:
            compute_muscle_fiber_length = ComputeMuscleFiberLengthInstantaneousEquilibrium()
        if isinstance(compute_muscle_fiber_length, ComputeMuscleFiberLengthRigidTendon):
            raise ValueError("The compute_muscle_fiber_length must be a flexible tendon")

        if compute_muscle_fiber_velocity is None:
            compute_muscle_fiber_velocity = ComputeMuscleFiberVelocityFlexibleTendonFromForceDefects()
        if isinstance(compute_muscle_fiber_velocity, ComputeMuscleFiberVelocityRigidTendon):
            raise ValueError("The compute_muscle_fiber_velocity must be a flexible tendon")

        super().__init__(
            name=name,
            compute_muscle_fiber_length=compute_muscle_fiber_length,
            compute_muscle_fiber_velocity=compute_muscle_fiber_velocity,
            **kwargs,
        )

        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.kt = kt

    @override
    @property
    def copy(self) -> Self:
        return MuscleHillModelFlexibleTendon(
            name=self.name,
            c1=self.c1,
            c2=self.c2,
            c3=self.c3,
            kt=self.kt,
            maximal_force=self.maximal_force,
            optimal_length=self.optimal_length,
            tendon_slack_length=self.tendon_slack_length,
            maximal_velocity=self.maximal_velocity,
            label=self.label,
            compute_force_passive=self.compute_force_passive.copy,
            compute_force_active=self.compute_force_active.copy,
            compute_force_velocity=self.compute_force_velocity.copy,
            compute_force_damping=self.compute_force_damping.copy,
            compute_pennation_angle=self.compute_pennation_angle.copy,
            compute_muscle_fiber_length=self.compute_muscle_fiber_length.copy,
            compute_muscle_fiber_velocity=self.compute_muscle_fiber_velocity.copy,
        )

    def normalize_tendon_length(self, tendon_length: MX) -> MX:
        return tendon_length / self.tendon_slack_length

    @override
    def denormalize_tendon_length(self, normalized_tendon_length: MX) -> MX:
        return normalized_tendon_length * self.tendon_slack_length

    @override
    def compute_tendon_length(self, muscle_tendon_length: MX, muscle_fiber_length: MX) -> MX:
        return muscle_tendon_length - self.compute_pennation_angle.apply(muscle_fiber_length, muscle_fiber_length)

    @override
    def compute_tendon_force(self, tendon_length: MX) -> MX:
        normalized_tendon_length = self.normalize_tendon_length(tendon_length)

        return self.c1 * exp(self.kt * (normalized_tendon_length - self.c2)) - self.c3


class MuscleHillModelFlexibleTendonAlwaysPositive(MuscleHillModelFlexibleTendon):
    @property
    def offset(self) -> float:
        """
        Get the offset to ensure the tendon force is always positive, by offsetting the force by the value at slack length
        """
        return super(MuscleHillModelFlexibleTendonAlwaysPositive, self).compute_tendon_force(self.tendon_slack_length)

    @override
    def compute_tendon_force(self, tendon_length: MX) -> MX:
        return (
            super(MuscleHillModelFlexibleTendonAlwaysPositive, self).compute_tendon_force(tendon_length) - self.offset
        )
