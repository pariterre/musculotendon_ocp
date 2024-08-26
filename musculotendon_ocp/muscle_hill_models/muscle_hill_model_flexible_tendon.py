from typing import override

from casadi import MX, exp, cos

from .compute_muscle_fiber_length import (
    ComputeMuscleFiberLengthRigidTendon,
    ComputeMuscleFiberLengthInstantaneousEquilibrium,
)
from .compute_muscle_fiber_velocity import (
    ComputeMuscleFiberVelocityRigidTendon,
    ComputeMuscleFiberVelocityFlexibleTendonImplicit,
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
            compute_muscle_fiber_velocity = ComputeMuscleFiberVelocityFlexibleTendonImplicit()
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
    def compute_muscle_force_velocity_inverse(self, activation: MX, muscle_fiber_length: MX, tendon_length: MX) -> MX:
        # Get the normalized muscle length and velocity
        normalized_length = self.normalize_muscle_fiber_length(muscle_fiber_length)

        # Compute the passive, active, velocity and damping factors
        pennation_angle = self.compute_pennation_angle(muscle_fiber_length)
        force_passive = self.compute_force_passive(normalized_length)
        force_active = self.compute_force_active(normalized_length)
        force_damping = self.compute_force_damping(1)
        if force_damping != 0:
            raise NotImplementedError("Damping with value != 0 is not implemented yet")

        tendon_force = self.compute_tendon_force(tendon_length)
        fv_inverse = self.compute_muscle_fiber_velocity.inverse(
            activation, pennation_angle, force_passive, force_active, force_damping, tendon_force
        )

        return self.maximal_velocity * muscle._force_velocity_inverse(
            (tendon_force / cos(pennation_angle) - force_passive - force_damping) / (activation * force_active)
        )
        # fv_inv = (self.ft(tendon_length_normalized) / casadi.cos(pennationAngle) - self.fpas(lm_normalized)) / (
        #     activation * self.fact(lm_normalized)
        # )

        # vm_normalized = 1 / d2 * (casadi.sinh(1 / d1 * (fv_inv - d4)) - d3)
        # return vm_normalized

    @override
    def normalize_tendon_length(self, tendon_length: MX) -> MX:
        return tendon_length / self.tendon_slack_length

    @override
    def compute_tendon_length(self, muscle_tendon_length: MX, muscle_fiber_length: MX) -> MX:
        return muscle_tendon_length - muscle_fiber_length

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
