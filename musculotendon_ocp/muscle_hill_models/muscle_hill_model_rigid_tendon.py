from typing import override

from casadi import MX

from .compute_pennation_angle import ComputePennationAngleConstant
from .compute_force_active import ComputeForceActiveHillType
from .compute_force_damping import ComputeForceDampingConstant
from .compute_force_passive import ComputeForcePassiveHillType
from .compute_force_velocity import ComputeForceVelocityHillType
from .compute_muscle_fiber_length import ComputeMuscleFiberLengthRigidTendon
from .compute_muscle_fiber_velocity import ComputeMuscleFiberVelocityRigidTendon
from .muscle_hill_model_abstract import (
    MuscleHillModelAbstract,
    ComputeForcePassive,
    ComputeForceActive,
    ComputeForceVelocity,
    ComputeForceDamping,
    ComputeMuscleFiberLength,
    ComputeMuscleFiberVelocity,
    ComputePennationAngle,
)


class MuscleHillModelRigidTendon(MuscleHillModelAbstract):
    def __init__(
        self,
        name: str,
        maximal_force: MX,
        optimal_length: MX,
        tendon_slack_length: MX,
        maximal_velocity: MX,
        compute_force_passive: ComputeForcePassive | None = None,
        compute_force_active: ComputeForceActive | None = None,
        compute_force_velocity: ComputeForceVelocity | None = None,
        compute_force_damping: ComputeForceDamping | None = None,
        compute_pennation_angle: ComputePennationAngle | None = None,
        compute_muscle_fiber_length: ComputeMuscleFiberLength | None = None,
        compute_muscle_fiber_velocity: ComputeMuscleFiberVelocity | None = None,
    ):
        compute_pennation_angle = (
            ComputePennationAngleConstant() if compute_pennation_angle is None else compute_pennation_angle
        )
        compute_force_passive = (
            ComputeForcePassiveHillType() if compute_force_passive is None else compute_force_passive
        )
        compute_force_active = ComputeForceActiveHillType() if compute_force_active is None else compute_force_active
        compute_force_velocity = (
            ComputeForceVelocityHillType() if compute_force_velocity is None else compute_force_velocity
        )
        compute_force_damping = (
            ComputeForceDampingConstant() if compute_force_damping is None else compute_force_damping
        )
        compute_muscle_fiber_length = (
            ComputeMuscleFiberLengthRigidTendon()
            if compute_muscle_fiber_length is None
            else compute_muscle_fiber_length
        )
        compute_muscle_fiber_velocity = (
            ComputeMuscleFiberVelocityRigidTendon()
            if compute_muscle_fiber_velocity is None
            else compute_muscle_fiber_velocity
        )

        super().__init__(
            name=name,
            maximal_force=maximal_force,
            optimal_length=optimal_length,
            tendon_slack_length=tendon_slack_length,
            maximal_velocity=maximal_velocity,
            compute_force_passive=compute_force_passive,
            compute_force_active=compute_force_active,
            compute_force_velocity=compute_force_velocity,
            compute_force_damping=compute_force_damping,
            compute_pennation_angle=compute_pennation_angle,
            compute_muscle_fiber_length=compute_muscle_fiber_length,
            compute_muscle_fiber_velocity=compute_muscle_fiber_velocity,
        )

    @override
    def normalize_muscle_fiber_length(self, muscle_fiber_length: MX) -> MX:
        return muscle_fiber_length / self.optimal_length

    @override
    def denormalize_muscle_fiber_length(self, normalized_muscle_fiber_length: MX) -> MX:
        return normalized_muscle_fiber_length * self.optimal_length

    @override
    def normalize_muscle_fiber_velocity(self, muscle_fiber_velocity: MX) -> MX:
        return muscle_fiber_velocity / self.maximal_velocity

    @override
    def denormalize_muscle_fiber_velocity(self, normalized_muscle_fiber_velocity: MX) -> MX:
        return normalized_muscle_fiber_velocity * self.maximal_velocity

    @override
    def normalize_tendon_length(self, tendon_length: MX) -> MX:
        raise RuntimeError("The tendon length should not be normalized with a rigid tendon")

    @override
    def denormalize_tendon_length(self, normalized_tendon_length: MX) -> MX:
        raise RuntimeError("The tendon length should not be denormalized with a rigid tendon")

    @override
    def compute_muscle_force(self, activation: MX, muscle_fiber_length: MX, muscle_fiber_velocity: MX) -> MX:
        # Get the normalized muscle length and velocity
        normalized_length = self.normalize_muscle_fiber_length(muscle_fiber_length)
        normalized_velocity = self.normalize_muscle_fiber_velocity(muscle_fiber_velocity)

        # Compute the passive, active, velocity and damping factors
        force_passive = self.compute_force_passive(normalized_length)
        force_active = self.compute_force_active(normalized_length)
        force_velocity = self.compute_force_velocity(normalized_velocity)
        force_damping = self.compute_force_damping(normalized_velocity)

        # TODO Are we supposed to apply pennation here?
        return self.compute_pennation_angle.apply(
            muscle_fiber_length,
            self.maximal_force * (force_passive + activation * force_active * force_velocity + force_damping),
        )

    @override
    def compute_muscle_fiber_velocity_from_inverse(
        self, activation: MX, muscle_fiber_length: MX, muscle_fiber_velocity: MX, tendon_length: MX
    ) -> MX:
        raise RuntimeError("The inverse of muscle fiber velocity should not be computed with a rigid tendon")

    @override
    def compute_muscle_fiber_velocity_from_linear_approximation(
        self, activation: MX, muscle_fiber_length: MX, muscle_fiber_velocity: MX, tendon_length: MX
    ) -> MX:
        raise RuntimeError(
            "The linear approximation of muscle fiber velocity should not be computed with a rigid tendon"
        )

    @override
    def compute_tendon_length(self, muscle_tendon_length: MX, muscle_fiber_length: MX) -> MX:
        return self.tendon_slack_length

    @override
    def compute_tendon_force(self, tendon_length: MX) -> MX:
        return 0.0
