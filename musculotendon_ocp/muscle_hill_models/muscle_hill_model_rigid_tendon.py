from typing import override, Self

from casadi import MX

from .compute_pennation_angle import ComputePennationAngleMethods
from .compute_force_active import ComputeForceActiveMethods
from .compute_force_damping import ComputeForceDampingMethods
from .compute_force_passive import ComputeForcePassiveMethods
from .compute_force_velocity import ComputeForceVelocityMethods
from .compute_muscle_fiber_length import ComputeMuscleFiberLengthMethods
from .compute_muscle_fiber_velocity import ComputeMuscleFiberVelocityMethods
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
        label: str = None,
        compute_force_passive: ComputeForcePassive | None = None,
        compute_force_active: ComputeForceActive | None = None,
        compute_force_velocity: ComputeForceVelocity | None = None,
        compute_force_damping: ComputeForceDamping | None = None,
        compute_pennation_angle: ComputePennationAngle | None = None,
        compute_muscle_fiber_length: ComputeMuscleFiberLength | None = None,
        compute_muscle_fiber_velocity: ComputeMuscleFiberVelocity | None = None,
    ):
        compute_pennation_angle = (
            ComputePennationAngleMethods.Constant() if compute_pennation_angle is None else compute_pennation_angle
        )
        compute_force_passive = (
            ComputeForcePassiveMethods.HillType() if compute_force_passive is None else compute_force_passive
        )
        compute_force_active = (
            ComputeForceActiveMethods.HillType() if compute_force_active is None else compute_force_active
        )
        compute_force_velocity = (
            ComputeForceVelocityMethods.HillType() if compute_force_velocity is None else compute_force_velocity
        )
        compute_force_damping = (
            ComputeForceDampingMethods.Constant() if compute_force_damping is None else compute_force_damping
        )
        compute_muscle_fiber_length = (
            ComputeMuscleFiberLengthMethods.RigidTendon()
            if compute_muscle_fiber_length is None
            else compute_muscle_fiber_length
        )
        compute_muscle_fiber_velocity = (
            ComputeMuscleFiberVelocityMethods.RigidTendon()
            if compute_muscle_fiber_velocity is None
            else compute_muscle_fiber_velocity
        )

        super(MuscleHillModelRigidTendon, self).__init__(
            name=name,
            label=label,
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
    @property
    def copy(self) -> Self:
        return MuscleHillModelRigidTendon(
            name=self.name,
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

    @override
    def serialize(self) -> dict:
        return {**super(MuscleHillModelRigidTendon, self).serialize(), **{"method": "MuscleHillModelRigidTendon"}}

    @override
    @staticmethod
    def deserialize(data: dict) -> Self:
        if data["method"] != "MuscleHillModelRigidTendon":
            raise ValueError(f"Cannot deserialize {data['method']} as MuscleHillModelRigidTendon")
        return MuscleHillModelRigidTendon(
            name=data["name"],
            maximal_force=data["maximal_force"],
            optimal_length=data["optimal_length"],
            tendon_slack_length=data["tendon_slack_length"],
            maximal_velocity=data["maximal_velocity"],
            label=data["label"],
            compute_force_passive=ComputeForcePassiveMethods.deserialize(data["compute_force_passive"]),
            compute_force_active=ComputeForceActiveMethods.deserialize(data["compute_force_active"]),
            compute_force_velocity=ComputeForceVelocityMethods.deserialize(data["compute_force_velocity"]),
            compute_force_damping=ComputeForceDampingMethods.deserialize(data["compute_force_damping"]),
            compute_pennation_angle=ComputePennationAngleMethods.deserialize(data["compute_pennation_angle"]),
            compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.deserialize(
                data["compute_muscle_fiber_length"]
            ),
            compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.deserialize(
                data["compute_muscle_fiber_velocity"]
            ),
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
    def compute_tendon_length(self, muscle_tendon_length: MX, muscle_fiber_length: MX) -> MX:
        return self.tendon_slack_length

    @override
    def compute_tendon_force(self, tendon_length: MX) -> MX:
        return 0.0
