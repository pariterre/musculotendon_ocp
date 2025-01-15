from typing import override, Self
from casadi import MX, exp

from .compute_pennation_angle import ComputePennationAngleMethods
from .compute_force_active import ComputeForceActiveMethods
from .compute_force_damping import ComputeForceDampingMethods
from .compute_force_passive import ComputeForcePassiveMethods
from .compute_force_velocity import ComputeForceVelocityMethods
from .compute_muscle_fiber_length import ComputeMuscleFiberLengthMethods
from .compute_muscle_fiber_velocity import ComputeMuscleFiberVelocityMethods
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
            compute_muscle_fiber_length = ComputeMuscleFiberLengthMethods.InstantaneousEquilibrium()
        if isinstance(compute_muscle_fiber_length, ComputeMuscleFiberLengthMethods.RigidTendon.value):
            raise ValueError("The compute_muscle_fiber_length must be a flexible tendon")

        if compute_muscle_fiber_velocity is None:
            compute_muscle_fiber_velocity = ComputeMuscleFiberVelocityMethods.FlexibleTendonFromForceDefects()
        if isinstance(compute_muscle_fiber_velocity, ComputeMuscleFiberVelocityMethods.RigidTendon.value):
            raise ValueError("The compute_muscle_fiber_velocity must be a flexible tendon")

        super(MuscleHillModelFlexibleTendon, self).__init__(
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

    @override
    def serialize(self):
        return {
            **super(MuscleHillModelFlexibleTendon, self).serialize(),
            **{"method": "MuscleHillModelFlexibleTendon", "c1": self.c1, "c2": self.c2, "c3": self.c3, "kt": self.kt},
        }

    @override
    @staticmethod
    def deserialize(data: dict) -> Self:
        if data["method"] != "MuscleHillModelFlexibleTendon":
            raise ValueError(f"Cannot deserialize {data['method']} as MuscleHillModelFlexibleTendon")
        return MuscleHillModelFlexibleTendon(
            name=data["name"],
            c1=data["c1"],
            c2=data["c2"],
            c3=data["c3"],
            kt=data["kt"],
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
    @override
    def copy(self) -> Self:
        return MuscleHillModelFlexibleTendonAlwaysPositive(
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

    @override
    def serialize(self) -> dict:
        return {
            **super(MuscleHillModelFlexibleTendonAlwaysPositive, self).serialize(),
            **{"method": "MuscleHillModelFlexibleTendonAlwaysPositive"},
        }

    @override
    @staticmethod
    def deserialize(data: dict) -> Self:
        if data["method"] != "MuscleHillModelFlexibleTendonAlwaysPositive":
            raise ValueError(f"Cannot deserialize {data['method']} as MuscleHillModelFlexibleTendonAlwaysPositive")
        return MuscleHillModelFlexibleTendonAlwaysPositive(
            name=data["name"],
            c1=data["c1"],
            c2=data["c2"],
            c3=data["c3"],
            kt=data["kt"],
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
