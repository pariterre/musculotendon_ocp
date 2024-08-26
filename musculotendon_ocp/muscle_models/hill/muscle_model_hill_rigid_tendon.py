from typing import override, Protocol

from casadi import MX

from ..compute_pennation_angle import ComputePennationAngleConstant
from .compute_force_active import ComputeForceActiveHillType
from .compute_force_damping import ComputeForceDampingConstant
from .compute_force_passive import ComputeForcePassiveHillType
from .compute_force_velocity import ComputeForceVelocityHillType
from ..muscle_model_abstract import (
    MuscleModelAbstract,
    ComputeMuscleFiberLength,
    ComputeMuscleFiberVelocity,
    ComputePennationAngle,
)
from ..compute_muscle_fiber_length import ComputeMuscleFiberLengthRigidTendon
from ..compute_muscle_fiber_velocity import ComputeMuscleFiberVelocityRigidTendon


class ComputeForcePassive(Protocol):
    def __call__(self, normalized_muscle_length: MX) -> MX:
        """
        Compute the normalized force from the passive force-length relationship

        Parameters
        ----------
        normalized_muscle_length: MX
            The normalized muscle length that impacts the pennation angle

        Returns
        -------
        MX
            The normalized passive force corresponding to the given muscle length
        """


class ComputeForceActive(Protocol):
    def __call__(self, normalized_muscle_length: MX) -> MX:
        """
        Compute the normalized force from the active force-length relationship

        Parameters
        ----------
        normalized_muscle_length: MX
            The normalized muscle length

        Returns
        -------
        MX
            The normalized active force corresponding to the given muscle length
        """


class ComputeForceVelocity(Protocol):
    def __call__(self, normalized_muscle_fiber_length: MX, normalized_muscle_fiber_velocity: MX) -> MX:
        """
        Compute the normalized force from the force-velocity relationship

        Parameters
        ----------
        normalized_muscle_fiber_length: MX
            The normalized muscle length
        normalized_muscle_fiber_velocity: MX
            The normalized muscle velocity

        Returns
        -------
        MX
            The normalized force corresponding to the given muscle length and velocity
        """


class ComputeForceDamping:
    def __call__(self, normalized_muscle_fiber_velocity: MX) -> MX:
        """
        Compute the normalized force from the damping

        Parameters
        ----------
        normalized_muscle_fiber_velocity: MX
            The normalized muscle velocity

        Returns
        -------
        MX
            The normalized force corresponding to the given muscle velocity
        """


class MuscleModelHillRigidTendon(MuscleModelAbstract):
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
        """
        Parameters
        ----------
        name: str
            The muscle name
        maximal_force: MX
            The maximal force the muscle can produce
        optimal_length: MX
            The optimal length of the muscle
        tendon_slack_length: MX
            The tendon slack length
        maximal_velocity: MX
            The maximal velocity of the muscle
        pennation_angle: ComputePennationAngle
            The pennation angle function
        compute_force_passive: ComputeForcePassive
            The passive force-length relationship function
        compute_force_active: ComputeForceActive
            The active force-length relationship function
        compute_force_velocity: ComputeForceVelocity
            The force-velocity relationship function
        compute_force_damping: ComputeForceDamping
            The damping function
        """
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
            compute_pennation_angle=compute_pennation_angle,
            compute_muscle_fiber_length=compute_muscle_fiber_length,
            compute_muscle_fiber_velocity=compute_muscle_fiber_velocity,
        )

        self.compute_force_passive = compute_force_passive
        self.compute_force_active = compute_force_active
        self.compute_force_velocity = compute_force_velocity
        self.compute_force_damping = compute_force_damping

    @override
    def normalize_muscle_fiber_length(self, muscle_fiber_length: MX) -> MX:
        return muscle_fiber_length / self.optimal_length

    @override
    def normalize_muscle_fiber_velocity(self, muscle_fiber_velocity: MX) -> MX:
        return muscle_fiber_velocity / self.maximal_velocity

    @override
    def normalize_tendon_length(self, tendon_length: MX) -> MX:
        raise RuntimeError("The tendon length should not be normalized with a rigid tendon")

    @override
    def compute_muscle_force(self, activation: MX, muscle_fiber_length: MX, muscle_fiber_velocity: MX) -> MX:
        # Get the normalized muscle length and velocity
        normalized_length = self.normalize_muscle_fiber_length(muscle_fiber_length)
        normalized_velocity = self.normalize_muscle_fiber_velocity(muscle_fiber_velocity)

        # Compute the passive, active, velocity and damping factors
        pennation_angle = self.compute_pennation_angle(muscle_fiber_length)
        force_passive = self.compute_force_passive(normalized_length)
        force_active = self.compute_force_active(normalized_length)
        force_velocity = self.compute_force_velocity(normalized_velocity)
        force_damping = self.compute_force_damping(normalized_velocity)

        return (
            pennation_angle
            * self.maximal_force
            * (force_passive + activation * force_active * force_velocity + force_damping)
        )

    @override
    def compute_muscle_force_velocity_inverse(self, activation: MX, muscle_fiber_length: MX, tendon_length: MX) -> MX:
        raise NotImplementedError("The force-velocity relationship is not invertible with a rigid tendon")

    @override
    def compute_tendon_length(self, muscle_tendon_length: MX, muscle_fiber_length: MX) -> MX:
        return self.tendon_slack_length

    @override
    def compute_tendon_force(self, tendon_length: MX) -> MX:
        return 0.0
