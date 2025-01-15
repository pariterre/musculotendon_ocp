from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Callable, Iterable, Protocol, Self

import biorbd_casadi as biorbd
from casadi import MX, Function, DM


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

    @property
    def copy(self) -> Self:
        """
        Get a copy of the passive force-length relationship
        """

    def serialize(self) -> dict:
        """
        Serialize the passive force-length relationship
        """

    @staticmethod
    def deserialize(data: dict) -> Self:
        """
        Deserialize the passive force-length relationship

        Parameters
        ----------
        data: dict
            The serialized passive force-length relationship

        Returns
        -------
        ComputeForcePassive
            The deserialized passive force-length relationship
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

    @property
    def copy(self) -> Self:
        """
        Get a copy of the active force-length relationship
        """

    def serialize(self) -> dict:
        """
        Serialize the active force-length relationship
        """

    @staticmethod
    def deserialize(data: dict) -> Self:
        """
        Deserialize the active force-length relationship

        Parameters
        ----------
        data: dict
            The serialized active force-length relationship

        Returns
        -------
        ComputeForceActive
            The deserialized active force-length relationship
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

    @property
    def copy(self) -> Self:
        """
        Get a copy of the force-velocity relationship
        """

    def serialize(self) -> dict:
        """
        Serialize the force-velocity relationship
        """

    @staticmethod
    def deserialize(data: dict) -> Self:
        """
        Deserialize the force-velocity relationship

        Parameters
        ----------
        data: dict
            The serialized force-velocity relationship

        Returns
        -------
        ComputeForceVelocity
            The deserialized force-velocity relationship
        """

    def inverse(force_velocity_inverse: MX) -> MX:
        """
        Compute the normalized velocity from the inverse of the force-velocity relationship

        Parameters
        ----------
        force_velocity_inverse: MX
            The inverse of the force-velocity relationship

        Returns
        -------
        MX
            The normalized velocity corresponding to the given inverse of the force-velocity relationship
        """

    def first_derivative(self, normalized_muscle_fiber_velocity: MX) -> MX:
        """
        Computation the first order derivative of the force-velocity relationship at a given normalized_muscle_fiber_velocity.

        Parameters
        ----------
        normalized_muscle_fiber_velocity: MX
            The normalized muscle velocity

        Returns
        -------
        MX
            The first derivative of the force-velocity relationship at the given normalized_muscle_fiber_velocity
        """

    def second_derivative(self, normalized_muscle_fiber_velocity: MX) -> MX:
        """
        Computation the second order derivative of the force-velocity relationship at a given normalized_muscle_fiber_velocity.

        Parameters
        ----------
        normalized_muscle_fiber_velocity: MX
            The normalized muscle velocity

        Returns
        -------
        MX
            The second derivative of the force-velocity relationship at the given normalized_muscle_fiber_velocity
        """


class ComputeForceDamping(Protocol):
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

    @property
    def copy(self) -> Self:
        """
        Get a copy of the damping
        """

    def serialize(self) -> dict:
        """
        Serialize the damping
        """

    @staticmethod
    def deserialize(data: dict) -> Self:
        """
        Deserialize the damping

        Parameters
        ----------
        data: dict
            The serialized damping

        Returns
        -------
        ComputeForceDamping
            The deserialized damping
        """

    @property
    def factor(self):
        """
        Return the factor of the damping
        """


class ComputePennationAngle(Protocol):
    def __call__(self, muscle_fiber_length: MX) -> MX:
        """
        Compute the pennation angle

        Parameters
        ----------
        muscle_fiber_length: MX
            The muscle length

        Returns
        -------
        MX
            The pennation angle corresponding to the given muscle length
        """

    @property
    def copy(self) -> Self:
        """
        Get a copy of the pennation angle
        """

    def serialize(self) -> dict:
        """
        Serialize the pennation angle
        """

    @staticmethod
    def deserialize(data: dict) -> Self:
        """
        Deserialize the pennation angle

        Parameters
        ----------
        data: dict
            The serialized pennation angle

        Returns
        -------
        ComputePennationAngle
            The deserialized pennation angle
        """

    def apply(self, muscle_fiber_length: MX, element: MX) -> MX:
        """
        Apply the pennation angle to an element

        Parameters
        ----------
        muscle_fiber_length: MX
            The muscle length
        element: MX
            The element to apply the pennation angle to

        Returns
        -------
        MX
            The element with the pennation angle applied
        """

    def remove(self, muscle_fiber_length: MX, element: MX) -> MX:
        """
        Remove the pennation angle from an element

        Parameters
        ----------
        muscle_fiber_length: MX
            The muscle length
        element: MX
            The element to remove the pennation angle from

        Returns
        -------
        MX
            The element with the pennation angle removed
        """


class ComputeMuscleFiberLength(Protocol):
    def __call__(
        self,
        muscle: "MuscleHillModelAbstract",
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
    ) -> MX:
        """
        Compute the muscle length

        Parameters
        ----------
        muscle: MuscleModelAbstract
            The muscle model
        model_kinematic_updated: biorbd.Model
            The updated biorbd model up to q
        biorbd_muscle: biorbd.Muscle
            The biorbd muscle associated with the muscle model
        activation: MX
            The muscle activation
        q: MX
            The generalized coordinates
        qdot: MX

        Returns
        -------
        MX
            The muscle length
        """

    @property
    def copy(self) -> Self:
        """
        Get a copy of the muscle fiber length function
        """

    def serialize(self) -> dict:
        """
        Serialize the muscle fiber length function
        """

    @staticmethod
    def deserialize(data: dict) -> Self:
        """
        Deserialize the muscle fiber length function

        Parameters
        ----------
        data: dict
            The serialized muscle fiber length function

        Returns
        -------
        ComputeMuscleFiberLength
            The deserialized muscle fiber length function
        """

    @property
    def mx_variable(self) -> MX:
        """
        Get the muscle fiber length MX
        """


class ComputeMuscleFiberVelocity(Protocol):
    def __call__(
        self,
        muscle: "MuscleHillModelAbstract",
        model_kinematic_updated: biorbd.Model,
        biorbd_muscle: biorbd.Muscle,
        activation: MX,
        q: MX,
        qdot: MX,
        muscle_fiber_length: MX,
        muscle_fiber_velocity_initial_guess: MX,
    ) -> MX:
        """
        Compute the muscle velocity

        Parameters
        ----------
        muscle: MuscleModelAbstract
            The muscle model
        model_kinematic_updated: biorbd.Model
            The updated biorbd model up to q
        biorbd_muscle: biorbd.Muscle
            The biorbd muscle associated with the muscle model
        activation: MX
            The muscle activation
        q: MX
            The generalized coordinates
        qdot: MX
            The generalized velocities
        muscle_fiber_length: MX
            The muscle fiber length
        muscle_fiber_velocity_initial_guess: MX
            The initial guess of the muscle fiber velocity

        Returns
        -------
        MX
            The muscle length
        """

    @property
    def copy(self) -> Self:
        """
        Get a copy of the muscle fiber velocity function
        """

    def serialize(self) -> dict:
        """
        Serialize the muscle fiber velocity function
        """

    @staticmethod
    def deserialize(data: dict) -> Self:
        """
        Deserialize the muscle fiber velocity function

        Parameters
        ----------
        data: dict
            The serialized muscle fiber velocity function

        Returns
        -------
        ComputeMuscleFiberVelocity
            The deserialized muscle fiber velocity function
        """

    @property
    def mx_variable(self) -> MX:
        """
        Get the muscle fiber velocity MX
        """


class MuscleHillModelAbstract(ABC):
    def __init__(
        self,
        name: str,
        label: str | None,
        maximal_force: MX,
        optimal_length: MX,
        tendon_slack_length: MX,
        maximal_velocity: MX,
        compute_force_passive: ComputeForcePassive,
        compute_force_active: ComputeForceActive,
        compute_force_velocity: ComputeForceVelocity,
        compute_force_damping: ComputeForceDamping,
        compute_pennation_angle: ComputePennationAngle,
        compute_muscle_fiber_length: ComputeMuscleFiberLength,
        compute_muscle_fiber_velocity: ComputeMuscleFiberVelocity,
    ):
        """
        Parameters
        ----------
        name: str
            The muscle name
        label: str | None
            The muscle label to identify. If None, the name is used.
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
        self._name = name
        self._label = label if label is not None else name

        if maximal_force < 0:
            raise ValueError("The maximal force must be positive")
        self.maximal_force = maximal_force

        if optimal_length < 0:
            raise ValueError("The optimal length must be positive")
        self.optimal_length = optimal_length

        if tendon_slack_length < 0:
            raise ValueError("The tendon slack length must be positive")
        self.tendon_slack_length = tendon_slack_length

        if maximal_velocity < 0:
            raise ValueError("The maximal velocity must be positive")
        self.maximal_velocity = maximal_velocity

        self.compute_force_passive = compute_force_passive
        self.compute_force_active = compute_force_active
        self.compute_force_velocity = compute_force_velocity
        self.compute_force_damping = compute_force_damping

        self.compute_pennation_angle = compute_pennation_angle
        self.compute_muscle_fiber_length = compute_muscle_fiber_length
        self.compute_muscle_fiber_velocity = compute_muscle_fiber_velocity

    @property
    @abstractmethod
    def copy(self) -> Self:
        """
        Get a copy of the muscle model
        """

    def serialize(self) -> dict:
        """
        Serialize the muscle model
        """
        return {
            "name": self.name,
            "label": self.label,
            "maximal_force": self.maximal_force,
            "optimal_length": self.optimal_length,
            "tendon_slack_length": self.tendon_slack_length,
            "maximal_velocity": self.maximal_velocity,
            "compute_force_passive": self.compute_force_passive.serialize(),
            "compute_force_active": self.compute_force_active.serialize(),
            "compute_force_velocity": self.compute_force_velocity.serialize(),
            "compute_force_damping": self.compute_force_damping.serialize(),
            "compute_pennation_angle": self.compute_pennation_angle.serialize(),
            "compute_muscle_fiber_length": self.compute_muscle_fiber_length.serialize(),
            "compute_muscle_fiber_velocity": self.compute_muscle_fiber_velocity.serialize(),
        }

    @staticmethod
    @abstractmethod
    def deserialize(data: dict) -> Self:
        """
        Deserialize the muscle model

        Parameters
        ----------
        data: dict
            The serialized muscle model

        Returns
        -------
        MuscleHillModelAbstract
            The deserialized muscle model
        """

    @property
    def name(self) -> str:
        """
        Get the muscle name
        """
        return self._name

    @property
    def label(self) -> str:
        """
        Get the muscle label
        """
        return self._label

    @cached_property
    def activation_mx(self) -> MX:
        """
        Get the muscle activation MX
        """
        return MX.sym("activation", 1, 1)

    @cached_property
    def muscle_fiber_length_mx(self) -> MX:
        """
        Get the muscle fiber length MX
        """
        return MX.sym("muscle_fiber_length", 1, 1)

    @cached_property
    def muscle_fiber_velocity_mx(self) -> MX:
        """
        Get the muscle fiber velocity MX
        """
        return MX.sym("muscle_fiber_velocity", 1, 1)

    @cached_property
    def tendon_length_mx(self) -> MX:
        """
        Get the tendon length MX
        """
        return MX.sym("tendon_length", 1, 1)

    @abstractmethod
    def normalize_muscle_fiber_length(self, muscle_fiber_length: MX) -> MX:
        """
        Compute the normalized muscle length

        Parameters
        ----------
        muscle_fiber_length: MX
            The muscle length

        Returns
        -------
        MX
            The normalized muscle length
        """

    @abstractmethod
    def denormalize_muscle_fiber_length(self, normalized_muscle_fiber_length: MX) -> MX:
        """
        Compute the denormalized muscle length

        Parameters
        ----------
        normalized_muscle_fiber_length: MX
            The normalized muscle length

        Returns
        -------
        MX
            The denormalized muscle length
        """

    @abstractmethod
    def normalize_muscle_fiber_velocity(self, muscle_fiber_velocity: MX) -> MX:
        """
        Compute the normalized muscle velocity

        Parameters
        ----------
        muscle_fiber_velocity: MX
            The muscle velocity

        Returns
        -------
        MX
            The normalized muscle velocity
        """

    @abstractmethod
    def denormalize_muscle_fiber_velocity(self, normalized_muscle_fiber_velocity: MX) -> MX:
        """
        Compute the denormalized muscle velocity

        Parameters
        ----------
        normalized_muscle_fiber_velocity: MX
            The normalized muscle velocity

        Returns
        -------
        MX
            The denormalized muscle velocity
        """

    @abstractmethod
    def normalize_tendon_length(self, tendon_length: MX) -> MX:
        """
        Compute the normalized tendon length

        Parameters
        ----------
        tendon_length: MX
            The tendon length

        Returns
        -------
        MX
            The normalized tendon length
        """

    @abstractmethod
    def denormalize_tendon_length(self, normalized_tendon_length: MX) -> MX:
        """
        Compute the denormalized tendon length

        Parameters
        ----------
        normalized_tendon_length: MX
            The normalized tendon length

        Returns
        -------
        MX
            The denormalized tendon length
        """

    @abstractmethod
    def compute_muscle_force(self, activation: MX, muscle_fiber_length: MX, muscle_fiber_velocity: MX) -> MX:
        """
        Compute the muscle force

        Parameters
        ----------
        activation: MX
            The muscle activation
        muscle_fiber_length: MX
            The length of the muscle fibers
        muscle_fiber_velocity: MX
            The velocity of the muscle fibers

        Returns
        -------
        MX
            The muscle force corresponding to the given muscle activation, length and velocity
        """

    @abstractmethod
    def compute_tendon_length(self, muscle_tendon_length: MX, muscle_fiber_length: MX) -> MX:
        """
        Compute the tendon length

        Parameters
        ----------
        muscle_tendon_length: MX
            The length of the muscle-tendon unit
        muscle_fiber_length: MX
            The length of the muscle fibers

        Returns
        -------
        MX
            The tendon length corresponding to the given muscle-tendon length and muscle fiber length
        """

    @abstractmethod
    def compute_tendon_force(self, tendon_length: MX) -> MX:
        """
        Compute the tendon force

        Parameters
        ----------
        tendon_length: MX
            The length of tendon unit

        Returns
        -------
        MX
            The tendon force corresponding to the given tendon length
        """

    def to_casadi_function(self, mx_function: Callable[[Any], MX], *keys: Iterable[str]) -> Function:
        """
        Convert a CasADi MX to a CasADi Function with specific parameters. To evaluate the function you can call it
        and get the ["output"] value.

        Parameters
        ----------
        mx_function: Callable
            The CasADi MX function to convert to a casadi function.
        keys: Iterable[str]
            The keys of the parameters of the function. The keys must be in the order of the parameters of the function.
            The keys must be 'activation', 'muscle_fiber_length', 'muscle_fiber_velocity' or 'tendon_length'.

        Returns
        -------
        DM
            The result of the evaluation
        """

        keys_to_mx = {}
        for key in keys:
            if key == "activation":
                keys_to_mx["activation"] = self.activation_mx
            elif key == "muscle_fiber_length":
                keys_to_mx["muscle_fiber_length"] = self.muscle_fiber_length_mx
            elif key == "muscle_fiber_velocity":
                keys_to_mx["muscle_fiber_velocity"] = self.muscle_fiber_velocity_mx
            elif key == "tendon_length":
                keys_to_mx["tendon_length"] = self.tendon_length_mx
            else:
                raise ValueError(
                    f"Expected 'activation', 'muscle_fiber_length', 'muscle_fiber_velocity' or 'tendon_length', got {key}"
                )

        return Function(
            "f",
            [self.activation_mx, self.muscle_fiber_length_mx, self.muscle_fiber_velocity_mx, self.tendon_length_mx],
            [mx_function(**keys_to_mx)],
            ["activation", "muscle_fiber_length", "muscle_fiber_velocity", "tendon_length"],
            ["output"],
        )

    def function_to_dm(self, mx_to_evaluate: Callable, **kwargs) -> DM:
        """
        Evaluate a function with specific parameters.

        Parameters
        ----------
        mx_to_evaluate: Callable
            The CasADi MX to evaluate.
        **kwargs
            The values of the variables at which to evaluate the function, limited to q, qdot, lm, activations, vm. The
            default values are 0.

        Returns
        -------
        DM
            The result of the evaluation
        """
        func = self.to_casadi_function(mx_to_evaluate, *kwargs.keys())
        return func(**kwargs)["output"]
