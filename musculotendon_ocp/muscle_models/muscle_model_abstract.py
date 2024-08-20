from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable

import biorbd_casadi as biorbd
from casadi import MX, Function, DM


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
type ComputeMuscleFiberLengthCallable = Callable[[MuscleModelAbstract, biorbd.Model, biorbd.Muscle, MX, MX, MX], MX]


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
tendon_length: MX
    The tendon length

Returns
-------
MX
    The muscle length
"""
type ComputeMuscleFiberVelocityCallable = Callable[
    [MuscleModelAbstract, biorbd.Model, biorbd.Muscle, MX, MX, MX, MX, MX], MX
]


class MuscleModelAbstract(ABC):
    def __init__(
        self,
        name: str,
        maximal_force: MX,
        optimal_length: MX,
        tendon_slack_length: MX,
        maximal_velocity: MX,
        compute_muscle_fiber_length: ComputeMuscleFiberLengthCallable,
        compute_muscle_fiber_velocity: ComputeMuscleFiberVelocityCallable,
    ):
        self._name = name

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

        self.compute_muscle_fiber_length = compute_muscle_fiber_length
        self.compute_muscle_fiber_velocity = compute_muscle_fiber_velocity

    @property
    def name(self) -> str:
        """
        Get the muscle name
        """
        return self._name

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
        return muscle_tendon_length - muscle_fiber_length

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
        Convert a CasADi MX to a CasADi Function with specific parameters. To evaluate the function, use evaluate_function.

        Parameters
        ----------
        model: MuscleBiorbdModel
            The model that implements MuscleBiorbdModel.
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

        activation_mx = MX.sym("activation", 1, 1)
        muscle_fiber_length_mx = MX.sym("muscle_fiber_length", 1, 1)
        muscle_fiber_velocity_mx = MX.sym("muscle_fiber_velocity", 1, 1)
        tendon_length_mx = MX.sym("tendon_length", 1, 1)

        keys_to_mx = {}
        for key in keys:
            if key == "activation":
                keys_to_mx["activation"] = activation_mx
            elif key == "muscle_fiber_length":
                keys_to_mx["muscle_fiber_length"] = muscle_fiber_length_mx
            elif key == "muscle_fiber_velocity":
                keys_to_mx["muscle_fiber_velocity"] = muscle_fiber_velocity_mx
            elif key == "tendon_length":
                keys_to_mx["tendon_length"] = tendon_length_mx
            else:
                raise ValueError(
                    f"Expected 'activation', 'muscle_fiber_length', 'muscle_fiber_velocity' or 'tendon_length', got {key}"
                )

        return Function(
            "f",
            [activation_mx, muscle_fiber_length_mx, muscle_fiber_velocity_mx, tendon_length_mx],
            [mx_function(**keys_to_mx)],
            ["activation", "muscle_fiber_length", "muscle_fiber_velocity", "tensor_length"],
            ["output"],
        )

    def evaluate_function(self, function_to_evaluate: Function, **kwargs) -> DM:
        """
        Evaluate a CasADi Function with specific parameters. The function must be created with to_function.

        Parameters
        ----------
        function_to_evaluate: Function
            The CasADi Function to evaluate.
        **kwargs
            The values of the variables at which to evaluate the function, limited to 'activation', 'muscle_fiber_length',
            'muscle_fiber_velocity' or 'tendon_length'.
            The default values are 0.

        Returns
        -------
        DM
            The result of the evaluation
        """

        def get_value_from_kwargs(key, default):
            if key in kwargs:
                element = kwargs[key]
                del kwargs[key]
                return element
            return default

        activation = get_value_from_kwargs("activation", 0)
        muscle_fiber_length = get_value_from_kwargs("muscle_fiber_length", 0)
        muscle_fiber_velocity = get_value_from_kwargs("muscle_fiber_velocity", 0)
        tendon_length = get_value_from_kwargs("tendon_length", 0)

        if kwargs:
            raise ValueError(f"Unknown arguments: {kwargs.keys()}")

        return function_to_evaluate(
            activation=activation,
            muscle_fiber_length=muscle_fiber_length,
            muscle_fiber_velocity=muscle_fiber_velocity,
            tensor_length=tendon_length,
        )["output"]

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
        return self.evaluate_function(func, **kwargs)
