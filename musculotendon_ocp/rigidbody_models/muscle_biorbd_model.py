from typing import Iterable, Callable, override, Any

import biorbd_casadi as biorbd
from bioptim import BiorbdModel, OptimalControlProgram, NonLinearProgram, ConfigureProblem
from casadi import MX, DM, Function, vertcat

from ..muscle_models.muscle_model_abstract import MuscleModelAbstract
from ..muscle_models.compute_muscle_fiber_length import (
    ComputeMuscleFiberLengthAsVariable,
    ComputeMuscleFiberLengthInstantaneousEquilibrium,
)
from ..muscle_models.compute_muscle_fiber_velocity import ComputeMuscleFiberVelocityFlexibleTendon


class MuscleBiorbdModel(BiorbdModel):
    def __init__(self, bio_model: str, muscles: Iterable[MuscleModelAbstract], *args, **kwargs):
        super().__init__(bio_model, *args, **kwargs)

        self._muscle_index_to_biorbd_model = []
        for muscle in muscles:
            if muscle.name not in self.muscle_names:
                raise ValueError(f"Muscle {muscle.name} was not found in the biorbd model")
            self._muscle_index_to_biorbd_model.append(self.muscle_names.index(muscle.name))

        self.muscles = muscles

    @property
    def nb_muscles(self) -> int:
        """
        Get the number of muscles

        Returns
        -------
        int
            The number of muscles
        """
        return len(self.muscles)

    @property
    def _muscle_fiber_length_mx_variables(self) -> MX:
        variables = []
        for muscle in self.muscles:
            if isinstance(muscle.compute_muscle_fiber_length, ComputeMuscleFiberLengthAsVariable):
                variables.append(muscle.compute_muscle_fiber_length.mx_variable)
        return vertcat(*variables)

    @property
    def _muscle_fiber_velocity_mx_variables(self) -> MX:
        variables = []
        for muscle in self.muscles:
            if isinstance(muscle.compute_muscle_fiber_velocity, ComputeMuscleFiberVelocityFlexibleTendon):
                variables.append(muscle.compute_muscle_fiber_velocity.mx_variable)
        return vertcat(*variables)

    def tendon_force(self, tendon_length: MX) -> MX:
        """
        Compute the tendon force

        Parameters
        ----------
        tendon_length: MX
            The tendon length

        Returns
        -------
        MX
            The tendon force corresponding to the given tendon length
        """
        raise NotImplementedError("TODO")

    @override
    def muscle_length_jacobian(self, q) -> MX:
        raise RuntimeError(
            "In the context of this project, the name 'muscle_length_jacobian' is confusing as it is the "
            "jacobian of the muscle-tendon-unit length (as opposed to the muscle-fiber-unit length)."
        )

    def muscle_tendon_lengths(self, q) -> MX:
        # TODO ADD TEST THIS METHOD
        updated_model = self.model.UpdateKinematicsCustom(q)
        out = []
        for index in range(self.nb_muscles):
            mus: biorbd.Muscle = self.model.muscle(self._muscle_index_to_biorbd_model[index])
            mus.updateOrientations(updated_model, q)
            out.append(mus.musculoTendonLength(updated_model, q).to_mx())
        return vertcat(*out)

    def muscle_tendon_length_jacobian(self, q) -> MX:
        return super(MuscleBiorbdModel, self).muscle_length_jacobian(q)[self._muscle_index_to_biorbd_model, :]

    def tendon_lengths(self, activations: MX, q: MX, qdot: MX) -> MX:
        # TODO ADD TEST THIS METHOD
        muscle_tendon_lengths = self.muscle_tendon_lengths(q)

        out = []
        for index in range(self.nb_muscles):
            muscle: MuscleModelAbstract = self.muscles[index]

            muscle_fiber_length = muscle.compute_muscle_fiber_length(
                muscle=muscle,
                model_kinematic_updated=self.model.UpdateKinematicsCustom(q),
                biorbd_muscle=self.model.muscle(self._muscle_index_to_biorbd_model[index]),
                activation=activations[index],
                q=q,
                qdot=qdot,
            )

            out.append(
                muscle.compute_tendon_length(
                    muscle_tendon_length=muscle_tendon_lengths[index], muscle_fiber_length=muscle_fiber_length
                )
            )
        return vertcat(*out)

    def muscle_fiber_lengths(self, activations: MX, q: MX, qdot: MX) -> MX:
        """
        Compute the muscle lengths

        Parameters
        ----------
        activations: MX
            The muscle activations vector of size (n_muscles x 1)
        q: MX
            The generalized coordinates vector of size (n_dof x 1)
        qdot: MX
            The generalized velocities vector of size (n_dof x 1)

        Returns
        -------
        MX
            The muscle lengths vector of size (n_muscles x 1)
        """

        updated_model = self.model.UpdateKinematicsCustom(q)

        lengths = MX.zeros(self.nb_muscles, 1)
        for index, muscle in enumerate(self.muscles):
            lengths[index] = muscle.compute_muscle_fiber_length(
                muscle=muscle,
                model_kinematic_updated=updated_model,
                biorbd_muscle=self.model.muscle(self._muscle_index_to_biorbd_model[index]),
                activation=activations[index],
                q=q,
                qdot=qdot,
            )

        return lengths

    def muscle_fiber_lengths_equilibrated(self, activations: MX, q: MX, qdot: MX) -> MX:
        """
        Compute the muscle lengths normally for the RigidTendon but as if the muscle and tendon were instantaneous
        equilibrium for the FlexibleTendon

        Parameters
        ----------
        activations: MX
            The muscle activations vector of size (n_muscles x 1)
        q: MX
            The generalized coordinates vector of size (n_dof x 1)
        qdot: MX
            The generalized velocities vector of size (n_dof x 1)

        Returns
        -------
        MX
            The muscle lengths vector of size (n_muscles x 1)
        """

        updated_model = self.model.UpdateKinematicsCustom(q)

        lengths = MX.zeros(self.nb_muscles, 1)
        for index, muscle in enumerate(self.muscles):
            computer = (
                ComputeMuscleFiberLengthInstantaneousEquilibrium(
                    mx_symbolic=muscle.compute_muscle_fiber_length.mx_variable
                )
                if isinstance(muscle.compute_muscle_fiber_length, ComputeMuscleFiberLengthAsVariable)
                else muscle.compute_muscle_fiber_length
            )

            lengths[index] = computer(
                muscle=muscle,
                model_kinematic_updated=updated_model,
                biorbd_muscle=self.model.muscle(self._muscle_index_to_biorbd_model[index]),
                activation=activations[index],
                q=q,
                qdot=qdot,
            )

        return lengths

    def muscle_fiber_velocities(self, activations: MX, q: MX, qdot: MX) -> MX:
        """
        Compute the muscle velocities

        Parameters
        ----------
        activations: MX
            The muscle activations vector of size (n_muscles x 1)
        q: MX
            The generalized coordinates vector of size (n_dof x 1)
        qdot: MX
            The generalized velocities vector of size (n_dof x 1)

        Returns
        -------
        MX
            The muscle velocities vector of size (n_muscles x 1)
        """

        updated_model = self.model.UpdateKinematicsCustom(q, qdot)
        muscle_fiber_lengths = self.muscle_fiber_lengths(activations, q, qdot)
        tendon_lengths = self.tendon_lengths(activations, q, qdot)

        velocities = MX.zeros(self.nb_muscles, 1)
        for index, muscle in enumerate(self.muscles):
            velocities[index] = muscle.compute_muscle_fiber_velocity(
                muscle=muscle,
                model_kinematic_updated=updated_model,
                biorbd_muscle=self.model.muscle(self._muscle_index_to_biorbd_model[index]),
                activation=activations[index],
                q=q,
                qdot=qdot,
                muscle_fiber_length=muscle_fiber_lengths[index],
                tendon_length=tendon_lengths[index],
            )
        return velocities

    def muscle_forces(self, activations: MX, q: MX, qdot: MX) -> MX:
        """
        Compute the muscle forces

        Parameters
        ----------
        activations: MX
            The muscle activations vector of size (n_muscles x 1)
        q: MX
            The generalized coordinates vector of size (n_dof x 1)
        qdot: MX
            The generalized velocities vector of size (n_dof x 1)

        Returns
        -------
        MX
            The muscle forces vector of size (n_muscles x 1)
        """

        muscle_fiber_lengths = self.muscle_fiber_lengths(activations, q, qdot)
        muscle_fiber_velocities = self.muscle_fiber_velocities(activations, q, qdot)

        forces = MX.zeros(self.nb_muscles, 1)
        for i, muscle in enumerate(self.muscles):
            forces[i] = muscle.compute_muscle_force(
                activation=activations[i],
                muscle_fiber_length=muscle_fiber_lengths[i],
                muscle_fiber_velocity=muscle_fiber_velocities[i],
            )

        return forces

    @override
    def muscle_joint_torque(self, activations: MX, q: MX, qdot: MX) -> MX:
        """
        Compute the muscle joint torque

        Parameters
        ----------
        muscle_activations: MX
            The muscle activations vector of size (n_muscles x 1)
        q: MX
            The generalized coordinates vector of size (n_dof x 1)
        qdot: MX
            The generalized velocities vector of size (n_dof x 1)

        Returns
        -------
        MX
            The muscle joint torque vector of size (n_dof x 1)
        """
        muscle_tendon_length_jacobian = self.muscle_tendon_length_jacobian(q)
        muscle_forces = self.muscle_forces(activations, q, qdot)

        return -muscle_tendon_length_jacobian.T @ muscle_forces

    def configure_bioptim_dynamics(
        self, ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None
    ):
        """
        Configure the bioptim dynamics

        Returns
        -------
        DynamicsFcn
            The bioptim dynamics function
        """
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        ConfigureProblem.configure_qddot(ocp, nlp, as_states=False, as_controls=False, as_states_dot=True)

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
            The keys of the variables to use in the function, limited to "q", "qdot", "activations".
            The order of the keys must match the order of the arguments in the mx_function.

        Returns
        -------
        DM
            The result of the evaluation
        """

        q_mx = MX.sym("q", self.nb_q, 1)
        qdot_mx = MX.sym("qdot", self.nb_qdot, 1)
        activations_mx = MX.sym("activations", self.nb_muscles, 1)

        keys_to_mx = {}
        for key in keys:
            if key == "q":
                keys_to_mx[key] = q_mx
            elif key == "qdot":
                keys_to_mx[key] = qdot_mx
            elif key == "activations":
                keys_to_mx[key] = activations_mx
            else:
                raise ValueError(f"Key {key} is not recognized")

        muscle_lengths_mx = self._muscle_fiber_length_mx_variables
        muscle_velocities_mx = self._muscle_fiber_velocity_mx_variables

        return Function(
            "f",
            [q_mx, qdot_mx, activations_mx, muscle_lengths_mx, muscle_velocities_mx],
            [mx_function(**keys_to_mx)],
            ["q", "qdot", "activations", "muscle_lengths_mx", "muscle_velocities_mx"],
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
            The values of the variables at which to evaluate the function, limited to q, qdot, activations, muscle_lengths,
            muscle_velocities. The default values are 0.

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

        q = get_value_from_kwargs("q", [0] * self.nb_q)
        qdot = get_value_from_kwargs("qdot", [0] * self.nb_qdot)
        activations = get_value_from_kwargs("activations", [0] * self.nb_muscles)
        muscle_lengths_mx = get_value_from_kwargs("muscle_lengths", [0] * self.nb_muscles)
        muscle_velocities_mx = get_value_from_kwargs("muscle_velocities", [0] * self.nb_muscles)

        if kwargs:
            raise ValueError(f"Unexpected arguments: {kwargs.keys()}")

        return function_to_evaluate(
            q=q,
            qdot=qdot,
            activations=activations,
            muscle_lengths_mx=muscle_lengths_mx,
            muscle_velocities_mx=muscle_velocities_mx,
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
