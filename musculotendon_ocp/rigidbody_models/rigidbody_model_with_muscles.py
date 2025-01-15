from functools import cached_property
from typing import Iterable, Callable, override, Any, Self

from bioptim import BiorbdModel
import biorbd_casadi as biorbd
from casadi import MX, DM, Function, vertcat

from ..muscle_hill_models import MuscleHillModels
from ..muscle_hill_models.muscle_hill_model_abstract import MuscleHillModelAbstract
from ..muscle_hill_models.compute_muscle_fiber_length import (
    ComputeMuscleFiberLengthMethods,
    ComputeMuscleFiberLengthAsVariable,
    ComputeMuscleFiberLengthRigidTendon,
    ComputeMuscleFiberLengthInstantaneousEquilibrium,
)
from ..muscle_hill_models.compute_muscle_fiber_velocity import (
    ComputeMuscleFiberVelocityMethods,
    ComputeMuscleFiberVelocityAsVariable,
)


class RigidbodyModelWithMuscles(BiorbdModel):
    def __init__(self, bio_model: str, muscles: Iterable[MuscleHillModelAbstract], *args, **kwargs):
        super(RigidbodyModelWithMuscles, self).__init__(bio_model, *args, **kwargs)

        self._muscle_index_to_biorbd_model = []
        biorbd_muscle_names = super(RigidbodyModelWithMuscles, self).muscle_names
        for muscle in muscles:
            if muscle.name not in biorbd_muscle_names:
                raise ValueError(f"Muscle {muscle.name} was not found in the biorbd model")
            self._muscle_index_to_biorbd_model.append(biorbd_muscle_names.index(muscle.name))

        self.muscles = muscles

    @override
    @property
    def copy(self) -> Self:
        new_muscles = []
        for muscle in self.muscles:
            new_muscles.append(muscle.copy)

            # Keep the same mx_variables though (by overwriting the cached_property)
            new_muscles[-1].activation_mx = muscle.activation_mx
            new_muscles[-1].muscle_fiber_length_mx = muscle.muscle_fiber_length_mx
            new_muscles[-1].muscle_fiber_velocity_mx = muscle.muscle_fiber_velocity_mx
            new_muscles[-1].tendon_length_mx = muscle.tendon_length_mx

        new_model = RigidbodyModelWithMuscles(
            bio_model=self.model.path().absolutePath().to_string(),
            friction_coefficients=self._friction_coefficients,
            segments_to_apply_external_forces=self._segments_to_apply_external_forces,
            muscles=new_muscles,
        )

        new_model.q_mx = self.q_mx
        new_model.qdot_mx = self.qdot_mx
        new_model.activations_mx = self.activations_mx
        new_model.muscle_fiber_lengths_mx = self.muscle_fiber_lengths_mx
        new_model.muscle_fiber_velocities_mx = self.muscle_fiber_velocities_mx
        new_model.muscle_fiber_velocity_initial_guesses_mx = self.muscle_fiber_velocity_initial_guesses_mx

        return new_model

    def copy_with_all_flexible_tendons(self) -> Self:
        new_muscles = []
        for muscle in self.muscles:
            new_muscles.append(
                MuscleHillModels.FlexibleTendon(
                    name=muscle.name,
                    maximal_force=muscle.maximal_force,
                    optimal_length=muscle.optimal_length,
                    tendon_slack_length=muscle.tendon_slack_length,
                    maximal_velocity=muscle.maximal_velocity,
                    compute_force_passive=muscle.compute_force_passive,
                    compute_force_active=muscle.compute_force_active,
                    compute_force_velocity=muscle.compute_force_velocity,
                    compute_force_damping=muscle.compute_force_damping,
                    compute_pennation_angle=muscle.compute_pennation_angle,
                    compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.AsVariable(
                        mx_symbolic=muscle.compute_muscle_fiber_length.mx_variable
                    ),
                    compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonFromVelocityDefects(
                        mx_symbolic=muscle.compute_muscle_fiber_velocity.mx_variable
                    ),
                )
            )

            # Keep the same mx_variables though (by overwriting the cached_property)
            new_muscles[-1].activation_mx = muscle.activation_mx
            new_muscles[-1].muscle_fiber_length_mx = muscle.muscle_fiber_length_mx
            new_muscles[-1].muscle_fiber_velocity_mx = muscle.muscle_fiber_velocity_mx
            new_muscles[-1].tendon_length_mx = muscle.tendon_length_mx

        new_model = RigidbodyModelWithMuscles(
            bio_model=self.model.path().absolutePath().to_string(),
            friction_coefficients=self._friction_coefficients,
            segments_to_apply_external_forces=self._segments_to_apply_external_forces,
            muscles=new_muscles,
        )
        new_model.q_mx = self.q_mx
        new_model.qdot_mx = self.qdot_mx
        new_model.activations_mx = self.activations_mx
        new_model.muscle_fiber_lengths_mx = self.muscle_fiber_lengths_mx
        new_model.muscle_fiber_velocities_mx = self.muscle_fiber_velocities_mx
        new_model.muscle_fiber_velocity_initial_guesses_mx = self.muscle_fiber_velocity_initial_guesses_mx

        return new_model

    @cached_property
    def q_mx(self) -> MX:
        """
        Get the symbolic generalized coordinates

        Returns
        -------
        MX
            The symbolic generalized coordinates
        """

        return MX.sym("q", self.nb_q, 1)

    @cached_property
    def qdot_mx(self) -> MX:
        """
        Get the symbolic generalized velocities

        Returns
        -------
        MX
            The symbolic generalized velocities
        """
        return MX.sym("qdot", self.nb_qdot, 1)

    @cached_property
    def activations_mx(self) -> MX:
        """
        Get the symbolic muscle activations

        Returns
        -------
        MX
            The symbolic muscle activations
        """
        return MX.sym("activations", self.nb_muscles, 1)

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
    @override
    def muscle_names(self) -> list[str]:
        """
        Get the muscle names

        Returns
        -------
        list[str]
            The muscle names
        """
        return [muscle.name for muscle in self.muscles]

    @cached_property
    def muscle_fiber_lengths_mx(self) -> MX:
        variables = []
        for muscle in self.muscles:
            if isinstance(muscle.compute_muscle_fiber_length, ComputeMuscleFiberLengthAsVariable):
                variables.append(muscle.compute_muscle_fiber_length.mx_variable)
        return vertcat(*variables)

    @cached_property
    def muscle_fiber_velocities_mx(self) -> MX:
        variables = []
        for muscle in self.muscles:
            if isinstance(muscle.compute_muscle_fiber_velocity, ComputeMuscleFiberVelocityAsVariable):
                variables.append(muscle.compute_muscle_fiber_velocity.mx_variable)
        return vertcat(*variables)

    @cached_property
    def muscle_fiber_velocity_initial_guesses_mx(self) -> MX:
        variables = []
        for muscle in self.muscles:
            if isinstance(muscle.compute_muscle_fiber_velocity, ComputeMuscleFiberVelocityAsVariable):
                variables.append(MX.sym(f"vm_{muscle.name}", 1, 1))
        return vertcat(*variables)

    def muscle_tendon_lengths(self, q) -> MX:
        """
        Compute the muscle-tendon lengths

        Parameters
        ----------
        q: MX
            The generalized coordinates vector of size (n_dof x 1)

        Returns
        -------
        MX
            The muscle-tendon lengths vector of size (n_muscles x 1)
        """

        updated_model = self.model.UpdateKinematicsCustom(q)
        out = []
        for index in range(self.nb_muscles):
            mus: biorbd.Muscle = self.model.muscle(self._muscle_index_to_biorbd_model[index])
            mus.updateOrientations(updated_model, q)
            out.append(mus.musculoTendonLength(updated_model, q).to_mx())
        return vertcat(*out)

    def muscle_tendon_length_jacobian(self, q) -> MX:
        return super(RigidbodyModelWithMuscles, self).muscle_length_jacobian(q)[self._muscle_index_to_biorbd_model, :]

    @override
    def muscle_length_jacobian(self, q) -> MX:
        raise RuntimeError(
            "In the context of this project, the name 'muscle_length_jacobian' is confusing as it is the "
            "jacobian of the muscle-tendon-unit length (as opposed to the muscle-fiber-unit length)."
        )

    def tendon_lengths(self, activations: MX, q: MX, qdot: MX) -> MX:
        muscle_tendon_lengths = self.muscle_tendon_lengths(q)

        out = []
        for index in range(self.nb_muscles):
            muscle: MuscleHillModelAbstract = self.muscles[index]

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

    def tendon_forces(self, activations: MX, q: MX, qdot: MX) -> MX:
        """
        Compute the tendon force for each muscles

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
            The tendon forces vector of size (n_muscles x 1)
        """

        tendon_lengths = self.tendon_lengths(activations, q, qdot)

        forces = MX.zeros(self.nb_muscles, 1)
        for i, muscle in enumerate(self.muscles):
            forces[i] = muscle.compute_tendon_force(tendon_length=tendon_lengths[i])

        return forces

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
                muscle.compute_muscle_fiber_length
                if isinstance(muscle.compute_muscle_fiber_length, ComputeMuscleFiberLengthRigidTendon)
                else ComputeMuscleFiberLengthInstantaneousEquilibrium(
                    mx_symbolic=muscle.compute_muscle_fiber_length.mx_variable
                )
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

    def muscle_fiber_velocities(
        self, activations: MX, q: MX, qdot: MX, muscle_fiber_lengths: MX, muscle_fiber_velocity_initial_guesses: MX
    ) -> MX:
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
        muscle_fiber_lengths: MX
            The muscle fiber lengths vector of size (n_muscles x 1)
        muscle_fiber_velocity_initial_guesses: MX
            The initial guesses for the muscle fiber velocities vector of size (n_muscles x 1)

        Returns
        -------
        MX
            The muscle velocities vector of size (n_muscles x 1)
        """

        updated_model = self.model.UpdateKinematicsCustom(q, qdot)

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
                muscle_fiber_velocity_initial_guess=muscle_fiber_velocity_initial_guesses[index],
            )
        return velocities

    def muscle_forces(
        self, activations: MX, q: MX, qdot: MX, muscle_fiber_lengths: MX, muscle_fiber_velocities: MX
    ) -> MX:
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
        muscle_fiber_lengths: MX
            The muscle fiber lengths vector of size (n_muscles x 1)
        muscle_fiber_velocities: MX
            The muscle fiber velocities vector of size (n_muscles x 1)

        Returns
        -------
        MX
            The muscle forces vector of size (n_muscles x 1)
        """

        forces = MX.zeros(self.nb_muscles, 1)
        for i, muscle in enumerate(self.muscles):
            forces[i] = muscle.compute_muscle_force(
                activation=activations[i],
                muscle_fiber_length=muscle_fiber_lengths[i],
                muscle_fiber_velocity=muscle_fiber_velocities[i],
            )

        return forces

    @override
    def muscle_joint_torque(
        self, activations: MX, q: MX, qdot: MX, muscle_fiber_lengths: MX, muscle_fiber_velocities: MX
    ) -> MX:
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
        muscle_fiber_lengths: MX
            The muscle fiber lengths vector of size (n_muscles x 1)
        muscle_fiber_velocities: MX
            The muscle fiber velocities vector of size (n_muscles x 1)

        Returns
        -------
        MX
            The muscle joint torque vector of size (n_dof x 1)
        """
        muscle_tendon_length_jacobian = self.muscle_tendon_length_jacobian(q)
        muscle_forces = self.muscle_forces(
            activations=activations,
            q=q,
            qdot=qdot,
            muscle_fiber_lengths=muscle_fiber_lengths,
            muscle_fiber_velocities=muscle_fiber_velocities,
        )

        return -muscle_tendon_length_jacobian.T @ muscle_forces

    def to_casadi_function(self, mx_function: Callable[[Any], MX], *keys: Iterable[str]) -> Function:
        """
        Convert a CasADi MX to a CasADi Function with specific parameters. To evaluate the function you can call it
        with the following parameters: ["q", "qdot", "activations", "muscle_fiber_lengths", "muscle_fiber_velocities", "muscle_fiber_velocity_initial_guesses"]
        and get the ["output"] value.

        Parameters
        ----------
        mx_function: Callable
            The CasADi MX function to convert to a casadi function.
        keys: Iterable[str]
            The keys of the variables to use in the function, limited to "q", "qdot", "activations",
            "muscle_fiber_lengths", "muscle_fiber_velocities", "muscle_fiber_velocity_initial_guesses".
            The order of the keys must match the order of the arguments in the mx_function.

        Returns
        -------
        DM
            The result of the evaluation
        """

        keys_to_mx = {}
        for key in keys:
            if key == "q":
                keys_to_mx[key] = self.q_mx
            elif key == "qdot":
                keys_to_mx[key] = self.qdot_mx
            elif key == "activations":
                keys_to_mx[key] = self.activations_mx
            elif key == "muscle_fiber_lengths":
                keys_to_mx[key] = self.muscle_fiber_lengths_mx
            elif key == "muscle_fiber_velocities":
                keys_to_mx[key] = self.muscle_fiber_velocities_mx
            elif key == "muscle_fiber_velocity_initial_guesses":
                keys_to_mx[key] = self.muscle_fiber_velocity_initial_guesses_mx
            else:
                raise ValueError(
                    f"Expected 'q', 'qdot', 'activations', 'muscle_fiber_lengths', 'muscle_fiber_velocities', "
                    f"'muscle_fiber_velocity_initial_guesses' but got {key}"
                )

        return Function(
            "f",
            [
                self.q_mx,
                self.qdot_mx,
                self.activations_mx,
                self.muscle_fiber_lengths_mx,
                self.muscle_fiber_velocities_mx,
                self.muscle_fiber_velocity_initial_guesses_mx,
            ],
            [mx_function(**keys_to_mx)],
            [
                "q",
                "qdot",
                "activations",
                "muscle_fiber_lengths",
                "muscle_fiber_velocities",
                "muscle_fiber_velocity_initial_guesses",
            ],
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
