from casadi import MX

from ..rigidbody_models import RigidbodyModelWithMuscles


class CasadiHelpers:
    @staticmethod
    def prepare_forward_dynamics_mx(model: RigidbodyModelWithMuscles, activations: MX, q: MX, qdot: MX) -> MX:
        tau = model.muscle_joint_torque(
            activations=activations,
            q=q,
            qdot=qdot,
            muscle_fiber_lengths=model.muscle_fiber_lengths_mx,
            muscle_fiber_velocities=model.muscle_fiber_velocities_mx,
        )
        qddot = model.forward_dynamics(q, qdot, tau)
        return qddot

    @staticmethod
    def prepare_muscle_forces_mx(
        model: RigidbodyModelWithMuscles,
        activations: MX,
        q: MX,
        qdot: MX,
        muscle_fiber_lengths: MX,
        muscle_fiber_velocities: MX,
    ) -> MX:
        muscle_forces = model.muscle_forces(
            activations=activations,
            q=q,
            qdot=qdot,
            muscle_fiber_lengths=muscle_fiber_lengths,
            muscle_fiber_velocities=muscle_fiber_velocities,
        )
        return muscle_forces

    @staticmethod
    def prepare_fiber_lmdot_mx(model: RigidbodyModelWithMuscles, activations: MX, q: MX, qdot: MX) -> MX:
        fiber_lmdot = model.muscle_fiber_velocities(
            activations=activations,
            q=q,
            qdot=qdot,
            muscle_fiber_lengths=model.muscle_fiber_lengths_mx,
            muscle_fiber_velocity_initial_guesses=model.muscle_fiber_velocity_initial_guesses_mx,
        )

        return fiber_lmdot

    @staticmethod
    def prepare_tendon_forces_mx(model: RigidbodyModelWithMuscles, activations: MX, q: MX, qdot: MX) -> MX:
        tendon_forces = model.tendon_forces(activations=activations, q=q, qdot=qdot)
        return tendon_forces
