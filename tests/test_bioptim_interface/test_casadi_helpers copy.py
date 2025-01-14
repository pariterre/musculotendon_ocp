import pathlib

from casadi import Function
from musculotendon_ocp import (
    MuscleHillModels,
    RigidbodyModels,
    CasadiHelpers,
)
import numpy as np


model_path = (
    pathlib.Path(__file__).parent.resolve()
    / "../../musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod"
).as_posix()


def test_prepare_forward_dynamics_mx():
    rigid = MuscleHillModels.RigidTendon
    flexible = MuscleHillModels.FlexibleTendonAlwaysPositive

    model = RigidbodyModels.WithMuscles(
        model_path,
        muscles=[
            rigid("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.123, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.130, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.123, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
        ],
    )

    q_mx = model.q_mx
    qdot_mx = model.qdot_mx
    activations_mx = model.activations_mx
    muscle_fiber_lengths_mx = model.muscle_fiber_lengths_mx
    muscle_fiber_velocities_mx = model.muscle_fiber_velocities_mx

    fd = CasadiHelpers.prepare_forward_dynamics_mx(model=model, activations=activations_mx, q=q_mx, qdot=qdot_mx)
    func = Function("fd", [activations_mx, q_mx, qdot_mx, muscle_fiber_lengths_mx, muscle_fiber_velocities_mx], [fd])

    np.testing.assert_array_equal(float(func(0.5, -0.24, 5, 0.16, 0.5)), 325.82845444983036)


def test_prepare_muscle_forces_mx():
    rigid = MuscleHillModels.RigidTendon
    flexible = MuscleHillModels.FlexibleTendonAlwaysPositive

    model = RigidbodyModels.WithMuscles(
        model_path,
        muscles=[
            rigid("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.123, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.130, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.123, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
        ],
    )

    q_mx = model.q_mx
    qdot_mx = model.qdot_mx
    activations_mx = model.activations_mx
    muscle_fiber_lengths_mx = model.muscle_fiber_lengths_mx
    muscle_fiber_velocities_mx = model.muscle_fiber_velocities_mx

    muscle_forces = CasadiHelpers.prepare_muscle_forces_mx(
        model=model,
        activations=activations_mx,
        q=q_mx,
        qdot=qdot_mx,
        muscle_fiber_lengths=muscle_fiber_lengths_mx,
        muscle_fiber_velocities=muscle_fiber_velocities_mx,
    )
    func = Function(
        "muscle_forces",
        [activations_mx, q_mx, qdot_mx, muscle_fiber_lengths_mx, muscle_fiber_velocities_mx],
        [muscle_forces],
    )

    np.testing.assert_array_equal(
        func(0.5, -0.24, 5, 0.16, 0.5).T,
        [
            [
                537.4933677388334,
                537.4933677388334,
                301.6027683857425,
                301.6027683857425,
                537.4933677388334,
                537.4933677388334,
                301.6027683857425,
                301.6027683857425,
            ]
        ],
    )


def test_prepare_fiber_lmdot_mx():
    rigid = MuscleHillModels.RigidTendon
    flexible = MuscleHillModels.FlexibleTendonAlwaysPositive

    model = RigidbodyModels.WithMuscles(
        model_path,
        muscles=[
            rigid("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.123, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.130, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.123, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
        ],
    )

    q_mx = model.q_mx
    qdot_mx = model.qdot_mx
    activations_mx = model.activations_mx
    muscle_fiber_lengths_mx = model.muscle_fiber_lengths_mx
    muscle_fiber_velocities_mx = model.muscle_fiber_velocities_mx
    muscle_fiber_velocity_initial_guesses_mx = model.muscle_fiber_velocity_initial_guesses_mx

    fiber_lmdot = CasadiHelpers.prepare_fiber_lmdot_mx(model=model, activations=activations_mx, q=q_mx, qdot=qdot_mx)
    func = Function(
        "fd",
        [
            activations_mx,
            q_mx,
            qdot_mx,
            muscle_fiber_lengths_mx,
            muscle_fiber_velocities_mx,
            muscle_fiber_velocity_initial_guesses_mx,
        ],
        [fiber_lmdot],
    )

    np.testing.assert_array_equal(
        func(0.5, -0.24, 5, 0.12, 0.5, 0.5).T,
        [[-5.0, -5.0, -5.0, -5.0, -7.7552291661850985, -7.765452251999365, -4.710749056077481, -4.710749056077481]],
    )


def test_prepare_tendon_forces_mx():
    rigid = MuscleHillModels.RigidTendon
    flexible = MuscleHillModels.FlexibleTendonAlwaysPositive

    model = RigidbodyModels.WithMuscles(
        model_path,
        muscles=[
            rigid("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.123, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.130, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.123, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
        ],
    )

    q_mx = model.q_mx
    qdot_mx = model.qdot_mx
    activations_mx = model.activations_mx
    muscle_fiber_lengths_mx = model.muscle_fiber_lengths_mx
    muscle_fiber_velocities_mx = model.muscle_fiber_velocities_mx

    tendon_forces = CasadiHelpers.prepare_tendon_forces_mx(
        model=model, activations=activations_mx, q=q_mx, qdot=qdot_mx
    )
    func = Function(
        "fd", [activations_mx, q_mx, qdot_mx, muscle_fiber_lengths_mx, muscle_fiber_velocities_mx], [tendon_forces]
    )

    np.testing.assert_array_equal(
        func(0.5, -0.24, 5, 0.16, 0.5).T,
        [[0.0, 0.0, 0.0, 0.0, 237.25518371597045, 215.53308744898047, 62.123045948982416, 62.123045948982416]],
    )
