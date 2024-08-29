import os
import re

from casadi import MX, Function
from musculotendon_ocp import (
    RigidbodyModels,
    MuscleHillModels,
    ComputeMuscleFiberLengthMethods,
    ComputeMuscleFiberVelocityMethods,
)
import numpy as np
import pytest

model_path = (
    (os.getcwd() + "/musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod")
    .replace("\\", "/")
    .replace("c:/", "C:/")
)


def test_muscle_biorbd_model_wrong_constructor():
    with pytest.raises(ValueError, match="Muscle Wrong muscle was not found in the biorbd model"):
        RigidbodyModels.WithMuscles(
            model_path,
            muscles=[
                MuscleHillModels.RigidTendon(
                    name="Wrong muscle",
                    maximal_force=500,
                    optimal_length=0.1,
                    tendon_slack_length=0.123,
                    maximal_velocity=5.0,
                )
            ],
        )


def test_muscle_biorbd_model_number_of_muscles():
    model = RigidbodyModels.WithMuscles(
        model_path,
        muscles=[
            MuscleHillModels.RigidTendon(
                name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
            ),
            MuscleHillModels.RigidTendon(
                name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
            ),
            MuscleHillModels.RigidTendon(
                name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
            ),
        ],
    )
    assert model.nb_muscles == 3
    assert model.muscle_names == ["Mus1", "Mus1", "Mus1"]

    # Can load a model with less muscles
    model_none = RigidbodyModels.WithMuscles(model_path, muscles=[])
    assert model_none.nb_muscles == 0


def test_get_mx_variables():
    model = RigidbodyModels.WithMuscles(
        model_path,
        muscles=[
            MuscleHillModels.RigidTendon(
                name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
            )
        ],
    )

    # Test the mx variables are returned and they are always the same (cached)
    assert isinstance(model.q_mx, MX)
    assert id(model.q_mx) == id(model.q_mx)

    assert isinstance(model.qdot_mx, MX)
    assert id(model.qdot_mx) == id(model.qdot_mx)

    assert isinstance(model.activations_mx, MX)
    assert id(model.activations_mx) == id(model.activations_mx)


def test_muscle_biorbd_model_get_mx_variables():
    dummy_params = {"maximal_force": 500, "optimal_length": 0.1, "tendon_slack_length": 0.123, "maximal_velocity": 5.0}
    muscle_fiber_length_mx = MX.sym("dummy", 1, 1)
    muscle_fiber_velocity_mx = MX.sym("dummy", 1, 1)
    model = RigidbodyModels.WithMuscles(
        model_path,
        muscles=[
            MuscleHillModels.RigidTendon(name="Mus1", **dummy_params),
            MuscleHillModels.RigidTendon(name="Mus1", **dummy_params),
            MuscleHillModels.FlexibleTendonAlwaysPositive(
                name="Mus1",
                compute_muscle_fiber_length=ComputeMuscleFiberLengthMethods.AsVariable(
                    mx_symbolic=muscle_fiber_length_mx
                ),
                compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonImplicit(
                    mx_symbolic=muscle_fiber_velocity_mx
                ),
                **dummy_params
            ),
            MuscleHillModels.FlexibleTendonAlwaysPositive(name="Mus1", **dummy_params),
        ],
    )

    muscle_fiber_length_mx_variables = model.muscle_fiber_lengths_mx
    assert isinstance(muscle_fiber_length_mx_variables, MX)
    assert muscle_fiber_length_mx_variables.shape == (4, 1)
    np.testing.assert_almost_equal(
        Function("f", [muscle_fiber_length_mx], [muscle_fiber_length_mx_variables[2]])(1.234), 1.234
    )
    with pytest.raises(RuntimeError):
        # Make sure the muscle_fiber_length_mx_variable[0] is actually a MX.sym variable
        Function("f", [MX.sym("dummy", 1, 1)], [muscle_fiber_length_mx_variables[2]])
    with pytest.raises(RuntimeError):
        # Make sure the muscle_fiber_length_mx_variable[1] is not the same as muscle_fiber_length_mx_variable[0]
        Function("f", [muscle_fiber_length_mx], [muscle_fiber_length_mx_variables[3]])

    muscle_fiber_velocity_mx_variables = model.muscle_fiber_velocities_mx
    assert isinstance(muscle_fiber_velocity_mx_variables, MX)
    assert muscle_fiber_velocity_mx_variables.shape == (4, 1)
    np.testing.assert_almost_equal(
        Function("f", [muscle_fiber_velocity_mx], [muscle_fiber_velocity_mx_variables[2]])(1.234), 1.234
    )
    with pytest.raises(RuntimeError):
        # Make sure the muscle_fiber_velocity_mx_variables[0] is actually a MX.sym variable
        Function("f", [MX.sym("dummy", 1, 1)], [muscle_fiber_velocity_mx_variables[2]])
    with pytest.raises(RuntimeError):
        # Make sure the muscle_fiber_velocity_mx_variables[1] is not the same as muscle_fiber_velocity_mx_variables[0]
        Function("f", [muscle_fiber_velocity_mx], [muscle_fiber_velocity_mx_variables[3]])

    muscle_fiber_velocity_mx_variables = model.muscle_fiber_velocity_initial_guesses_mx
    assert isinstance(muscle_fiber_velocity_mx_variables, MX)
    assert muscle_fiber_velocity_mx_variables.shape == (4, 1)
    np.testing.assert_almost_equal(
        Function("f", [muscle_fiber_velocity_mx_variables], [muscle_fiber_velocity_mx_variables])(1.234).T,
        np.array([[1.234] * model.nb_muscles]),
    )
    # Make sure the muscle_fiber_velocity_mx_variables[0] is actually a MX.sym variable
    with pytest.raises(RuntimeError):
        Function("f", [MX.sym("dummy", 1, 1)], [muscle_fiber_velocity_mx_variables[2]])
    # Make sure the muscle_fiber_velocity_mx_variables[1] is not the same as muscle_fiber_velocity_mx_variables[0]
    with pytest.raises(RuntimeError):
        Function("f", [muscle_fiber_velocity_mx], [muscle_fiber_velocity_mx_variables[3]])
    # Make sure calling the method twice returns the same variable
    assert id(muscle_fiber_velocity_mx_variables) == id(model.muscle_fiber_velocity_initial_guesses_mx)


def test_muscle_tendon_lengths():
    rigid = MuscleHillModels.RigidTendon
    flexible = MuscleHillModels.FlexibleTendonAlwaysPositive

    model = RigidbodyModels.WithMuscles(
        model_path,
        muscles=[
            rigid("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.123, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.130, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.123, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
        ],
    )

    muscle_tendon_lengths = model.function_to_dm(model.muscle_tendon_lengths, q=np.array([-0.3]))
    assert muscle_tendon_lengths.shape == (6, 1)
    np.testing.assert_almost_equal(muscle_tendon_lengths.T, [[0.3, 0.3, 0.3, 0.3, 0.3, 0.3]])


def test_muscle_biorbd_model_muscle_tendon_length_jacobian():
    rigid = MuscleHillModels.RigidTendon
    flexible = MuscleHillModels.FlexibleTendonAlwaysPositive

    model = RigidbodyModels.WithMuscles(
        model_path,
        muscles=[
            rigid("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.123, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.130, maximal_velocity=5.0),
            rigid("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.123, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.10, tendon_slack_length=0.130, maximal_velocity=5.0),
            flexible("Mus1", maximal_force=500, optimal_length=0.15, tendon_slack_length=0.130, maximal_velocity=5.0),
        ],
    )

    jac = model.function_to_dm(model.muscle_tendon_length_jacobian, q=np.array([-0.3]))
    assert jac.shape == (6, 1)
    np.testing.assert_almost_equal(jac.T, [[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])


def test_muscle_biorbd_model_muscle_jacobian():
    model_none = RigidbodyModels.WithMuscles(model_path, muscles=[])

    # Do not allow to call muscle_length_jacobian directly
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "In the context of this project, the name 'muscle_length_jacobian' is confusing as it is the jacobian of "
            "the muscle-tendon-unit length (as opposed to the muscle-fiber-unit length)."
        ),
    ):
        model_none.muscle_length_jacobian(q=[])


def test_tendon_lengths():
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

    activations = np.array([0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2])
    q = np.array([-0.3])
    qdot = np.array([0.1])

    tendon_lengths = model.function_to_dm(model.tendon_lengths, activations=activations, q=q, qdot=qdot)
    assert tendon_lengths.shape == (8, 1)
    np.testing.assert_almost_equal(
        tendon_lengths.T,
        [[0.123, 0.13, 0.13, 0.13, 0.14806816214724597, 0.1549001402515909, 0.14988633341690807, 0.15239805246705462]],
    )


def test_tendon_forces():
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

    activations = np.array([0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2])
    q = np.array([-0.3])
    qdot = np.array([0.1])

    tendon_forces = model.function_to_dm(model.tendon_forces, activations=activations, q=q, qdot=qdot)
    assert tendon_forces.shape == (8, 1)
    np.testing.assert_almost_equal(
        tendon_forces.T, [[0.0, 0.0, 0.0, 0.0, 298.264445, 194.07003687, 50.1407764, 98.82980024]]
    )


def test_muscle_biorbd_model_muscle_fiber_lengths():
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

    activations = np.array([0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2])
    q = np.array([-0.3])
    qdot = np.array([0.1])

    lengths = model.function_to_dm(model.muscle_fiber_lengths, activations=activations, q=q, qdot=qdot)
    assert lengths.shape == (8, 1)
    np.testing.assert_almost_equal(
        lengths.T, [[0.177, 0.17, 0.17, 0.17, 0.15193184, 0.14509986, 0.15011367, 0.14760195]]
    )


def test_muscle_biorbd_model_muscle_fiber_lengths_equilibrated():
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

    activations = np.array([0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2])
    q = np.array([-0.3])
    qdot = np.array([0.1])

    lengths = model.function_to_dm(model.muscle_fiber_lengths_equilibrated, activations=activations, q=q, qdot=qdot)
    assert lengths.shape == (8, 1)
    np.testing.assert_almost_equal(
        lengths.T, [[0.177, 0.17, 0.17, 0.17, 0.15193184, 0.14509986, 0.15011367, 0.14760195]]
    )


def test_muscle_biorbd_model_muscle_fiber_velocities():
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

    activations = np.array([0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2])
    q = np.array([-0.3])
    qdot = np.array([0.1])
    muscle_fiber_lengths_equilibrated = model.function_to_dm(
        model.muscle_fiber_lengths_equilibrated, activations=activations, q=q, qdot=qdot
    )

    muscle_fiber_velocities = model.function_to_dm(
        model.muscle_fiber_velocities,
        activations=activations,
        q=q,
        qdot=qdot,
        muscle_fiber_lengths=muscle_fiber_lengths_equilibrated,
        muscle_fiber_velocity_initial_guesses=np.zeros((model.nb_muscles, 1)),
    )
    assert muscle_fiber_velocities.shape == (8, 1)
    np.testing.assert_almost_equal(muscle_fiber_velocities.T, [[-0.1, -0.1, -0.1, -0.1, 0.0, 0.0, 0.0, 0.0]])

    muscle_fiber_velocities = model.function_to_dm(
        model.muscle_fiber_velocities,
        muscle_fiber_lengths=muscle_fiber_lengths_equilibrated - 0.0001,
        muscle_fiber_velocity_initial_guesses=np.zeros((model.nb_muscles, 1)),
        activations=activations,
        q=q,
        qdot=qdot,
    )
    np.testing.assert_almost_equal(
        muscle_fiber_velocities.T,
        [[-0.1, -0.1, -0.1, -0.1, 11.187527520733362, 1.4657000024215545, 0.059596149458245075, 0.05815367592241861]],
    )


def test_muscle_biorbd_model_muscle_forces():
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

    activations = np.array([0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2])
    q = np.array([-0.3])
    qdot = np.array([0.1])
    muscle_fiber_lengths_equilibrated = model.function_to_dm(
        model.muscle_fiber_lengths_equilibrated, activations=activations, q=q, qdot=qdot
    )
    muscle_fiber_velocities = model.function_to_dm(
        model.muscle_fiber_velocities,
        muscle_fiber_lengths=muscle_fiber_lengths_equilibrated,
        activations=activations,
        q=q,
        qdot=qdot,
        muscle_fiber_velocity_initial_guesses=np.zeros((model.nb_muscles, 1)),
    )

    muscle_forces = model.function_to_dm(
        model.muscle_forces,
        muscle_fiber_lengths=muscle_fiber_lengths_equilibrated,
        muscle_fiber_velocities=muscle_fiber_velocities,
        activations=activations,
        q=q,
        qdot=qdot,
    )
    np.testing.assert_almost_equal(
        muscle_forces.T,
        [[1574.49953068, 985.75728269, 56.69574069, 100.02884243, 298.264445, 194.07003687, 50.1407764, 98.82980024]],
    )


def test_muscle_biorbd_model_muscle_joint_torque():
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

    activations = np.array([0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2])
    q = np.array([-0.3])
    qdot = np.array([0.1])
    muscle_fiber_lengths_equilibrated = model.function_to_dm(
        model.muscle_fiber_lengths_equilibrated, activations=activations, q=q, qdot=qdot
    )
    muscle_fiber_velocities = model.function_to_dm(
        model.muscle_fiber_velocities,
        muscle_fiber_lengths=muscle_fiber_lengths_equilibrated,
        activations=activations,
        q=q,
        qdot=qdot,
        muscle_fiber_velocity_initial_guesses=np.zeros((model.nb_muscles, 1)),
    )

    muscle_joint_torque = model.function_to_dm(
        model.muscle_joint_torque,
        muscle_fiber_lengths=muscle_fiber_lengths_equilibrated,
        muscle_fiber_velocities=muscle_fiber_velocities,
        activations=activations,
        q=q,
        qdot=qdot,
    )
    np.testing.assert_almost_equal(muscle_joint_torque.T, [[3358.2864550040204]])


def test_muscle_biorbd_model_casadi_function_interface():
    model = RigidbodyModels.WithMuscles(
        model_path,
        muscles=[
            MuscleHillModels.RigidTendon(
                name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
            )
        ],
    )

    def my_small_function(activations: MX, q: MX) -> MX:
        return activations + q

    def my_long_function(activations: MX, q: MX, qdot: MX) -> MX:
        return my_small_function(activations, q) + qdot

    # Only certain inputs are allowed
    with pytest.raises(
        ValueError,
        match=(
            "Expected 'q', 'qdot', 'activations', 'muscle_fiber_lengths', 'muscle_fiber_velocities', 'muscle_fiber_velocity_initial_guesses' but got dummy"
        ),
    ):
        model.to_casadi_function(my_small_function, "dummy")

    long_func = model.to_casadi_function(my_long_function, "q", "qdot", "activations")
    np.testing.assert_almost_equal(float(long_func(activations=1, q=2, qdot=3)["output"]), 6.0)
    np.testing.assert_almost_equal(float(model.function_to_dm(my_long_function, activations=1, q=2, qdot=3)), 6.0)

    small_func = model.to_casadi_function(my_small_function, "activations", "q")
    np.testing.assert_almost_equal(float(small_func(activations=1, q=2)["output"]), 3.0)
    np.testing.assert_almost_equal(float(model.function_to_dm(my_small_function, activations=1, q=2)), 3.0)
