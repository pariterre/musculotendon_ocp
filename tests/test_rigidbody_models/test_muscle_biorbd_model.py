import os
import re

from casadi import MX, Function
from musculotendon_ocp import (
    MuscleBiorbdModel,
    MuscleModelHillRigidTendon,
    MuscleModelHillFlexibleTendon,
    ComputeMuscleFiberLengthAsVariable,
    ComputeMuscleFiberVelocityFlexibleTendon,
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
        MuscleBiorbdModel(
            model_path,
            muscles=[
                MuscleModelHillRigidTendon(
                    name="Wrong muscle",
                    maximal_force=500,
                    optimal_length=0.1,
                    tendon_slack_length=0.123,
                    maximal_velocity=5.0,
                )
            ],
        )


def test_muscle_biorbd_model_number_of_muscles():
    model = MuscleBiorbdModel(
        model_path,
        muscles=[
            MuscleModelHillRigidTendon(
                name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
            )
        ],
    )
    assert model.nb_muscles == 1

    # Can load a model with less muscles
    model_none = MuscleBiorbdModel(model_path, muscles=[])
    assert model_none.nb_muscles == 0


def test_get_mx_variables():
    model = MuscleBiorbdModel(
        model_path,
        muscles=[
            MuscleModelHillRigidTendon(
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
    model = MuscleBiorbdModel(
        model_path,
        muscles=[
            MuscleModelHillRigidTendon(name="Mus1", **dummy_params),
            MuscleModelHillRigidTendon(name="Mus1", **dummy_params),
            MuscleModelHillFlexibleTendon(
                name="Mus1",
                compute_muscle_fiber_length=ComputeMuscleFiberLengthAsVariable(mx_symbolic=muscle_fiber_length_mx),
                compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityFlexibleTendon(
                    mx_symbolic=muscle_fiber_velocity_mx
                ),
                **dummy_params
            ),
            MuscleModelHillFlexibleTendon(name="Mus1", **dummy_params),
        ],
    )

    muscle_fiber_length_mx_variables = model.muscle_fiber_length_mx_variables
    assert isinstance(muscle_fiber_length_mx_variables, MX)
    assert muscle_fiber_length_mx_variables.shape == (2, 1)
    np.testing.assert_almost_equal(
        Function("f", [muscle_fiber_length_mx], [muscle_fiber_length_mx_variables[0]])(1.234), 1.234
    )
    with pytest.raises(RuntimeError):
        # Make sure the muscle_fiber_length_mx_variable[0] is actually a MX.sym variable
        Function("f", [MX.sym("dummy", 1, 1)], [muscle_fiber_length_mx_variables[0]])
    with pytest.raises(RuntimeError):
        # Make sure the muscle_fiber_length_mx_variable[1] is not the same as muscle_fiber_length_mx_variable[0]
        Function("f", [muscle_fiber_length_mx], [muscle_fiber_length_mx_variables[1]])

    muscle_fiber_velocity_mx_variables = model.muscle_fiber_velocity_mx_variables
    assert isinstance(muscle_fiber_velocity_mx_variables, MX)
    assert muscle_fiber_velocity_mx_variables.shape == (2, 1)
    np.testing.assert_almost_equal(
        Function("f", [muscle_fiber_velocity_mx], [muscle_fiber_velocity_mx_variables[0]])(1.234), 1.234
    )
    with pytest.raises(RuntimeError):
        # Make sure the muscle_fiber_velocity_mx_variables[0] is actually a MX.sym variable
        Function("f", [MX.sym("dummy", 1, 1)], [muscle_fiber_velocity_mx_variables[0]])
    with pytest.raises(RuntimeError):
        # Make sure the muscle_fiber_velocity_mx_variables[1] is not the same as muscle_fiber_velocity_mx_variables[0]
        Function("f", [muscle_fiber_velocity_mx], [muscle_fiber_velocity_mx_variables[1]])


def test_muscle_tendon_lengths():
    rigid = MuscleModelHillRigidTendon
    flexible = MuscleModelHillFlexibleTendon

    model = MuscleBiorbdModel(
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
    rigid = MuscleModelHillRigidTendon
    flexible = MuscleModelHillFlexibleTendon

    model = MuscleBiorbdModel(
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
    model_none = MuscleBiorbdModel(model_path, muscles=[])

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
    rigid = MuscleModelHillRigidTendon
    flexible = MuscleModelHillFlexibleTendon

    model = MuscleBiorbdModel(
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
    rigid = MuscleModelHillRigidTendon
    flexible = MuscleModelHillFlexibleTendon

    model = MuscleBiorbdModel(
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


def test_muscle_biorbd_model_muscle_fiber_length():
    rigid = MuscleModelHillRigidTendon
    flexible = MuscleModelHillFlexibleTendon

    model = MuscleBiorbdModel(
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


def test_muscle_biorbd_model_muscle_fiber_length_equilibrated():
    rigid = MuscleModelHillRigidTendon
    flexible = MuscleModelHillFlexibleTendon

    model = MuscleBiorbdModel(
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


def test_muscle_biorbd_model_muscle_fiber_velocity():
    rigid = MuscleModelHillRigidTendon
    flexible = MuscleModelHillFlexibleTendon

    model = MuscleBiorbdModel(
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

    # TODO RENDU ICI TRYING TO FIX THE FREE VARIABLES
    muscle_fiber_velocities = model.function_to_dm(
        model.muscle_fiber_velocities, activations=activations, q=q, qdot=qdot
    )
    assert muscle_fiber_velocities.shape == (1, 1)
    np.testing.assert_almost_equal(muscle_fiber_velocities.T, [[-0.2]])


def test_muscle_biorbd_model_casadi_function_interface():
    model = MuscleBiorbdModel(
        model_path,
        muscles=[
            MuscleModelHillRigidTendon(
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
        match="Expected 'q', 'qdot' or 'activations', got dummy",
    ):
        model.to_casadi_function(my_small_function, "dummy")

    long_func = model.to_casadi_function(my_long_function, "q", "qdot", "activations")
    np.testing.assert_almost_equal(float(long_func(activations=1, q=2, qdot=3)["output"]), 6.0)
    np.testing.assert_almost_equal(float(model.function_to_dm(my_long_function, activations=1, q=2, qdot=3)), 6.0)

    small_func = model.to_casadi_function(my_small_function, "activations", "q")
    np.testing.assert_almost_equal(float(small_func(activations=1, q=2)["output"]), 3.0)
    np.testing.assert_almost_equal(float(model.function_to_dm(my_small_function, activations=1, q=2)), 3.0)
