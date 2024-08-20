import os
import re

from musculotendon_ocp import MuscleBiorbdModel, MuscleModelHillRigidTendon
import numpy as np
import pytest

model_path = (
    (os.getcwd() + "/musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod")
    .replace("\\", "/")
    .replace("c:/", "C:/")
)


def test_muscle_biorbd_model_construction():
    with pytest.raises(ValueError, match="Muscle Wrong muscle was not found in the biorbd model"):
        MuscleBiorbdModel(
            model_path,
            muscles=[
                MuscleModelHillRigidTendon(
                    name="Wrong muscle", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123
                )
            ],
        )

    # Can load a model with the right muscles
    model_all = MuscleBiorbdModel(
        model_path,
        muscles=[
            MuscleModelHillRigidTendon(name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123)
        ],
    )
    assert model_all.nb_muscles == 1

    # Can load a model with less muscles
    model_none = MuscleBiorbdModel(model_path, muscles=[])
    assert model_none.nb_muscles == 0


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


def test_muscle_biorbd_model_muscle_fiber_length():
    model = MuscleBiorbdModel(
        model_path,
        muscles=[
            MuscleModelHillRigidTendon(name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123),
        ],
    )

    lengths = model.function_to_dm(
        model.muscle_fiber_lengths, activations=np.array([0.5]), q=np.array([-0.3]), qdot=np.array([0.1])
    )
    assert lengths.shape == (1, 1)
    np.testing.assert_almost_equal(lengths, [[0.177]])  # q + tendon_slack_length


def test_muscle_biorbd_model_muscle_fiber_velocity():
    model = MuscleBiorbdModel(
        model_path,
        muscles=[
            MuscleModelHillRigidTendon(name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123),
        ],
    )

    model.muscle_fiber_velocities(activations=np.array([0.5]), q=np.array([-0.3]), qdot=np.array([0.2]))
    velocities = model.function_to_dm(
        model.muscle_fiber_velocities, activations=np.array([0.5]), q=np.array([-0.3]), qdot=np.array([0.2])
    )
    assert velocities.shape == (1, 1)
    np.testing.assert_almost_equal(velocities, [[-0.2]])


def test_muscle_biorbd_model_muscle_tendon_length_jacobian():
    model = MuscleBiorbdModel(
        model_path,
        muscles=[
            MuscleModelHillRigidTendon(name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123),
        ],
    )

    jac = model.function_to_dm(model.muscle_tendon_length_jacobian, q=np.array([-0.3]))
    assert jac.shape == (1, 1)
    np.testing.assert_almost_equal(jac, [[-1.0]])
