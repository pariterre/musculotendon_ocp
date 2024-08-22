from functools import partial
import os

from casadi import MX
from musculotendon_ocp import (
    MuscleBiorbdModel,
    MuscleModelHillRigidTendon,
    MuscleModelHillFlexibleTendonAlwaysPositive,
    ComputeMuscleFiberVelocityRigidTendon,
    ComputeMuscleFiberVelocityFlexibleTendon,
)
import numpy as np
import pytest

model_path = (
    (os.getcwd() + "/musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod")
    .replace("\\", "/")
    .replace("c:/", "C:/")
)


def test_compute_muscle_fiber_velocity_rigid_tendon():
    mus = MuscleModelHillRigidTendon(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = MuscleBiorbdModel(model_path, muscles=[mus])
    compute_muscle_velocity_length = ComputeMuscleFiberVelocityRigidTendon()

    activation = np.array([0.5])
    q = np.ones(model.nb_q) * -0.2
    qdot = np.ones(model.nb_qdot) * 0.5

    muscle_fiber_velocity = float(
        model.function_to_dm(
            partial(
                compute_muscle_velocity_length,
                muscle=mus,
                model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
                biorbd_muscle=model.model.muscle(0),
                activation=activation,
                muscle_fiber_length=None,
            ),
            q=q,
            qdot=qdot,
        )
    )
    np.testing.assert_almost_equal(muscle_fiber_velocity, -0.5)


def test_compute_muscle_fiber_velocity_flexible_tendon():
    mus = MuscleModelHillFlexibleTendonAlwaysPositive(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = MuscleBiorbdModel(model_path, muscles=[mus])

    compute_muscle_fiber_velocity = ComputeMuscleFiberVelocityFlexibleTendon()

    activation = np.array([0.5])
    q = np.ones(model.nb_q) * -0.2
    qdot = np.ones(model.nb_qdot)

    muscle_fiber_velocity = float(
        compute_muscle_fiber_velocity(
            activation=activation,
            q=np.array([-0.2]),
            qdot=qdot,
            muscle=mus,
            model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
            biorbd_muscle=model.model.muscle(0),
            muscle_fiber_length=np.array([0.1]),
        )
    )
    np.testing.assert_almost_equal(muscle_fiber_velocity, -5.201202604749881)


def test_compute_muscle_fiber_velocity_flexible_tendon_wrong_constructor():
    mus = MuscleModelHillRigidTendon(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = MuscleBiorbdModel(model_path, muscles=[mus])

    mx_symbolic = MX.sym("muscle_fiber_length", 1, 1)
    compute_muscle_fiber_velocity = ComputeMuscleFiberVelocityFlexibleTendon(mx_symbolic=mx_symbolic)

    activation = np.array([0.5])
    q = np.ones(model.nb_q) * -0.2
    qdot = np.ones(model.nb_qdot)

    with pytest.raises(
        ValueError, match="The compute_muscle_fiber_length must not be a ComputeMuscleFiberLengthRigidTendon"
    ):
        compute_muscle_fiber_velocity(
            activation=activation,
            q=np.array([-0.2]),
            qdot=qdot,
            muscle=mus,
            model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
            biorbd_muscle=model.model.muscle(0),
            muscle_fiber_length=np.array([0.1]),
        )
