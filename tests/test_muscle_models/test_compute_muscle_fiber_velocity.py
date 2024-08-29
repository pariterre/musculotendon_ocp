from functools import partial
import os

from casadi import MX
from musculotendon_ocp import (
    RigidbodyModels,
    MuscleHillModels,
    ComputeMuscleFiberVelocityMethods,
    ComputeForceDampingMethods,
)
import numpy as np
import pytest

model_path = (
    (os.getcwd() + "/musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod")
    .replace("\\", "/")
    .replace("c:/", "C:/")
)


def test_compute_muscle_fiber_velocity_methods():
    assert len(ComputeMuscleFiberVelocityMethods) == 4

    rigid_tendon = ComputeMuscleFiberVelocityMethods.RigidTendon()
    assert type(rigid_tendon) == ComputeMuscleFiberVelocityMethods.RigidTendon.value

    flexible_tendon_implicit = ComputeMuscleFiberVelocityMethods.FlexibleTendonImplicit()
    assert type(flexible_tendon_implicit) == ComputeMuscleFiberVelocityMethods.FlexibleTendonImplicit.value

    flexible_tendon_explicit = ComputeMuscleFiberVelocityMethods.FlexibleTendonExplicit()
    assert type(flexible_tendon_explicit) == ComputeMuscleFiberVelocityMethods.FlexibleTendonExplicit.value

    flexible_tendon_linearized = ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized()
    assert type(flexible_tendon_linearized) == ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized.value


def test_compute_muscle_fiber_velocity_rigid_tendon():
    mus = MuscleHillModels.RigidTendon(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])
    compute_muscle_velocity_length = ComputeMuscleFiberVelocityMethods.RigidTendon()

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


def test_compute_muscle_fiber_velocity_flexible_tendon_implicit():
    mus = MuscleHillModels.FlexibleTendonAlwaysPositive(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])

    compute_muscle_fiber_velocity = ComputeMuscleFiberVelocityMethods.FlexibleTendonImplicit()

    activation = np.array([0.5])
    q = np.ones(model.nb_q) * -0.2
    qdot = np.ones(model.nb_qdot)

    muscle_fiber_velocity = float(
        compute_muscle_fiber_velocity(
            activation=activation,
            q=q,
            qdot=qdot,
            muscle=mus,
            model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
            biorbd_muscle=model.model.muscle(0),
            muscle_fiber_length=np.array([0.1]),
        )
    )
    np.testing.assert_almost_equal(muscle_fiber_velocity, -5.201202604749881)


def test_compute_muscle_fiber_velocity_flexible_tendon_implicit_wrong_constructor():
    mus = MuscleHillModels.RigidTendon(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])

    mx_symbolic = MX.sym("muscle_fiber_length", 1, 1)
    compute_muscle_fiber_velocity = ComputeMuscleFiberVelocityMethods.FlexibleTendonImplicit(mx_symbolic=mx_symbolic)

    activation = np.array([0.5])
    q = np.ones(model.nb_q) * -0.2
    qdot = np.ones(model.nb_qdot)

    with pytest.raises(
        ValueError, match="The compute_muscle_fiber_length must not be a ComputeMuscleFiberLengthRigidTendon"
    ):
        compute_muscle_fiber_velocity(
            activation=activation,
            q=q,
            qdot=qdot,
            muscle=mus,
            model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
            biorbd_muscle=model.model.muscle(0),
            muscle_fiber_length=np.array([0.1]),
        )


def test_compute_muscle_fiber_velocity_flexible_tendon_explicit():
    mus = MuscleHillModels.FlexibleTendonAlwaysPositive(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])

    compute_muscle_fiber_velocity = ComputeMuscleFiberVelocityMethods.FlexibleTendonExplicit()

    activation = np.array([0.5])
    q = np.ones(model.nb_q) * -0.2
    qdot = np.ones(model.nb_qdot)

    muscle_fiber_velocity = model.function_to_dm(
        partial(
            compute_muscle_fiber_velocity,
            muscle=mus,
            model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
            biorbd_muscle=model.model.muscle(0),
            activation=activation,
            muscle_fiber_length=np.array([0.1]),
        ),
        q=q,
        qdot=qdot,
    )

    np.testing.assert_almost_equal(muscle_fiber_velocity, -5.201202604749881)


def test_compute_muscle_fiber_velocity_flexible_tendon_explicit_wrong_constructor():
    mus = MuscleHillModels.RigidTendon(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])

    mx_symbolic = MX.sym("muscle_fiber_length", 1, 1)
    compute_muscle_fiber_velocity = ComputeMuscleFiberVelocityMethods.FlexibleTendonExplicit(mx_symbolic=mx_symbolic)

    activation = np.array([0.5])
    q = np.ones(model.nb_q) * -0.2
    qdot = np.ones(model.nb_qdot)

    with pytest.raises(
        ValueError, match="The compute_muscle_fiber_length must not be a ComputeMuscleFiberLengthRigidTendon"
    ):
        compute_muscle_fiber_velocity(
            activation=activation,
            q=q,
            qdot=qdot,
            muscle=mus,
            model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
            biorbd_muscle=model.model.muscle(0),
            muscle_fiber_length=np.array([0.1]),
        )


def test_compute_muscle_fiber_velocity_flexible_tendon_linearized():
    def evaluate(muscle_fiber_velocity: float):
        return float(
            model.function_to_dm(
                partial(
                    mus.compute_muscle_fiber_velocity,
                    muscle=mus,
                    model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
                    biorbd_muscle=model.model.muscle(0),
                    activation=activation,
                    muscle_fiber_length=np.array([muscle_fiber_velocity]),
                ),
                q=q,
                qdot=qdot,
            )
        )

    mus = MuscleHillModels.FlexibleTendonAlwaysPositive(
        name="Mus1",
        maximal_force=500,
        optimal_length=0.1,
        tendon_slack_length=0.123,
        maximal_velocity=5.0,
        compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized(),
        compute_force_damping=ComputeForceDampingMethods.Linear(0.1),
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])

    activation = np.array([0.5])
    q = np.ones(model.nb_q) * -0.2
    qdot = np.ones(model.nb_qdot)

    np.testing.assert_almost_equal(evaluate(0.1), -1.9093250424184014)
    np.testing.assert_almost_equal(evaluate(0.2), -676.3254672085245)
