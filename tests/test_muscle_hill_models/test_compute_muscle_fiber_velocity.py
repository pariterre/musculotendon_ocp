from functools import partial
import pathlib

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
    pathlib.Path(__file__).parent.resolve()
    / "../../musculotendon_ocp/rigidbody_models/models/one_muscle_holding_a_cube.bioMod"
).as_posix()


def test_compute_muscle_fiber_velocity_methods():
    assert len(ComputeMuscleFiberVelocityMethods) == 5

    rigid_tendon = ComputeMuscleFiberVelocityMethods.RigidTendon()
    assert type(rigid_tendon) == ComputeMuscleFiberVelocityMethods.RigidTendon.value

    flexible_tendon_from_force_defects = ComputeMuscleFiberVelocityMethods.FlexibleTendonFromForceDefects()
    assert (
        type(flexible_tendon_from_force_defects)
        == ComputeMuscleFiberVelocityMethods.FlexibleTendonFromForceDefects.value
    )

    flexible_tendon_from_velocity_defects = ComputeMuscleFiberVelocityMethods.FlexibleTendonFromVelocityDefects()
    assert (
        type(flexible_tendon_from_velocity_defects)
        == ComputeMuscleFiberVelocityMethods.FlexibleTendonFromVelocityDefects.value
    )

    flexible_tendon_linearized = ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized()
    assert type(flexible_tendon_linearized) == ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized.value

    flexible_tendon_quadratic = ComputeMuscleFiberVelocityMethods.FlexibleTendonQuadratic()
    assert type(flexible_tendon_quadratic) == ComputeMuscleFiberVelocityMethods.FlexibleTendonQuadratic.value

    with pytest.raises(ValueError, match="Cannot deserialize Unknown as ComputeMuscleFiberVelocityMethods"):
        ComputeMuscleFiberVelocityMethods.deserialize({"method": "Unknown"})


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
                muscle_fiber_velocity_initial_guess=None,
            ),
            q=q,
            qdot=qdot,
        )
    )
    np.testing.assert_almost_equal(muscle_fiber_velocity, -0.5)

    # Test serialization
    serialized = compute_muscle_velocity_length.serialize()
    assert serialized == {"method": "ComputeMuscleFiberVelocityRigidTendon"}
    deserialized = ComputeMuscleFiberVelocityMethods.deserialize(serialized)
    assert type(deserialized) == ComputeMuscleFiberVelocityMethods.RigidTendon.value

    with pytest.raises(ValueError, match="Cannot deserialize Unknown as ComputeMuscleFiberVelocityRigidTendon"):
        ComputeMuscleFiberVelocityMethods.RigidTendon.value.deserialize({"method": "Unknown"})


def test_compute_muscle_fiber_velocity_flexible_tendon_from_force_defects():
    mus = MuscleHillModels.FlexibleTendonAlwaysPositive(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])

    compute_muscle_fiber_velocity = ComputeMuscleFiberVelocityMethods.FlexibleTendonFromForceDefects()

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
            muscle_fiber_velocity_initial_guess=np.array([0.0]),
        )
    )
    np.testing.assert_almost_equal(muscle_fiber_velocity, -5.201202604749881)

    # Test serialization
    serialized = compute_muscle_fiber_velocity.serialize()
    assert serialized == {"method": "ComputeMuscleFiberVelocityFlexibleTendonFromForceDefects"}
    deserialized = ComputeMuscleFiberVelocityMethods.deserialize(serialized)
    assert type(deserialized) == ComputeMuscleFiberVelocityMethods.FlexibleTendonFromForceDefects.value

    with pytest.raises(
        ValueError, match="Cannot deserialize Unknown as ComputeMuscleFiberVelocityFlexibleTendonFromForceDefects"
    ):
        ComputeMuscleFiberVelocityMethods.FlexibleTendonFromForceDefects.value.deserialize({"method": "Unknown"})


def test_compute_muscle_fiber_velocity_flexible_tendon_from_force_defects_wrong_constructor():
    mus = MuscleHillModels.RigidTendon(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])

    mx_symbolic = MX.sym("muscle_fiber_length", 1, 1)
    compute_muscle_fiber_velocity = ComputeMuscleFiberVelocityMethods.FlexibleTendonFromForceDefects(
        mx_symbolic=mx_symbolic
    )

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
            muscle_fiber_velocity_initial_guess=np.array([0.0]),
        )


def test_compute_muscle_fiber_velocity_flexible_tendon_from_velocity_defects():
    mus = MuscleHillModels.FlexibleTendonAlwaysPositive(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])

    compute_muscle_fiber_velocity = ComputeMuscleFiberVelocityMethods.FlexibleTendonFromVelocityDefects()

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
            muscle_fiber_velocity_initial_guess=np.array([0.0]),
        ),
        q=q,
        qdot=qdot,
    )

    np.testing.assert_almost_equal(muscle_fiber_velocity, -5.201202604749881)

    # Test serialization
    serialized = compute_muscle_fiber_velocity.serialize()
    assert serialized == {"method": "ComputeMuscleFiberVelocityFlexibleTendonFromVelocityDefects"}
    deserialized = ComputeMuscleFiberVelocityMethods.deserialize(serialized)
    assert type(deserialized) == ComputeMuscleFiberVelocityMethods.FlexibleTendonFromVelocityDefects.value

    with pytest.raises(
        ValueError, match="Cannot deserialize Unknown as ComputeMuscleFiberVelocityFlexibleTendonFromVelocityDefects"
    ):
        ComputeMuscleFiberVelocityMethods.FlexibleTendonFromVelocityDefects.value.deserialize({"method": "Unknown"})


def test_compute_muscle_fiber_velocity_flexible_tendon_from_velocity_defects_wrong_constructor():
    mus = MuscleHillModels.RigidTendon(
        name="Mus1", maximal_force=500, optimal_length=0.1, tendon_slack_length=0.123, maximal_velocity=5.0
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])

    mx_symbolic = MX.sym("muscle_fiber_length", 1, 1)
    compute_muscle_fiber_velocity = ComputeMuscleFiberVelocityMethods.FlexibleTendonFromVelocityDefects(
        mx_symbolic=mx_symbolic
    )

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
            muscle_fiber_velocity_initial_guess=np.array([0.0]),
        )


def test_compute_muscle_fiber_velocity_flexible_tendon_linearized():
    def evaluate(muscle_fiber_length: float, muscle_fiber_velocity_initial_guess: float):
        return float(
            model.function_to_dm(
                partial(
                    mus.compute_muscle_fiber_velocity,
                    muscle=mus,
                    model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
                    biorbd_muscle=model.model.muscle(0),
                    activation=activation,
                    muscle_fiber_length=np.array([muscle_fiber_length]),
                    muscle_fiber_velocity_initial_guess=np.array([muscle_fiber_velocity_initial_guess]),
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

    np.testing.assert_almost_equal(evaluate(0.1, 0.0), -1.9093250424184014)
    np.testing.assert_almost_equal(evaluate(0.1, -3.4), -3.437901945795994)
    np.testing.assert_almost_equal(evaluate(0.2, 0.0), -676.3254672085245)
    np.testing.assert_almost_equal(evaluate(0.2, -731.8), -731.8420239027942)

    # Test serialization
    serialized = mus.compute_muscle_fiber_velocity.serialize()
    assert serialized == {"method": "ComputeMuscleFiberVelocityFlexibleTendonLinearized"}
    deserialized = ComputeMuscleFiberVelocityMethods.deserialize(serialized)
    assert type(deserialized) == ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized.value

    with pytest.raises(
        ValueError, match="Cannot deserialize Unknown as ComputeMuscleFiberVelocityFlexibleTendonLinearized"
    ):
        ComputeMuscleFiberVelocityMethods.FlexibleTendonLinearized.value.deserialize({"method": "Unknown"})


def test_compute_muscle_fiber_velocity_flexible_tendon_quadratic():
    def evaluate(muscle_fiber_length: float, muscle_fiber_velocity_initial_guess: float):
        return float(
            model.function_to_dm(
                partial(
                    mus.compute_muscle_fiber_velocity,
                    muscle=mus,
                    model_kinematic_updated=model.model.UpdateKinematicsCustom(q),
                    biorbd_muscle=model.model.muscle(0),
                    activation=activation,
                    muscle_fiber_length=np.array([muscle_fiber_length]),
                    muscle_fiber_velocity_initial_guess=np.array([muscle_fiber_velocity_initial_guess]),
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
        compute_muscle_fiber_velocity=ComputeMuscleFiberVelocityMethods.FlexibleTendonQuadratic(),
        compute_force_damping=ComputeForceDampingMethods.Linear(0.1),
    )
    model = RigidbodyModels.WithMuscles(model_path, muscles=[mus])

    activation = np.array([0.5])
    q = np.ones(model.nb_q) * -0.2
    qdot = np.ones(model.nb_qdot)

    np.testing.assert_almost_equal(evaluate(0.1, 0.0), -1.4148579102945984)
    np.testing.assert_almost_equal(evaluate(0.1, -3.4), -3.438058496100624)
    np.testing.assert_almost_equal(evaluate(0.2, 0.0), -159.10620315897017)
    np.testing.assert_almost_equal(evaluate(0.2, -731.8), -731.8420239027369)

    # Test serialization
    serialized = mus.compute_muscle_fiber_velocity.serialize()
    assert serialized == {"method": "ComputeMuscleFiberVelocityFlexibleTendonQuadratic"}
    deserialized = ComputeMuscleFiberVelocityMethods.deserialize(serialized)
    assert type(deserialized) == ComputeMuscleFiberVelocityMethods.FlexibleTendonQuadratic.value

    with pytest.raises(
        ValueError, match="Cannot deserialize Unknown as ComputeMuscleFiberVelocityFlexibleTendonQuadratic"
    ):
        ComputeMuscleFiberVelocityMethods.FlexibleTendonQuadratic.value.deserialize({"method": "Unknown"})
