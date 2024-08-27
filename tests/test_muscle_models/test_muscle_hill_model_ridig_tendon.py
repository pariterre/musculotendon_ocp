import re

from musculotendon_ocp import (
    MuscleHillModels,
    ComputeForcePassiveMethods,
    ComputeForceActiveMethods,
    ComputeForceDampingMethods,
    ComputeForceVelocityMethods,
)
from numpy.testing import assert_almost_equal
import pytest


def test_muscle_hill_model_rigid_tendon():
    name = "Dummy"
    maximal_force = 123
    optimal_length = 0.123
    tendon_slack_length = 0.123
    maximal_velocity = 0.456
    force_passive = ComputeForceVelocityMethods.HillType()
    force_active = ComputeForceActiveMethods.HillType()
    force_damping = ComputeForceDampingMethods.Constant()
    force_velocity = ComputeForceVelocityMethods.HillType()

    model = MuscleHillModels.RigidTendon(
        name=name,
        maximal_force=maximal_force,
        optimal_length=optimal_length,
        tendon_slack_length=tendon_slack_length,
        maximal_velocity=maximal_velocity,
        compute_force_passive=force_passive,
        compute_force_active=force_active,
        compute_force_damping=force_damping,
        compute_force_velocity=force_velocity,
    )

    assert model.name == name
    assert model.maximal_force == maximal_force
    assert model.optimal_length == optimal_length
    assert model.tendon_slack_length == tendon_slack_length
    assert model.maximal_velocity == maximal_velocity
    assert id(model.compute_force_passive) == id(force_passive)
    assert id(model.compute_force_active) == id(force_active)
    assert id(model.compute_force_damping) == id(force_damping)
    assert id(model.compute_force_velocity) == id(force_velocity)

    model_default = MuscleHillModels.RigidTendon(
        name=name,
        maximal_force=maximal_force,
        optimal_length=optimal_length,
        tendon_slack_length=tendon_slack_length,
        maximal_velocity=maximal_velocity,
    )
    assert model_default.compute_force_passive.__dict__ == ComputeForcePassiveMethods.HillType().__dict__
    assert model_default.compute_force_active.__dict__ == ComputeForceActiveMethods.HillType().__dict__
    assert model_default.compute_force_damping.__dict__ == ComputeForceDampingMethods.Constant().__dict__
    assert model_default.compute_force_velocity.__dict__ == ComputeForceVelocityMethods.HillType().__dict__


def test_muscle_hill_model_rigid_tendon_normalize_muscle_fiber_length():
    optimal_length = 0.123
    model = MuscleHillModels.RigidTendon(
        name="Dummy", maximal_force=123, optimal_length=optimal_length, tendon_slack_length=0.123, maximal_velocity=5.0
    )

    fiber_length = 0.456
    normalized_fiber_length = model.normalize_muscle_fiber_length(fiber_length)
    assert_almost_equal(normalized_fiber_length, fiber_length / optimal_length)

    denormalized_fiber_length = model.denormalize_muscle_fiber_length(normalized_fiber_length)
    assert_almost_equal(denormalized_fiber_length, fiber_length)


def test_muscle_hill_model_rigid_tendon_normalize_muscle_fiber_velocity():
    maximal_velocity = 0.456
    model = MuscleHillModels.RigidTendon(
        name="Dummy",
        maximal_force=123,
        optimal_length=0.123,
        maximal_velocity=maximal_velocity,
        tendon_slack_length=0.123,
    )

    fiber_velocity = 0.789
    normalized_fiber_velocity = model.normalize_muscle_fiber_velocity(fiber_velocity)
    assert_almost_equal(normalized_fiber_velocity, fiber_velocity / maximal_velocity)

    denormalized_fiber_velocity = model.denormalize_muscle_fiber_velocity(normalized_fiber_velocity)
    assert_almost_equal(denormalized_fiber_velocity, fiber_velocity)


def test_muscle_hill_model_rigid_tendon_normalize_tendon_length():
    model = MuscleHillModels.RigidTendon(
        name="Dummy", maximal_force=123, optimal_length=0.123, tendon_slack_length=0.123, maximal_velocity=5.0
    )

    with pytest.raises(RuntimeError, match="The tendon length should not be normalized with a rigid tendon"):
        model.normalize_tendon_length(tendon_length=0.456)


def test_muscle_hill_model_rigid_tendon_compute_muscle_fiber_velocity_from_inverse():
    model = MuscleHillModels.RigidTendon(
        name="Dummy",
        maximal_force=123,
        optimal_length=0.123,
        tendon_slack_length=0.123,
        maximal_velocity=5.0,
    )

    with pytest.raises(
        RuntimeError, match="The inverse of muscle fiber velocity should not be computed with a rigid tendon"
    ):
        model.compute_muscle_fiber_velocity_from_inverse(
            activation=None, muscle_fiber_length=None, muscle_fiber_velocity=None, tendon_length=None
        )


def test_muscle_hill_model_rigid_tendon_compute_tendon_length():
    tendon_slack_length = 0.123
    model = MuscleHillModels.RigidTendon(
        name="Dummy",
        maximal_force=123,
        optimal_length=0.123,
        tendon_slack_length=tendon_slack_length,
        maximal_velocity=5.0,
    )

    assert_almost_equal(
        model.compute_tendon_length(muscle_tendon_length=0.0, muscle_fiber_length=0.0), tendon_slack_length
    )


def test_muscle_hill_model_rigid_tendon_compute_muscle_force():
    model = MuscleHillModels.RigidTendon(
        name="Dummy", maximal_force=123, optimal_length=0.123, tendon_slack_length=0.123, maximal_velocity=5.0
    )

    # Test exact values
    assert_almost_equal(
        model.compute_muscle_force(activation=0.123, muscle_fiber_length=0.09, muscle_fiber_velocity=5),
        18.382289948419388,
    )

    # Test values based on qualitative behavior
    assert 2 * model.compute_muscle_force(0.123, 0.09, 5) < model.compute_muscle_force(0.123 * 2, 0.09, 5)
    assert 2 * model.compute_muscle_force(0.123, 0.09, 5) < model.compute_muscle_force(0.123, 0.09 * 2, 5)


def test_muscle_hill_model_rigid_tendon_compute_tendon_force():
    tendon_slack_length = 0.123
    model = MuscleHillModels.RigidTendon(
        name="Dummy",
        maximal_force=123,
        optimal_length=0.123,
        tendon_slack_length=tendon_slack_length,
        maximal_velocity=5.0,
    )

    # Test exact value
    assert_almost_equal(model.compute_tendon_force(tendon_length=0.09), 0.0)

    # Test value based on qualitative behavior
    assert model.compute_tendon_force(tendon_slack_length / 2) == 0.0
    assert model.compute_tendon_force(tendon_slack_length) == 0.0
    assert model.compute_tendon_force(tendon_slack_length * 2) == 0.0


def test_muscle_hill_model_rigid_tendon_wrong_constructor():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "MuscleHillModelRigidTendon.__init__() missing 5 required positional arguments: "
            "'name', 'maximal_force', 'optimal_length', 'tendon_slack_length', and 'maximal_velocity'"
        ),
    ):
        MuscleHillModels.RigidTendon()
