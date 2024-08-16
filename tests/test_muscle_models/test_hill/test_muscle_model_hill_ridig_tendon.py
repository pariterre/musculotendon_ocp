import re

from musculotendon_ocp import (
    MuscleModelHillRigidTendon,
    ForcePassiveHillType,
    ForceActiveHillType,
    ForceDampingConstant,
    ForceVelocityHillType,
)
from numpy.testing import assert_almost_equal
import pytest


def test_muscle_model_hill_rigid_tendon():
    name = "Dummy"
    maximal_force = 123
    optimal_length = 0.123
    maximal_velocity = 0.456
    force_passive = ForceVelocityHillType()
    force_active = ForceActiveHillType()
    force_damping = ForceDampingConstant()
    force_velocity = ForceVelocityHillType()

    model = MuscleModelHillRigidTendon(
        name=name,
        maximal_force=maximal_force,
        optimal_length=optimal_length,
        maximal_velocity=maximal_velocity,
        force_passive=force_passive,
        force_active=force_active,
        force_damping=force_damping,
        force_velocity=force_velocity,
    )

    assert model.name == name
    assert model.maximal_force == maximal_force
    assert model.optimal_length == optimal_length
    assert model.maximal_velocity == maximal_velocity
    assert id(model.force_passive) == id(force_passive)
    assert id(model.force_active) == id(force_active)
    assert id(model.force_damping) == id(force_damping)
    assert id(model.force_velocity) == id(force_velocity)

    model_default = MuscleModelHillRigidTendon(name=name, maximal_force=maximal_force, optimal_length=optimal_length)
    assert model_default.maximal_velocity == 5.0
    assert model_default.force_passive.__dict__ == ForcePassiveHillType().__dict__
    assert model_default.force_active.__dict__ == ForceActiveHillType().__dict__
    assert model_default.force_damping.__dict__ == ForceDampingConstant().__dict__
    assert model_default.force_velocity.__dict__ == ForceVelocityHillType().__dict__


def test_muscle_model_hill_rigid_tendon_normalize_muscle_length():
    optimal_length = 0.123
    model = MuscleModelHillRigidTendon(name="Dummy", maximal_force=123, optimal_length=optimal_length)

    fiber_length = 0.456
    assert_almost_equal(model.normalize_muscle_length(fiber_length), fiber_length / optimal_length)


def test_muscle_model_hill_rigid_tendon_normalize_muscle_velocity():
    maximal_velocity = 0.456
    model = MuscleModelHillRigidTendon(
        name="Dummy", maximal_force=123, optimal_length=0.123, maximal_velocity=maximal_velocity
    )

    fiber_velocity = 0.789
    assert_almost_equal(model.normalize_muscle_velocity(fiber_velocity), fiber_velocity / maximal_velocity)


def test_muscle_model_hill_rigid_tendon_normalize_tendon_length():
    model = MuscleModelHillRigidTendon(name="Dummy", maximal_force=123, optimal_length=0.123)

    tendon_length = 0.456
    with pytest.raises(RuntimeError, match="The tendon length should not be normalized for this muscle model"):
        model.normalize_tendon_length(tendon_length)


def test_muscle_model_hill_rigid_tendon_compute_muscle_force():
    model = MuscleModelHillRigidTendon(name="Dummy", maximal_force=123, optimal_length=0.123)

    # Test exact values
    assert_almost_equal(
        model.compute_muscle_force(activation=0.123, muscle_fiber_length=0.09, muscle_fiber_velocity=5),
        18.382289948419388,
    )

    # Test values based on qualitative behavior
    assert 2 * model.compute_muscle_force(0.123, 0.09, 5) < model.compute_muscle_force(0.123 * 2, 0.09, 5)
    assert 2 * model.compute_muscle_force(0.123, 0.09, 5) < model.compute_muscle_force(0.123, 0.09 * 2, 5)


def test_muscle_model_hill_rigid_tendon_compute_tendon_force():
    model = MuscleModelHillRigidTendon(name="Dummy", maximal_force=123, optimal_length=0.123)

    # Test exact values
    assert_almost_equal(model.compute_tendon_force(tendon_length=0.09), 0.0)

    # Test values based on qualitative behavior
    assert_almost_equal(model.compute_tendon_force(0.09), 0.0)
    assert_almost_equal(model.compute_tendon_force(0.09 * 2), 0.0)


def test_muscle_model_hill_rigid_tendon_checking_inputs():
    with pytest.raises(ValueError, match="The maximal force must be positive"):
        MuscleModelHillRigidTendon(name="Dummy", maximal_force=-123, optimal_length=0.123)

    with pytest.raises(ValueError, match="The optimal length must be positive"):
        MuscleModelHillRigidTendon(name="Dummy", maximal_force=123, optimal_length=-0.123)

    with pytest.raises(ValueError, match="The maximal velocity must be positive"):
        MuscleModelHillRigidTendon(name="Dummy", maximal_force=123, optimal_length=0.123, maximal_velocity=-0.123)

    with pytest.raises(
        TypeError,
        match=re.escape(
            "MuscleModelHillRigidTendon.__init__() missing 3 required positional arguments: "
            "'name', 'maximal_force', and 'optimal_length'"
        ),
    ):
        MuscleModelHillRigidTendon()
