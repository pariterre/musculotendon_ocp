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
    tendon_slack_length = 0.123
    maximal_velocity = 0.456
    force_passive = ForceVelocityHillType()
    force_active = ForceActiveHillType()
    force_damping = ForceDampingConstant()
    force_velocity = ForceVelocityHillType()

    model = MuscleModelHillRigidTendon(
        name=name,
        maximal_force=maximal_force,
        optimal_length=optimal_length,
        tendon_slack_length=tendon_slack_length,
        maximal_velocity=maximal_velocity,
        force_passive=force_passive,
        force_active=force_active,
        force_damping=force_damping,
        force_velocity=force_velocity,
    )

    assert model.name == name
    assert model.maximal_force == maximal_force
    assert model.optimal_length == optimal_length
    assert model.tendon_slack_length == tendon_slack_length
    assert model.maximal_velocity == maximal_velocity
    assert id(model._force_passive) == id(force_passive)
    assert id(model._force_active) == id(force_active)
    assert id(model._force_damping) == id(force_damping)
    assert id(model._force_velocity) == id(force_velocity)

    model_default = MuscleModelHillRigidTendon(
        name=name,
        maximal_force=maximal_force,
        optimal_length=optimal_length,
        tendon_slack_length=tendon_slack_length,
        maximal_velocity=maximal_velocity,
    )
    assert model_default._force_passive.__dict__ == ForcePassiveHillType().__dict__
    assert model_default._force_active.__dict__ == ForceActiveHillType().__dict__
    assert model_default._force_damping.__dict__ == ForceDampingConstant().__dict__
    assert model_default._force_velocity.__dict__ == ForceVelocityHillType().__dict__


def test_muscle_model_hill_rigid_tendon_normalize_muscle_fiber_length():
    optimal_length = 0.123
    model = MuscleModelHillRigidTendon(
        name="Dummy", maximal_force=123, optimal_length=optimal_length, tendon_slack_length=0.123, maximal_velocity=5.0
    )

    fiber_length = 0.456
    assert_almost_equal(model.normalize_muscle_fiber_length(fiber_length), fiber_length / optimal_length)


def test_muscle_model_hill_rigid_tendon_normalize_muscle_fiber_velocity():
    maximal_velocity = 0.456
    model = MuscleModelHillRigidTendon(
        name="Dummy",
        maximal_force=123,
        optimal_length=0.123,
        maximal_velocity=maximal_velocity,
        tendon_slack_length=0.123,
    )

    fiber_velocity = 0.789
    assert_almost_equal(model.normalize_muscle_fiber_velocity(fiber_velocity), fiber_velocity / maximal_velocity)


def test_muscle_model_hill_rigid_tendon_normalize_tendon_length():
    model = MuscleModelHillRigidTendon(
        name="Dummy", maximal_force=123, optimal_length=0.123, tendon_slack_length=0.123, maximal_velocity=5.0
    )

    with pytest.raises(RuntimeError, match="The tendon length should not be normalized with a rigid tendon"):
        model.normalize_tendon_length(tendon_length=0.456)


def test_muscle_model_hill_rigid_tendon_compute_tendon_length():
    tendon_slack_length = 0.123
    model = MuscleModelHillRigidTendon(
        name="Dummy",
        maximal_force=123,
        optimal_length=0.123,
        tendon_slack_length=tendon_slack_length,
        maximal_velocity=5.0,
    )

    assert_almost_equal(
        model.compute_tendon_length(muscle_tendon_length=0.0, muscle_fiber_length=0.0), tendon_slack_length
    )


def test_muscle_model_hill_rigid_tendon_compute_muscle_force():
    model = MuscleModelHillRigidTendon(
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


def test_muscle_model_hill_rigid_tendon_compute_tendon_force():
    tendon_slack_length = 0.123
    model = MuscleModelHillRigidTendon(
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


def test_muscle_model_hill_rigid_tendon_wrong_constructor():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "MuscleModelHillRigidTendon.__init__() missing 5 required positional arguments: "
            "'name', 'maximal_force', 'optimal_length', 'tendon_slack_length', and 'maximal_velocity'"
        ),
    ):
        MuscleModelHillRigidTendon()
