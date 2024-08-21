import re

from musculotendon_ocp import (
    MuscleModelHillFlexibleTendon,
    ForcePassiveHillType,
    ForceActiveHillType,
    ForceDampingConstant,
    ForceVelocityHillType,
)
from numpy.testing import assert_almost_equal
import pytest


def test_muscle_model_hill_flexible_tendon():
    name = "Dummy"
    maximal_force = 123
    optimal_length = 0.123
    tendon_slack_length = 0.123
    maximal_velocity = 0.456
    c1 = 0.4
    c2 = 1.5
    c3 = 0.3
    kt = 40.0
    force_passive = ForceVelocityHillType()
    force_active = ForceActiveHillType()
    force_damping = ForceDampingConstant()
    force_velocity = ForceVelocityHillType()

    model = MuscleModelHillFlexibleTendon(
        name=name,
        maximal_force=maximal_force,
        optimal_length=optimal_length,
        tendon_slack_length=tendon_slack_length,
        maximal_velocity=maximal_velocity,
        c1=c1,
        c2=c2,
        c3=c3,
        kt=kt,
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
    assert model.c1 == c1
    assert model.c2 == c2
    assert model.c3 == c3
    assert model.kt == kt
    assert id(model._force_passive) == id(force_passive)
    assert id(model._force_active) == id(force_active)
    assert id(model._force_damping) == id(force_damping)
    assert id(model._force_velocity) == id(force_velocity)

    model_default = MuscleModelHillFlexibleTendon(
        name=name,
        maximal_force=maximal_force,
        optimal_length=optimal_length,
        tendon_slack_length=tendon_slack_length,
        maximal_velocity=5.0,
    )
    assert model_default.maximal_velocity == 5.0
    assert model_default.c1 == 0.2
    assert model_default.c2 == 0.995
    assert model_default.c3 == 0.250
    assert model_default.kt == 35.0
    assert model_default._force_passive.__dict__ == ForcePassiveHillType().__dict__
    assert model_default._force_active.__dict__ == ForceActiveHillType().__dict__
    assert model_default._force_damping.__dict__ == ForceDampingConstant().__dict__
    assert model_default._force_velocity.__dict__ == ForceVelocityHillType().__dict__


def test_muscle_model_hill_flexible_tendon_normalize_tendon_length():
    model = MuscleModelHillFlexibleTendon(
        name="Dummy", maximal_force=123, optimal_length=0.123, tendon_slack_length=0.123, maximal_velocity=5.0
    )

    assert_almost_equal(model.normalize_tendon_length(tendon_length=0.246), 2.0)


def test_muscle_model_hill_flexible_tendon_compute_tendon_length():
    model = MuscleModelHillFlexibleTendon(
        name="Dummy", maximal_force=123, optimal_length=0.123, tendon_slack_length=0.123, maximal_velocity=5.0
    )

    assert_almost_equal(model.compute_tendon_length(muscle_tendon_length=0.246, muscle_fiber_length=0.123), 0.123)


def test_muscle_model_hill_flexible_tendon_compute_tendon_force():
    model = MuscleModelHillFlexibleTendon(
        name="Dummy", maximal_force=123, optimal_length=0.123, tendon_slack_length=0.123, maximal_velocity=5.0
    )

    # Test exact values
    assert_almost_equal(model.compute_tendon_force(tendon_length=0.15), 516.9924060762903)

    # Test values based on qualitative behavior
    assert_almost_equal(model.compute_tendon_force(0.123), 0.0)
    assert model.compute_tendon_force(0.123 / 2) < 0.0
    assert model.compute_tendon_force(0.123 * 2) > 0.0
