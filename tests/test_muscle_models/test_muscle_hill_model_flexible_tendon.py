from musculotendon_ocp import (
    MuscleHillModels,
    ComputeForcePassiveMethods,
    ComputeForceActiveMethods,
    ComputeForceDampingMethods,
    ComputeForceVelocityMethods,
)
from numpy.testing import assert_almost_equal


def test_muscle_hill_model_flexible_tendon():
    name = "Dummy"
    maximal_force = 123
    optimal_length = 0.123
    tendon_slack_length = 0.123
    maximal_velocity = 0.456
    c1 = 0.4
    c2 = 1.5
    c3 = 0.3
    kt = 40.0
    compute_force_passive = ComputeForceVelocityMethods.HillType()
    compute_force_active = ComputeForceActiveMethods.HillType()
    compute_force_damping = ComputeForceDampingMethods.Constant()
    compute_force_velocity = ComputeForceVelocityMethods.HillType()

    model = MuscleHillModels.FlexibleTendon(
        name=name,
        maximal_force=maximal_force,
        optimal_length=optimal_length,
        tendon_slack_length=tendon_slack_length,
        maximal_velocity=maximal_velocity,
        c1=c1,
        c2=c2,
        c3=c3,
        kt=kt,
        compute_force_passive=compute_force_passive,
        compute_force_active=compute_force_active,
        compute_force_damping=compute_force_damping,
        compute_force_velocity=compute_force_velocity,
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
    assert id(model.compute_force_passive) == id(compute_force_passive)
    assert id(model.compute_force_active) == id(compute_force_active)
    assert id(model.compute_force_damping) == id(compute_force_damping)
    assert id(model.compute_force_velocity) == id(compute_force_velocity)

    model_default = MuscleHillModels.FlexibleTendon(
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
    assert model_default.compute_force_passive.__dict__ == ComputeForcePassiveMethods.HillType().__dict__
    assert model_default.compute_force_active.__dict__ == ComputeForceActiveMethods.HillType().__dict__
    assert model_default.compute_force_damping.__dict__ == ComputeForceDampingMethods.Constant().__dict__
    assert model_default.compute_force_velocity.__dict__ == ComputeForceVelocityMethods.HillType().__dict__


def test_muscle_hill_model_flexible_tendon_normalize_tendon_length():
    model = MuscleHillModels.FlexibleTendon(
        name="Dummy", maximal_force=123, optimal_length=0.123, tendon_slack_length=0.123, maximal_velocity=5.0
    )

    tendon_length = 0.15
    normalized_tendon_length = model.normalize_tendon_length(tendon_length=tendon_length)
    assert_almost_equal(normalized_tendon_length, 1.2195121951219512)

    denormalized_tendon_length = model.denormalize_tendon_length(normalized_tendon_length)
    assert_almost_equal(denormalized_tendon_length, tendon_length)


def test_muscle_hill_model_flexible_tendon_compute_muscle_fiber_velocity_from_inverse():
    model = MuscleHillModels.FlexibleTendon(
        name="Dummy", maximal_force=123, optimal_length=0.123, tendon_slack_length=0.123, maximal_velocity=5.0
    )

    muscle_fiber_velocity = model.compute_muscle_fiber_velocity_from_inverse(
        activation=0.6, muscle_fiber_length=0.120, muscle_fiber_velocity=0.0, tendon_length=0.124
    )
    assert_almost_equal(muscle_fiber_velocity, -5.098821204647615)


def test_muscle_hill_model_flexible_tendon_compute_tendon_length():
    model = MuscleHillModels.FlexibleTendon(
        name="Dummy", maximal_force=123, optimal_length=0.123, tendon_slack_length=0.123, maximal_velocity=5.0
    )

    assert_almost_equal(model.compute_tendon_length(muscle_tendon_length=0.246, muscle_fiber_length=0.123), 0.123)


def test_muscle_hill_model_flexible_tendon_compute_tendon_force():
    model = MuscleHillModels.FlexibleTendon(
        name="Dummy", maximal_force=123, optimal_length=0.123, tendon_slack_length=0.123, maximal_velocity=5.0
    )

    # Test exact values
    assert_almost_equal(model.compute_tendon_force(tendon_length=0.15), 516.9806553196128)

    # Test values based on qualitative behavior
    assert model.compute_tendon_force(0.123) < 0.0
    assert model.compute_tendon_force(0.123 / 2) < 0.0
    assert model.compute_tendon_force(0.123 * 2) > 0.0


def test_muscle_hill_model_flexible_tendon_always_positive_compute_tendon_force():
    model = MuscleHillModels.FlexibleTendonAlwaysPositive(
        name="Dummy", maximal_force=123, optimal_length=0.123, tendon_slack_length=0.123, maximal_velocity=5.0
    )

    # Test exact values
    assert_almost_equal(model.compute_tendon_force(tendon_length=0.15), 516.9924060762903)

    # Test values based on qualitative behavior
    assert_almost_equal(model.compute_tendon_force(0.123), 0.0)
    assert model.compute_tendon_force(0.123 / 2) < 0.0
    assert model.compute_tendon_force(0.123 * 2) > 0.0
