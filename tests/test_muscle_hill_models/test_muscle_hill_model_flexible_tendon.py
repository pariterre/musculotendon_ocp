from musculotendon_ocp import (
    MuscleHillModels,
    ComputeForcePassiveMethods,
    ComputeForceActiveMethods,
    ComputeForceDampingMethods,
    ComputeForceVelocityMethods,
)
from numpy.testing import assert_almost_equal
import pytest


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
    compute_force_passive = ComputeForcePassiveMethods.HillType()
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

    # Test serialization
    serialized = model.serialize()
    assert serialized == {
        "method": "MuscleHillModelFlexibleTendon",
        "name": name,
        "label": name,
        "maximal_force": maximal_force,
        "optimal_length": optimal_length,
        "tendon_slack_length": tendon_slack_length,
        "maximal_velocity": maximal_velocity,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "kt": kt,
        "compute_force_passive": compute_force_passive.serialize(),
        "compute_force_active": compute_force_active.serialize(),
        "compute_force_damping": compute_force_damping.serialize(),
        "compute_force_velocity": compute_force_velocity.serialize(),
        "compute_pennation_angle": model_default.compute_pennation_angle.serialize(),
        "compute_muscle_fiber_length": model_default.compute_muscle_fiber_length.serialize(),
        "compute_muscle_fiber_velocity": model_default.compute_muscle_fiber_velocity.serialize(),
    }
    deserialized = MuscleHillModels.deserialize(serialized)
    assert type(deserialized) == MuscleHillModels.FlexibleTendon.value
    assert deserialized.name == name
    assert deserialized.maximal_force == maximal_force
    assert deserialized.optimal_length == optimal_length
    assert deserialized.tendon_slack_length == tendon_slack_length
    assert deserialized.maximal_velocity == maximal_velocity
    assert deserialized.c1 == c1
    assert deserialized.c2 == c2
    assert deserialized.c3 == c3
    assert deserialized.kt == kt
    assert type(deserialized.compute_force_passive) == type(model.compute_force_passive)
    assert deserialized.compute_force_passive.__dict__ == model.compute_force_passive.__dict__
    assert type(deserialized.compute_force_active) == type(model.compute_force_active)
    assert deserialized.compute_force_active.__dict__ == model.compute_force_active.__dict__
    assert type(deserialized.compute_force_damping) == type(model.compute_force_damping)
    assert deserialized.compute_force_damping.__dict__ == model.compute_force_damping.__dict__
    assert type(deserialized.compute_force_velocity) == type(model.compute_force_velocity)
    assert deserialized.compute_force_velocity.__dict__ == model.compute_force_velocity.__dict__

    with pytest.raises(ValueError, match="Cannot deserialize Unknown as MuscleHillModelFlexibleTendon"):
        MuscleHillModels.FlexibleTendon.value.deserialize({"method": "Unknown"})


def test_muscle_hill_model_flexible_tendon_normalize_tendon_length():
    model = MuscleHillModels.FlexibleTendon(
        name="Dummy", maximal_force=123, optimal_length=0.123, tendon_slack_length=0.123, maximal_velocity=5.0
    )

    tendon_length = 0.15
    normalized_tendon_length = model.normalize_tendon_length(tendon_length=tendon_length)
    assert_almost_equal(normalized_tendon_length, 1.2195121951219512)

    denormalized_tendon_length = model.denormalize_tendon_length(normalized_tendon_length)
    assert_almost_equal(denormalized_tendon_length, tendon_length)


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

    # Test serialization
    serialized = model.serialize()
    assert serialized == {
        "method": "MuscleHillModelFlexibleTendonAlwaysPositive",
        "name": "Dummy",
        "label": "Dummy",
        "maximal_force": 123,
        "optimal_length": 0.123,
        "tendon_slack_length": 0.123,
        "maximal_velocity": 5.0,
        "c1": 0.2,
        "c2": 0.995,
        "c3": 0.250,
        "kt": 35.0,
        "compute_force_passive": model.compute_force_passive.serialize(),
        "compute_force_active": model.compute_force_active.serialize(),
        "compute_force_damping": model.compute_force_damping.serialize(),
        "compute_force_velocity": model.compute_force_velocity.serialize(),
        "compute_pennation_angle": model.compute_pennation_angle.serialize(),
        "compute_muscle_fiber_length": model.compute_muscle_fiber_length.serialize(),
        "compute_muscle_fiber_velocity": model.compute_muscle_fiber_velocity.serialize(),
    }
    deserialized = MuscleHillModels.deserialize(serialized)
    assert type(deserialized) == MuscleHillModels.FlexibleTendonAlwaysPositive.value
    assert deserialized.name == "Dummy"
    assert deserialized.maximal_force == 123
    assert deserialized.optimal_length == 0.123
    assert deserialized.tendon_slack_length == 0.123
    assert deserialized.maximal_velocity == 5.0
    assert deserialized.c1 == 0.2
    assert deserialized.c2 == 0.995
    assert deserialized.c3 == 0.250
    assert deserialized.kt == 35.0
    assert type(deserialized.compute_force_passive) == type(model.compute_force_passive)
    assert deserialized.compute_force_passive.__dict__ == model.compute_force_passive.__dict__
    assert type(deserialized.compute_force_active) == type(model.compute_force_active)
    assert deserialized.compute_force_active.__dict__ == model.compute_force_active.__dict__
    assert type(deserialized.compute_force_damping) == type(model.compute_force_damping)
    assert deserialized.compute_force_damping.__dict__ == model.compute_force_damping.__dict__
    assert type(deserialized.compute_force_velocity) == type(model.compute_force_velocity)
    assert deserialized.compute_force_velocity.__dict__ == model.compute_force_velocity.__dict__
