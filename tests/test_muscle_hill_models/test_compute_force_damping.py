from musculotendon_ocp import ComputeForceDampingMethods
from numpy.testing import assert_almost_equal
import pytest


def test_compute_force_damping_methods():
    assert len(ComputeForceDampingMethods) == 2

    constant = ComputeForceDampingMethods.Constant()
    assert type(constant) == ComputeForceDampingMethods.Constant.value

    linear = ComputeForceDampingMethods.Linear()
    assert type(linear) == ComputeForceDampingMethods.Linear.value

    with pytest.raises(ValueError, match="Cannot deserialize Unknown as ComputeForceDampingMethods"):
        ComputeForceDampingMethods.deserialize({"method": "Unknown"})


@pytest.mark.parametrize("factor", [0.0, 1.0, 2.0])
def test_compute_force_damping_constant(factor):
    force_damping_model = ComputeForceDampingMethods.Constant(factor)

    assert_almost_equal(force_damping_model(0), factor)
    assert_almost_equal(force_damping_model(1), factor)
    assert_almost_equal(force_damping_model(2), factor)

    # Test serialization
    serialized = force_damping_model.serialize()
    assert serialized == {"method": "ComputeForceDampingConstant", "factor": factor}
    deserialized = ComputeForceDampingMethods.deserialize(serialized)
    assert type(deserialized) == ComputeForceDampingMethods.Constant.value
    assert deserialized.factor == factor

    with pytest.raises(ValueError, match="Cannot deserialize Unknown as ComputeForceDampingConstant"):
        ComputeForceDampingMethods.Constant.value.deserialize({"method": "Unknown"})


@pytest.mark.parametrize("factor", [0.0, 1.0, 2.0])
def test_compute_force_damping_linear(factor):
    force_damping_model = ComputeForceDampingMethods.Linear(factor)

    assert_almost_equal(force_damping_model(normalized_muscle_fiber_velocity=0.0), 0)
    assert_almost_equal(force_damping_model(normalized_muscle_fiber_velocity=1.0), factor)
    assert_almost_equal(force_damping_model(normalized_muscle_fiber_velocity=2.0), 2 * factor)
    assert_almost_equal(force_damping_model(normalized_muscle_fiber_velocity=-1.0), -factor)
    assert_almost_equal(force_damping_model(normalized_muscle_fiber_velocity=-2.0), -2 * factor)
    assert_almost_equal(force_damping_model(normalized_muscle_fiber_velocity=0.5), 0.5 * factor)
    assert_almost_equal(force_damping_model(normalized_muscle_fiber_velocity=-0.5), -0.5 * factor)
    assert_almost_equal(force_damping_model(normalized_muscle_fiber_velocity=1.5), 1.5 * factor)
    assert_almost_equal(force_damping_model(normalized_muscle_fiber_velocity=-1.5), -1.5 * factor)

    # Test serialization
    serialized = force_damping_model.serialize()
    assert serialized == {"method": "ComputeForceDampingLinear", "factor": factor}
    deserialized = ComputeForceDampingMethods.deserialize(serialized)
    assert type(deserialized) == ComputeForceDampingMethods.Linear.value
    assert deserialized.factor == factor

    with pytest.raises(ValueError, match="Cannot deserialize Unknown as ComputeForceDampingLinear"):
        ComputeForceDampingMethods.Linear.value.deserialize({"method": "Unknown"})
