from musculotendon_ocp import ComputePennationAngleMethods
from numpy.testing import assert_almost_equal
import pytest


def test_compute_pennation_angle_methods():
    assert len(ComputePennationAngleMethods) == 2

    constant = ComputePennationAngleMethods.Constant()
    assert type(constant) == ComputePennationAngleMethods.Constant.value

    wrt_muscle_fiber_length = ComputePennationAngleMethods.WrtMuscleFiberLength()
    assert type(wrt_muscle_fiber_length) == ComputePennationAngleMethods.WrtMuscleFiberLength.value

    with pytest.raises(ValueError, match="Cannot deserialize Unknown as ComputePennationAngleMethods"):
        ComputePennationAngleMethods.deserialize({"method": "Unknown"})


def test_compute_pennation_angle_constant():
    with pytest.raises(ValueError, match="The pennation angle must be positive"):
        ComputePennationAngleMethods.Constant(pennation_angle=-0.1)

    pennation_angle_model = ComputePennationAngleMethods.Constant(pennation_angle=0.1)

    assert pennation_angle_model.pennation_angle == 0.1

    # Test exact values
    assert_almost_equal(pennation_angle_model(1.1), 0.1)
    assert_almost_equal(pennation_angle_model(1.2), 0.1)

    assert_almost_equal(pennation_angle_model.apply(1.1, 1.0), 0.9950041652780258)
    assert_almost_equal(pennation_angle_model.apply(1.1, 0.5), 0.4975020826390129)
    assert_almost_equal(pennation_angle_model.apply(1.1, 0.0), 0.0)
    assert_almost_equal(pennation_angle_model.apply(1.1, -0.5), -0.4975020826390129)
    assert_almost_equal(pennation_angle_model.apply(1.1, -1.0), -0.9950041652780258)

    assert_almost_equal(pennation_angle_model.apply(1.2, 1.0), 0.9950041652780258)
    assert_almost_equal(pennation_angle_model.apply(1.2, 0.5), 0.4975020826390129)
    assert_almost_equal(pennation_angle_model.apply(1.2, 0.0), 0.0)
    assert_almost_equal(pennation_angle_model.apply(1.2, -0.5), -0.4975020826390129)
    assert_almost_equal(pennation_angle_model.apply(1.2, -1.0), -0.9950041652780258)

    # Test values based on qualitative behavior (increasing exponential function)
    assert_almost_equal(pennation_angle_model.apply(1.0, 0.0), 0.0)
    assert_almost_equal(pennation_angle_model.apply(1.1, 0.5), pennation_angle_model.apply(1.2, 0.5))

    assert_almost_equal(pennation_angle_model.remove(1.1, 1.0), 1.0050209184004553)
    assert_almost_equal(pennation_angle_model.remove(1.1, 0.5), 0.5025104592002276)
    assert_almost_equal(pennation_angle_model.remove(1.1, 0.0), 0.0)
    assert_almost_equal(pennation_angle_model.remove(1.1, -0.5), -0.5025104592002276)
    assert_almost_equal(pennation_angle_model.remove(1.1, -1.0), -1.0050209184004553)

    assert_almost_equal(pennation_angle_model.remove(1.2, 1.0), 1.0050209184004553)
    assert_almost_equal(pennation_angle_model.remove(1.2, 0.5), 0.5025104592002276)
    assert_almost_equal(pennation_angle_model.remove(1.2, 0.0), 0.0)
    assert_almost_equal(pennation_angle_model.remove(1.2, -0.5), -0.5025104592002276)
    assert_almost_equal(pennation_angle_model.remove(1.2, -1.0), -1.0050209184004553)

    # Test values based on qualitative behavioremove (increasing exponential function)
    assert_almost_equal(pennation_angle_model.remove(1.0, 0.0), 0.0)
    assert pennation_angle_model.apply(1.1, 0.5) == pennation_angle_model.apply(1.2, 0.5)

    # Test serialization
    serialized = pennation_angle_model.serialize()
    assert serialized == {"method": "ComputePennationAngleConstant", "pennation_angle": 0.1}
    deserialized = ComputePennationAngleMethods.deserialize(serialized)
    assert type(deserialized) == ComputePennationAngleMethods.Constant.value
    assert deserialized.pennation_angle == 0.1

    with pytest.raises(ValueError, match="Cannot deserialize Unknown as ComputePennationAngleConstant"):
        ComputePennationAngleMethods.Constant.value.deserialize({"method": "Unknown"})


def test_compute_pennation_angle_wrt_muscle_fiber_length():
    with pytest.raises(ValueError, match="The optimal pennation angle must be positive"):
        ComputePennationAngleMethods.WrtMuscleFiberLength(optimal_pennation_angle=-0.1, optimal_muscle_fiber_length=1.1)

    pennation_angle_model = ComputePennationAngleMethods.WrtMuscleFiberLength(
        optimal_pennation_angle=0.1, optimal_muscle_fiber_length=1.1
    )

    assert_almost_equal(pennation_angle_model.optimal_pennation_angle, 0.1)
    assert_almost_equal(pennation_angle_model.optimal_muscle_fiber_length, 1.1)

    # Test exact values
    assert_almost_equal(pennation_angle_model(1.1), 0.1)
    assert_almost_equal(pennation_angle_model(1.2), 0.09164218434601921)

    assert_almost_equal(pennation_angle_model.apply(1.1, 1.0), 0.9950041652780258)
    assert_almost_equal(pennation_angle_model.apply(1.1, 0.5), 0.4975020826390129)
    assert_almost_equal(pennation_angle_model.apply(1.1, 0.0), 0.0)
    assert_almost_equal(pennation_angle_model.apply(1.1, -0.5), -0.4975020826390129)
    assert_almost_equal(pennation_angle_model.apply(1.1, -1.0), -0.9950041652780258)

    assert_almost_equal(pennation_angle_model.apply(1.2, 1.0), 0.9958037930046592)
    assert_almost_equal(pennation_angle_model.apply(1.2, 0.5), 0.4979018965023296)
    assert_almost_equal(pennation_angle_model.apply(1.2, 0.0), 0.0)
    assert_almost_equal(pennation_angle_model.apply(1.2, -0.5), -0.4979018965023296)
    assert_almost_equal(pennation_angle_model.apply(1.2, -1.0), -0.9958037930046592)

    # Test values based on qualitative behavior (increasing exponential function)
    assert pennation_angle_model.apply(1.0, 0.0) == 0.0
    assert pennation_angle_model.apply(1.1, 0.5) < pennation_angle_model.apply(1.2, 0.5)
    assert pennation_angle_model.apply(1.1, 0.5) == ComputePennationAngleMethods.Constant(pennation_angle=0.1).apply(
        1.1, 0.5
    )
    assert pennation_angle_model.apply(1.2, 0.5) != ComputePennationAngleMethods.Constant(pennation_angle=0.1).apply(
        1.2, 0.5
    )

    assert_almost_equal(pennation_angle_model.remove(1.1, 1.0), 1.0050209184004553)
    assert_almost_equal(pennation_angle_model.remove(1.1, 0.5), 0.5025104592002276)
    assert_almost_equal(pennation_angle_model.remove(1.1, 0.0), 0.0)
    assert_almost_equal(pennation_angle_model.remove(1.1, -0.5), -0.5025104592002276)
    assert_almost_equal(pennation_angle_model.remove(1.1, -1.0), -1.0050209184004553)

    assert_almost_equal(pennation_angle_model.remove(1.2, 1.0), 1.0042138893472976)
    assert_almost_equal(pennation_angle_model.remove(1.2, 0.5), 0.5021069446736488)
    assert_almost_equal(pennation_angle_model.remove(1.2, 0.0), 0.0)
    assert_almost_equal(pennation_angle_model.remove(1.2, -0.5), -0.5021069446736488)
    assert_almost_equal(pennation_angle_model.remove(1.2, -1.0), -1.0042138893472976)

    # Test values based on qualitative behavior (increasing exponential function)
    assert pennation_angle_model.remove(1.0, 0.0) == 0.0
    assert pennation_angle_model.remove(1.1, 0.5) > pennation_angle_model.remove(1.2, 0.5)
    assert pennation_angle_model.remove(1.1, 0.5) == ComputePennationAngleMethods.Constant(pennation_angle=0.1).remove(
        1.1, 0.5
    )
    assert pennation_angle_model.remove(1.2, 0.5) != ComputePennationAngleMethods.Constant(pennation_angle=0.1).remove(
        1.2, 0.5
    )

    # Test serialization
    serialized = pennation_angle_model.serialize()
    assert serialized == {
        "method": "ComputePennationAngleWrtMuscleFiberLength",
        "optimal_pennation_angle": 0.1,
        "optimal_muscle_fiber_length": 1.1,
    }
    deserialized = ComputePennationAngleMethods.deserialize(serialized)
    assert type(deserialized) == ComputePennationAngleMethods.WrtMuscleFiberLength.value
    assert_almost_equal(deserialized.optimal_pennation_angle, 0.1)
    assert_almost_equal(deserialized.optimal_muscle_fiber_length, 1.1)

    with pytest.raises(ValueError, match="Cannot deserialize Unknown as ComputePennationAngleWrtMuscleFiberLength"):
        ComputePennationAngleMethods.WrtMuscleFiberLength.value.deserialize({"method": "Unknown"})
