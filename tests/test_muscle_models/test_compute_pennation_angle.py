from musculotendon_ocp import ComputePennationAngleConstant, ComputePennationAngleWrtMuscleFiberLength
from numpy.testing import assert_almost_equal
import pytest


def test_compute_pennation_angle_constant():
    with pytest.raises(ValueError, match="The pennation angle must be positive"):
        ComputePennationAngleConstant(pennation_angle=-0.1)

    pennation_angle_model = ComputePennationAngleConstant(pennation_angle=0.1)

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


def test_compute_pennation_angle_wrt_muscle_fiber_length():
    with pytest.raises(ValueError, match="The optimal pennation angle must be positive"):
        ComputePennationAngleWrtMuscleFiberLength(optimal_pennation_angle=-0.1, optimal_muscle_fiber_length=1.1)

    pennation_angle_model = ComputePennationAngleWrtMuscleFiberLength(
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
    assert pennation_angle_model.apply(1.1, 0.5) == ComputePennationAngleConstant(pennation_angle=0.1).apply(1.1, 0.5)
    assert pennation_angle_model.apply(1.2, 0.5) != ComputePennationAngleConstant(pennation_angle=0.1).apply(1.2, 0.5)
