from musculotendon_ocp import ForceDampingConstant, ForceDampingLinear
from numpy.testing import assert_almost_equal
import pytest


@pytest.mark.parametrize("factor", [0.0, 1.0, 2.0])
def test_force_damping_constant(factor):
    force_damping_model = ForceDampingConstant(factor)

    assert_almost_equal(force_damping_model(0), factor)
    assert_almost_equal(force_damping_model(1), factor)
    assert_almost_equal(force_damping_model(2), factor)


@pytest.mark.parametrize("factor", [0.0, 1.0, 2.0])
def test_force_damping_linear(factor):
    force_damping_model = ForceDampingLinear(factor)

    assert_almost_equal(force_damping_model(normalized_muscle_velocity=0.0), 0)
    assert_almost_equal(force_damping_model(normalized_muscle_velocity=1.0), factor)
    assert_almost_equal(force_damping_model(normalized_muscle_velocity=2.0), 2 * factor)
    assert_almost_equal(force_damping_model(normalized_muscle_velocity=-1.0), -factor)
    assert_almost_equal(force_damping_model(normalized_muscle_velocity=-2.0), -2 * factor)
    assert_almost_equal(force_damping_model(normalized_muscle_velocity=0.5), 0.5 * factor)
    assert_almost_equal(force_damping_model(normalized_muscle_velocity=-0.5), -0.5 * factor)
    assert_almost_equal(force_damping_model(normalized_muscle_velocity=1.5), 1.5 * factor)
    assert_almost_equal(force_damping_model(normalized_muscle_velocity=-1.5), -1.5 * factor)
