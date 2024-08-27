from casadi import MX, jacobian, Function
from musculotendon_ocp import ComputeForceVelocityMethods
from numpy.testing import assert_almost_equal


def test_compute_force_velocity_methods():
    assert len(ComputeForceVelocityMethods) == 1

    hill_type = ComputeForceVelocityMethods.HillType()
    assert type(hill_type) == ComputeForceVelocityMethods.HillType.value


def test_compute_force_velocity_hill_type():
    force_velocity_model = ComputeForceVelocityMethods.HillType()

    assert force_velocity_model.d1 == -0.318
    assert force_velocity_model.d2 == -8.149
    assert force_velocity_model.d3 == -0.374
    assert force_velocity_model.d4 == 0.886

    # Test exact values
    # Isometric contraction
    assert_almost_equal(force_velocity_model(normalized_muscle_fiber_velocity=0.0), 1.002320622548512)

    # Slow and Fast eccentric contraction
    assert_almost_equal(force_velocity_model(normalized_muscle_fiber_velocity=0.5), 1.5850003902837804)
    assert_almost_equal(force_velocity_model(normalized_muscle_fiber_velocity=1.0), 1.7889099602998804)

    # Slow, Fast and Supra concentric contraction
    assert_almost_equal(force_velocity_model(normalized_muscle_fiber_velocity=-0.5), 0.2438336294197121)
    assert_almost_equal(force_velocity_model(normalized_muscle_fiber_velocity=-1.0), 0.012081678112282557)
    assert_almost_equal(force_velocity_model(normalized_muscle_fiber_velocity=-2.0), -0.21490297384011525)

    # Inverse function
    assert_almost_equal(force_velocity_model.inverse(-1.0), -23.14440612299026)
    assert_almost_equal(force_velocity_model.inverse(0.0), -1.0372450842799263)
    assert_almost_equal(force_velocity_model.inverse(1.0), -0.0009548832444486917)

    # Compute linear approximation coefficients
    mx = MX.sym("x", 1, 1)
    jaco = Function("jacobian", [mx], [jacobian(force_velocity_model(mx), mx)])
    assert_almost_equal(force_velocity_model.derivative(-1.0), float(jaco(-1.0)))
    assert_almost_equal(force_velocity_model.derivative(0.0), float(jaco(0.0)))
    assert_almost_equal(force_velocity_model.derivative(1.0), float(jaco(1.0)))

    # Test values based on qualitative behavior (increasing S-shaped function)
    assert force_velocity_model(0.0) < force_velocity_model(0.5)
    assert force_velocity_model(-0.5) < force_velocity_model(0.0)
    assert force_velocity_model(-2.0) < 0.0

    assert force_velocity_model.inverse(-1.0) < force_velocity_model.inverse(0.0)
    assert force_velocity_model.inverse(0.0) < force_velocity_model.inverse(1.0)
