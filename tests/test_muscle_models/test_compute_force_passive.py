from musculotendon_ocp import ComputeForcePassiveMethods
from numpy.testing import assert_almost_equal


def test_compute_force_passive_methods():
    assert len(ComputeForcePassiveMethods) == 2

    hill_type = ComputeForcePassiveMethods.HillType()
    assert type(hill_type) == ComputeForcePassiveMethods.HillType.value

    always_positive_hill_type = ComputeForcePassiveMethods.AlwaysPositiveHillType()
    assert type(always_positive_hill_type) == ComputeForcePassiveMethods.AlwaysPositiveHillType.value


def test_compute_force_passive_hill_type():

    force_passive_model = ComputeForcePassiveMethods.HillType()

    assert force_passive_model.kpe == 4.0
    assert force_passive_model.e0 == 0.6

    assert_almost_equal(force_passive_model(normalized_muscle_fiber_length=0.0), -0.018633616376331333)
    assert_almost_equal(force_passive_model(normalized_muscle_fiber_length=0.5), -0.01799177781427948)
    assert_almost_equal(force_passive_model(normalized_muscle_fiber_length=1.0), 0.000000000000000)
    assert_almost_equal(force_passive_model(normalized_muscle_fiber_length=1.5), 0.5043387668755398)


def test_compute_force_passive_always_positive_hill_type():

    force_passive_model = ComputeForcePassiveMethods.AlwaysPositiveHillType()

    assert force_passive_model.kpe == 4.0
    assert force_passive_model.e0 == 0.6

    offset = force_passive_model.offset
    assert offset == -0.018633616376331333

    # Test exact values
    assert_almost_equal(force_passive_model(normalized_muscle_fiber_length=0.0), 0.0)
    assert_almost_equal(force_passive_model(normalized_muscle_fiber_length=0.5), -0.01799177781427948 - offset)
    assert_almost_equal(force_passive_model(normalized_muscle_fiber_length=1.0), 0.000000000000000 - offset)
    assert_almost_equal(force_passive_model(normalized_muscle_fiber_length=1.5), 0.5043387668755398 - offset)

    # Test values based on qualitative behavior (increasing exponential function)
    assert force_passive_model(0.0) < force_passive_model(0.5)
    assert force_passive_model(0.5) < force_passive_model(1.0)
    assert force_passive_model(1.0) < force_passive_model(1.5)