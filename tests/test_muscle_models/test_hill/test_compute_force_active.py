from musculotendon_ocp import ComputeForceActiveHillType
from numpy.testing import assert_almost_equal


def test_compute_force_active_hill_type():

    force_active_model = ComputeForceActiveHillType()

    assert force_active_model.b11 == 0.814483478343008
    assert force_active_model.b21 == 1.055033428970575
    assert force_active_model.b31 == 0.162384573599574
    assert force_active_model.b41 == 0.063303448465465
    assert force_active_model.b12 == 0.433004984392647
    assert force_active_model.b22 == 0.716775413397760
    assert force_active_model.b32 == -0.029947116970696
    assert force_active_model.b42 == 0.200356847296188
    assert force_active_model.b13 == 0.100
    assert force_active_model.b23 == 1.000
    assert force_active_model.b33 == 0.354
    assert force_active_model.b43 == 0.000

    # Test exact values
    assert_almost_equal(force_active_model(normalized_muscle_fiber_length=0.5), 0.05419527682606315)
    assert_almost_equal(force_active_model(normalized_muscle_fiber_length=1.0), 0.9994334614323869)
    assert_almost_equal(force_active_model(normalized_muscle_fiber_length=1.5), 0.22611061742850164)

    # Test values based on qualitative behavior (inverted U-shaped function)
    assert force_active_model(0.5) < force_active_model(1.0)
    assert force_active_model(1.0) > force_active_model(1.5)
