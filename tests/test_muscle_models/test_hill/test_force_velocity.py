from musculotendon_ocp import ForceVelocityHillType


def test_force_velocity_hill_type():
    force_velocity_model = ForceVelocityHillType()

    assert force_velocity_model.d1 == -0.318
    assert force_velocity_model.d2 == -8.149
    assert force_velocity_model.d3 == -0.374
    assert force_velocity_model.d4 == 0.886

    # Test exact values
    assert force_velocity_model(normalized_muscle_velocity=0.0) == 1.002320622548512  # Isometric contraction

    assert force_velocity_model(normalized_muscle_velocity=0.5) == 1.5850003902837804  # Slow eccentric contraction
    assert force_velocity_model(normalized_muscle_velocity=1.0) == 1.7889099602998804  # Fast eccentric contraction

    assert force_velocity_model(normalized_muscle_velocity=-0.5) == 0.2438336294197121  # Slow concentric contraction
    assert force_velocity_model(normalized_muscle_velocity=-1.0) == 0.012081678112282557  # Fast concentric contraction
    assert force_velocity_model(normalized_muscle_velocity=-2.0) == -0.21490297384011525  # Supra concentric contraction

    # Test values based on qualitative behavior (increasing S-shaped function)
    assert force_velocity_model(0.0) < force_velocity_model(0.5)
    assert force_velocity_model(-0.5) < force_velocity_model(0.0)
    assert force_velocity_model(-2.0) < 0.0
