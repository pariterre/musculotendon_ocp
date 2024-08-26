from typing import override

from casadi import MX
from musculotendon_ocp.muscle_hill_models.muscle_hill_model_abstract import MuscleHillModelAbstract
import numpy as np
import pytest


class DummyMuscleModelAbstract(MuscleHillModelAbstract):
    @override
    def normalize_muscle_fiber_length(self, muscle_fiber_length: MX) -> MX:
        pass

    def denormalize_muscle_fiber_length(self, normalized_muscle_fiber_length: MX) -> MX:
        pass

    @override
    def normalize_muscle_fiber_velocity(self, muscle_fiber_velocity: MX) -> MX:
        pass

    @override
    def denormalize_muscle_fiber_velocity(self, normalized_muscle_fiber_velocity: MX) -> MX:
        pass

    @override
    def normalize_tendon_length(self, tendon_length: MX) -> MX:
        pass

    @override
    def denormalize_tendon_length(self, normalized_tendon_length: MX) -> MX:
        pass

    @override
    def compute_muscle_force(self, activation: MX, muscle_fiber_length: MX, muscle_fiber_velocity: MX) -> MX:
        pass

    @override
    def compute_muscle_fiber_velocity_from_inverse(
        self, muscle_fiber_length: MX, normalized_muscle_fiber_velocity: MX
    ) -> MX:
        pass

    @override
    def compute_tendon_length(self, muscle_tendon_length: MX, muscle_fiber_length: MX) -> MX:
        pass

    @override
    def compute_tendon_force(self, tendon_length: MX) -> MX:
        pass


def test_muscle_hill_model_rigid_tendon_checking_inputs():
    with pytest.raises(ValueError, match="The maximal force must be positive"):
        DummyMuscleModelAbstract(
            name="Dummy",
            maximal_force=-123,
            optimal_length=0.123,
            tendon_slack_length=0.123,
            maximal_velocity=0.123,
            compute_force_passive=None,
            compute_force_active=None,
            compute_force_velocity=None,
            compute_force_damping=None,
            compute_pennation_angle=None,
            compute_muscle_fiber_length=None,
            compute_muscle_fiber_velocity=None,
        )

    with pytest.raises(ValueError, match="The optimal length must be positive"):
        DummyMuscleModelAbstract(
            name="Dummy",
            maximal_force=123,
            optimal_length=-0.123,
            tendon_slack_length=0.123,
            maximal_velocity=0.123,
            compute_force_passive=None,
            compute_force_active=None,
            compute_force_velocity=None,
            compute_force_damping=None,
            compute_pennation_angle=None,
            compute_muscle_fiber_length=None,
            compute_muscle_fiber_velocity=None,
        )

    with pytest.raises(ValueError, match="The tendon slack length must be positive"):
        DummyMuscleModelAbstract(
            name="Dummy",
            maximal_force=123,
            optimal_length=0.123,
            tendon_slack_length=-0.123,
            maximal_velocity=0.123,
            compute_force_passive=None,
            compute_force_active=None,
            compute_force_velocity=None,
            compute_force_damping=None,
            compute_pennation_angle=None,
            compute_muscle_fiber_length=None,
            compute_muscle_fiber_velocity=None,
        )

    with pytest.raises(ValueError, match="The maximal velocity must be positive"):
        DummyMuscleModelAbstract(
            name="Dummy",
            maximal_force=123,
            optimal_length=0.123,
            tendon_slack_length=0.123,
            maximal_velocity=-0.123,
            compute_force_passive=None,
            compute_force_active=None,
            compute_force_velocity=None,
            compute_force_damping=None,
            compute_pennation_angle=None,
            compute_muscle_fiber_length=None,
            compute_muscle_fiber_velocity=None,
        )


def test_get_mx_variables():
    model = DummyMuscleModelAbstract(
        name="Dummy",
        maximal_force=123,
        optimal_length=0.123,
        tendon_slack_length=0.123,
        maximal_velocity=0.123,
        compute_force_passive=None,
        compute_force_active=None,
        compute_force_velocity=None,
        compute_force_damping=None,
        compute_pennation_angle=None,
        compute_muscle_fiber_length=None,
        compute_muscle_fiber_velocity=None,
    )

    # Test the mx variables are returned and they are always the same (cached)
    assert isinstance(model.activation_mx, MX)
    assert id(model.activation_mx) == id(model.activation_mx)

    assert isinstance(model.muscle_fiber_length_mx, MX)
    assert id(model.muscle_fiber_length_mx) == id(model.muscle_fiber_length_mx)

    assert isinstance(model.muscle_fiber_velocity_mx, MX)
    assert id(model.muscle_fiber_velocity_mx) == id(model.muscle_fiber_velocity_mx)

    assert isinstance(model.tendon_length_mx, MX)
    assert id(model.tendon_length_mx) == id(model.tendon_length_mx)


def test_muscle_model_abstract_casadi_function_interface():
    model = DummyMuscleModelAbstract(
        name="Dummy",
        maximal_force=123,
        optimal_length=0.123,
        tendon_slack_length=0.123,
        maximal_velocity=0.123,
        compute_force_passive=None,
        compute_force_active=None,
        compute_force_velocity=None,
        compute_force_damping=None,
        compute_pennation_angle=None,
        compute_muscle_fiber_length=None,
        compute_muscle_fiber_velocity=None,
    )

    def my_small_function(activation: MX, muscle_fiber_length: MX, muscle_fiber_velocity: MX) -> MX:
        return activation + muscle_fiber_length + muscle_fiber_velocity

    def my_long_function(activation: MX, muscle_fiber_length: MX, muscle_fiber_velocity: MX, tendon_length: MX) -> MX:
        return my_small_function(activation, muscle_fiber_length, muscle_fiber_velocity) + tendon_length

    # Only certain inputs are allowed
    with pytest.raises(
        ValueError,
        match="Expected 'activation', 'muscle_fiber_length', 'muscle_fiber_velocity' or 'tendon_length', got dummy",
    ):
        model.to_casadi_function(my_small_function, "dummy")

    long_func = model.to_casadi_function(
        my_long_function, "activation", "muscle_fiber_length", "muscle_fiber_velocity", "tendon_length"
    )
    np.testing.assert_almost_equal(
        float(long_func(activation=1, muscle_fiber_length=2, muscle_fiber_velocity=3, tendon_length=2)["output"]), 8.0
    )

    np.testing.assert_almost_equal(
        float(
            model.function_to_dm(
                my_long_function, activation=1, muscle_fiber_length=2, muscle_fiber_velocity=3, tendon_length=2
            )
        ),
        8.0,
    )

    small_func = model.to_casadi_function(
        my_small_function, "activation", "muscle_fiber_length", "muscle_fiber_velocity"
    )
    np.testing.assert_almost_equal(
        float(small_func(activation=1, muscle_fiber_length=2, muscle_fiber_velocity=3)["output"]), 6.0
    )

    np.testing.assert_almost_equal(
        float(model.function_to_dm(my_small_function, activation=1, muscle_fiber_length=2, muscle_fiber_velocity=3)),
        6.0,
    )
