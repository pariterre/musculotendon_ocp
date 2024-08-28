from musculotendon_ocp.math import precise_rk4, precise_rk45
import numpy as np


def test_precise_rk4():
    def dynamics(t, y):
        return y

    y0 = np.array([1])
    t_span = (0, 1)
    dt = 0.01
    t, y = precise_rk4(dynamics, y0, t_span, dt)

    # Make sure the time vector is uniform
    assert t.shape == (101,)
    assert np.allclose(t, np.linspace(0, 1.0, 101))

    # Make sure the integration is correct
    assert np.allclose(y, np.exp(t))


def test_precise_rk45():
    def dynamics(t, y):
        return y

    y0 = np.array([1])
    t_span = (0, 1)
    dt = 0.01
    t, y = precise_rk45(dynamics, y0, t_span, dt)

    # Make sure the time vector is uniform
    assert t.shape == (101,)
    assert np.allclose(t, np.linspace(0, 1.0, 101))

    # Make sure the integration is correct
    assert np.allclose(y, np.exp(t))
