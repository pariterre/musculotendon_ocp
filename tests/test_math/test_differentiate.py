from musculotendon_ocp.math import compute_finitediff
import numpy as np


def test_compute_finitediff():
    # Test with a simple function
    def f(x):
        return x**2

    def df(x):
        return 2 * x

    t = np.linspace(0, 1, 100)
    x = f(t)
    dx_finitediff = compute_finitediff(x, t)
    dx_exact = df(t)

    assert np.allclose(dx_finitediff[1:-1], dx_exact[1:-1])

    # Make sure the finitediff is computed using phase independent
    assert np.isnan(dx_finitediff[0])
    assert np.isnan(dx_finitediff[-1])
