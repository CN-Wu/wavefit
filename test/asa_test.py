import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter
from wavefit.core.propagation import angular_spectrum_propagation
import matplotlib.pyplot as plt

def test_forward_backward():
    N = 256
    dx = 5e-6
    lam = 532e-9
    x = (cp.arange(N)-N//2)*dx
    X,Y = cp.meshgrid(x,x)
    R = cp.sqrt(X**2 + Y**2)
    zernike_radius = 0.15e-3
    rho = R / zernike_radius
    theta = cp.arctan2(Y, X)
    # Gaussian with random smooth phase
    amp = cp.exp(-((X**2+Y**2)/(0.1e-3)**2))
    astig = (rho**2) * cp.cos(2 * theta)
    # spherical: use rho^4 as simple surrogate
    sph = rho**4
    rng = cp.random.RandomState(1)
    phi = gaussian_filter(rng.standard_normal((N,N)), sigma=3)
    E0 = amp * cp.exp(1j * (phi + 1*astig + 0.5*sph))
    z =4e-2

    plt.imshow(cp.asnumpy(cp.abs(E0)), cmap='jet')
    plt.colorbar()
    plt.show()

    Ez = angular_spectrum_propagation(E0, dx, lam, z, pad_factor=2)

    plt.imshow(cp.asnumpy(cp.abs(Ez)), cmap='jet')
    plt.colorbar()
    plt.show()

    Eback = angular_spectrum_propagation(Ez, dx, lam, -z, pad_factor=2)

    plt.imshow(cp.asnumpy(cp.abs(Eback)), cmap='jet')
    plt.colorbar()
    plt.show()

    rel_err = cp.linalg.norm(Eback - E0) / (cp.linalg.norm(E0) + 1e-16)
    print("forward-backward relative error:", float(rel_err))
    return float(rel_err)


if __name__ == "__main__":
    err1 = test_forward_backward()
    # assert err1 < 1e-6
    print(f"Error 1: {err1}.")